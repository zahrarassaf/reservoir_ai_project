"""
cnn_reservoir.py
SPE9 CNN baseline for spatial permeability reconstruction.
"""

from __future__ import annotations

import copy
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


# ============================================================
# CONFIG
# ============================================================

SEED = 42
N_SEEDS = 5

EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 16
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-3

GRID_SHAPE_KJI = (15, 25, 24)
PATCH_KJ = 5
HALF_PATCH = PATCH_KJ // 2

# Patch supports along J:
#   center range = [HALF_PATCH, J_SIZE - HALF_PATCH - 1] = [2, 22]
#   support of center c = [c - 2, c + 2] ⊂ [0, 24]
#
# Split (pairwise disjoint supports):
#   train support = 0..11
#   val support   = 12..18
#   test support  = 19..24
TRAIN_J = range(2, 10)     # centers 2..9
VAL_J = range(14, 17)      # centers 14..16
TEST_J = range(21, 23)     # centers 21..22

DATA_DIR = Path("data")
PERM_FILE = DATA_DIR / "PERMVALUES.DATA"
SPE9_FILE = DATA_DIR / "SPE9.DATA"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# ECLIPSE-DATA PARSING
# ============================================================

def strip_comments(text: str) -> str:
    return re.sub(r"--[^\n\r]*", "", text)


def extract_keyword_block(text: str, keyword: str) -> str:
    clean = strip_comments(text)

    match = re.search(
        rf"\b{re.escape(keyword)}\b(.*?)/",
        clean,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if match is None:
        raise ValueError(f"Keyword '{keyword}' was not found.")

    return match.group(1)


def expand_eclipse_values(block: str) -> np.ndarray:
    tokens = block.replace("\n", " ").replace("\r", " ").split()

    values: List[float] = []

    for token in tokens:
        token = token.strip().rstrip(",")

        if not token:
            continue

        if "*" in token:
            parts = token.split("*")

            if len(parts) != 2:
                raise ValueError(f"Unsupported Eclipse token: {token}")

            count_text, value_text = parts

            if value_text == "":
                raise ValueError(
                    f"Unspecified Eclipse value is not supported: {token}"
                )

            count = int(count_text)
            value = float(value_text)

            if count < 0:
                raise ValueError(f"Negative repeat count: {token}")

            values.extend([value] * count)
        else:
            values.append(float(token))

    return np.asarray(values, dtype=np.float32)


def load_spe9_permx(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing permeability file: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    block = extract_keyword_block(text, "PERMX")
    values = expand_eclipse_values(block)

    expected = int(np.prod(GRID_SHAPE_KJI))

    if values.size != expected:
        raise ValueError(
            f"PERMX contains {values.size} values; expected {expected} "
            f"for grid {GRID_SHAPE_KJI}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("PERMX contains NaN or infinite values.")

    if np.any(values <= 0):
        raise ValueError("PERMX must be strictly positive for log transform.")

    return values.reshape(GRID_SHAPE_KJI)


def load_spe9_poro(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing SPE9 data file: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    block = extract_keyword_block(text, "PORO")
    values = expand_eclipse_values(block)

    expected = int(np.prod(GRID_SHAPE_KJI))

    if values.size != expected:
        raise ValueError(
            f"PORO contains {values.size} values; expected {expected}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError("PORO contains NaN or infinite values.")

    if np.any((values <= 0) | (values >= 1)):
        raise ValueError("PORO contains values outside the physical range (0, 1).")

    return values.reshape(GRID_SHAPE_KJI)


def report_spe9_grid(permx: np.ndarray, poro: np.ndarray) -> None:
    if permx.shape != GRID_SHAPE_KJI:
        raise RuntimeError(f"Unexpected PERMX shape: {permx.shape}")

    if poro.shape != GRID_SHAPE_KJI:
        raise RuntimeError(f"Unexpected PORO shape: {poro.shape}")

    poro_layer_const = all(
        np.allclose(poro[k], poro[k].flat[0], atol=1e-6)
        for k in range(poro.shape[0])
    )

    perm_layer_means = permx.mean(axis=(1, 2))

    print("SPE9 grid checks:")
    print(f"  PORO constant within each K layer: {poro_layer_const}")
    print(
        f"  PERMX layer-mean range: "
        f"[{perm_layer_means.min():.3f}, {perm_layer_means.max():.3f}]"
    )
    print(
        "  Cell ordering of PERMX and PORO follows the SPE9 file "
        "specification; it is not verified programmatically."
    )


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_grid() -> Tuple[np.ndarray, np.ndarray]:
    permx = load_spe9_permx(PERM_FILE)
    poro = load_spe9_poro(SPE9_FILE)

    report_spe9_grid(permx, poro)

    print(f"PERMX shape (K,J,I): {permx.shape}")
    print(f"PORO  shape (K,J,I): {poro.shape}")

    return permx.astype(np.float32), poro.astype(np.float32)


# ============================================================
# DATASET
# ============================================================

class ReservoirPatchDataset(Dataset):
    def __init__(
        self,
        permx_kji: np.ndarray,
        poro_kji: np.ndarray,
        half_patch: int = HALF_PATCH,
    ) -> None:
        if permx_kji.shape != GRID_SHAPE_KJI:
            raise ValueError(
                f"Unexpected PERMX shape: {permx_kji.shape}; "
                f"expected {GRID_SHAPE_KJI}."
            )

        if poro_kji.shape != GRID_SHAPE_KJI:
            raise ValueError(
                f"Unexpected PORO shape: {poro_kji.shape}; "
                f"expected {GRID_SHAPE_KJI}."
            )

        self.permx = permx_kji
        self.poro = poro_kji
        self.half_patch = half_patch

        self.samples: List[Tuple[int, int]] = []

        k_size, j_size, _ = GRID_SHAPE_KJI

        for k in range(half_patch, k_size - half_patch):
            for j in range(half_patch, j_size - half_patch):
                self.samples.append((k, j))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self._build(index, augment=False)

    def _build(
        self, index: int, augment: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        k, j = self.samples[index]

        k0 = k - self.half_patch
        k1 = k + self.half_patch + 1
        j0 = j - self.half_patch
        j1 = j + self.half_patch + 1

        if j0 < 0 or j1 > GRID_SHAPE_KJI[1]:
            raise RuntimeError(
                f"Patch J range [{j0}, {j1}) out of bounds for center J={j}."
            )

        if k0 < 0 or k1 > GRID_SHAPE_KJI[0]:
            raise RuntimeError(
                f"Patch K range [{k0}, {k1}) out of bounds for center K={k}."
            )

        perm_patch = self.permx[k0:k1, j0:j1, :].transpose(2, 1, 0)
        poro_patch = self.poro[k0:k1, j0:j1, :].transpose(2, 1, 0)

        log_perm_patch = np.log10(perm_patch).astype(np.float32)

        center_y = self.half_patch
        center_z = self.half_patch

        observation_mask = np.ones_like(log_perm_patch, dtype=np.float32)
        observation_mask[:, center_y, center_z] = 0.0

        masked_log_perm = log_perm_patch.copy()
        masked_log_perm[:, center_y, center_z] = 0.0

        patch = np.stack(
            [
                masked_log_perm,
                poro_patch,
                observation_mask,
            ],
            axis=0,
        ).astype(np.float32)

        target_value = np.log10(self.permx[k, j, :].mean())

        if augment and random.random() > 0.5:
            patch = np.flip(patch, axis=2).copy()

        target = np.array([float(target_value)], dtype=np.float32)

        return torch.from_numpy(patch), torch.from_numpy(target)


# ============================================================
# SPATIAL BLOCK SPLIT
# ============================================================

def make_spatial_indices(
    dataset: ReservoirPatchDataset,
) -> Tuple[List[int], List[int], List[int]]:
    coord_to_index = {coord: i for i, coord in enumerate(dataset.samples)}

    def collect(js: Sequence[int]) -> List[int]:
        result = []

        for k in range(HALF_PATCH, GRID_SHAPE_KJI[0] - HALF_PATCH):
            for j in js:
                coord = (k, j)

                if coord in coord_to_index:
                    result.append(coord_to_index[coord])

        return result

    train_idx = collect(TRAIN_J)
    val_idx = collect(VAL_J)
    test_idx = collect(TEST_J)

    if not train_idx or not val_idx or not test_idx:
        raise RuntimeError("One of the spatial splits is empty.")

    def j_support(coords) -> set:
        cols = set()
        for _, j in coords:
            cols.update(range(j - HALF_PATCH, j + HALF_PATCH + 1))
        return cols

    train_coords = {dataset.samples[i] for i in train_idx}
    val_coords = {dataset.samples[i] for i in val_idx}
    test_coords = {dataset.samples[i] for i in test_idx}

    train_j = j_support(train_coords)
    val_j = j_support(val_coords)
    test_j = j_support(test_coords)

    if train_j & val_j:
        raise RuntimeError(
            f"Train/val J-column leakage: {sorted(train_j & val_j)}"
        )
    if train_j & test_j:
        raise RuntimeError(
            f"Train/test J-column leakage: {sorted(train_j & test_j)}"
        )
    if val_j & test_j:
        raise RuntimeError(
            f"Val/test J-column leakage: {sorted(val_j & test_j)}"
        )

    print(f"Train samples: {len(train_idx)}  (J centers = {list(TRAIN_J)})")
    print(f"Val samples:   {len(val_idx)}  (J centers = {list(VAL_J)})")
    print(f"Test samples:  {len(test_idx)}  (J centers = {list(TEST_J)})")
    print(f"Train J-columns: {sorted(train_j)}")
    print(f"Val   J-columns: {sorted(val_j)}")
    print(f"Test  J-columns: {sorted(test_j)}")

    return train_idx, val_idx, test_idx


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_input(
    dataset: ReservoirPatchDataset,
    train_indices: Sequence[int],
) -> Tuple[float, float, float, float]:
    train_j_centers = sorted({dataset.samples[i][1] for i in train_indices})

    j_lo = min(train_j_centers) - HALF_PATCH
    j_hi = max(train_j_centers) + HALF_PATCH + 1

    perm_block = dataset.permx[:, j_lo:j_hi, :]
    poro_block = dataset.poro[:, j_lo:j_hi, :]

    log_perm_block = np.log10(perm_block)

    perm_mean = float(log_perm_block.mean())
    perm_std = float(log_perm_block.std())
    poro_mean = float(poro_block.mean())
    poro_std = float(poro_block.std())

    if perm_std < 1e-8 or poro_std < 1e-8:
        raise ValueError("Training input has near-zero variance.")

    dataset.input_stats = (perm_mean, perm_std, poro_mean, poro_std)

    return dataset.input_stats


def normalize_target(
    dataset: ReservoirPatchDataset,
    train_indices: Sequence[int],
) -> StandardScaler:
    train_targets = np.asarray(
        [
            np.log10(dataset.permx[k, j, :].mean())
            for k, j in (dataset.samples[i] for i in train_indices)
        ],
        dtype=np.float32,
    ).reshape(-1, 1)

    scaler = StandardScaler()
    scaler.fit(train_targets)

    dataset.target_scaler = scaler

    return scaler


# ============================================================
# NORMALIZED DATASET
# ============================================================

class NormalizedReservoirDataset(Dataset):
    def __init__(
        self,
        base_dataset: ReservoirPatchDataset,
        indices: Sequence[int],
        augment: bool = False,
    ) -> None:
        self.base_dataset = base_dataset
        self.indices = list(indices)
        self.augment = augment

        if not hasattr(base_dataset, "input_stats"):
            raise RuntimeError("Input normalization statistics are missing.")

        if not hasattr(base_dataset, "target_scaler"):
            raise RuntimeError("Target scaler is missing.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        original_idx = self.indices[idx]

        patch, target = self.base_dataset._build(
            original_idx, augment=self.augment
        )

        perm_mean, perm_std, poro_mean, poro_std = self.base_dataset.input_stats

        patch = patch.numpy().copy()

        patch[0] = (patch[0] - perm_mean) / perm_std
        patch[1] = (patch[1] - poro_mean) / poro_std

        target_np = self.base_dataset.target_scaler.transform(
            target.numpy().reshape(1, -1)
        )[0].astype(np.float32)

        return (
            torch.from_numpy(patch.astype(np.float32)),
            torch.from_numpy(target_np),
        )


# ============================================================
# MODEL
# ============================================================

class CNNReservoir(nn.Module):
    def __init__(self, i_pool: int = 2) -> None:
        super().__init__()

        self.i_pool = i_pool

        self.features = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.GroupNorm(2, 8),
            nn.ReLU(inplace=True),

            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),

            nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1)),

            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.GroupNorm(4, 16),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool3d((i_pool, PATCH_KJ, PATCH_KJ)),
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * i_pool * PATCH_KJ * PATCH_KJ, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.regressor(x)


# ============================================================
# TRAIN / EVALUATE
# ============================================================

CRITERION = nn.HuberLoss(delta=1.0)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: StandardScaler,
) -> Dict[str, float]:
    model.eval()

    losses: List[float] = []
    predictions: List[np.ndarray] = []
    targets: List[np.ndarray] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(x)

            loss = CRITERION(pred, y)
            losses.append(float(loss.item()))

            predictions.append(pred.cpu().numpy())
            targets.append(y.cpu().numpy())

    pred_scaled = np.concatenate(predictions, axis=0)
    true_scaled = np.concatenate(targets, axis=0)

    pred_log = target_scaler.inverse_transform(pred_scaled).ravel()
    true_log = target_scaler.inverse_transform(true_scaled).ravel()

    pred_perm = 10.0 ** pred_log
    true_perm = 10.0 ** true_log

    metrics = {
        "loss": float(np.mean(losses)),
        "log_mae": float(mean_absolute_error(true_log, pred_log)),
        "log_rmse": float(np.sqrt(mean_squared_error(true_log, pred_log))),
        "log_r2": float(r2_score(true_log, pred_log)),
        "perm_mae": float(mean_absolute_error(true_perm, pred_perm)),
        "perm_rmse": float(np.sqrt(mean_squared_error(true_perm, pred_perm))),
        "perm_r2": float(r2_score(true_perm, pred_perm)),
    }

    metrics["_pred_log"] = pred_log
    metrics["_true_log"] = true_log
    metrics["_pred_perm"] = pred_perm
    metrics["_true_perm"] = true_perm

    return metrics


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    target_scaler: StandardScaler,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "val_log_rmse": [],
        "val_log_r2": [],
        "val_perm_rmse": [],
        "val_perm_r2": [],
    }

    best_val_loss = float("inf")
    best_state = None
    epochs_without_improvement = 0

    model.to(DEVICE)

    for epoch in range(1, EPOCHS + 1):
        model.train()

        train_losses: List[float] = []

        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            pred = model(x)
            loss = CRITERION(pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses))

        val_metrics = evaluate(model, val_loader, target_scaler)
        val_loss = val_metrics["loss"]

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_log_rmse"].append(val_metrics["log_rmse"])
        history["val_log_r2"].append(val_metrics["log_r2"])
        history["val_perm_rmse"].append(val_metrics["perm_rmse"])
        history["val_perm_r2"].append(val_metrics["perm_r2"])

        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} | "
            f"val_loss={val_loss:.5f} | "
            f"val_log_R2={val_metrics['log_r2']:.4f} | "
            f"val_perm_R2={val_metrics['perm_r2']:.4f} | "
            f"val_perm_RMSE={val_metrics['perm_rmse']:.4f} | "
            f"lr={current_lr:.2e}"
        )

        if val_loss < best_val_loss - 1e-7:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is None:
        raise RuntimeError("No best model state was recorded.")

    model.load_state_dict(best_state)

    return model, history


# ============================================================
# BASELINES
# ============================================================

def baseline_train_mean(
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    dataset: ReservoirPatchDataset,
) -> Dict[str, float]:
    train_targets = np.asarray(
        [
            np.log10(dataset.permx[k, j, :].mean())
            for k, j in (dataset.samples[i] for i in train_indices)
        ],
        dtype=np.float64,
    )

    test_targets = np.asarray(
        [
            np.log10(dataset.permx[k, j, :].mean())
            for k, j in (dataset.samples[i] for i in test_indices)
        ],
        dtype=np.float64,
    )

    prediction = np.full_like(test_targets, train_targets.mean())

    pred_perm = 10.0 ** prediction
    true_perm = 10.0 ** test_targets

    return {
        "log_mae": float(mean_absolute_error(test_targets, prediction)),
        "log_rmse": float(np.sqrt(mean_squared_error(test_targets, prediction))),
        "log_r2": float(r2_score(test_targets, prediction)),
        "perm_mae": float(mean_absolute_error(true_perm, pred_perm)),
        "perm_rmse": float(np.sqrt(mean_squared_error(true_perm, pred_perm))),
        "perm_r2": float(r2_score(true_perm, pred_perm)),
    }


def baseline_same_k_train_mean(
    train_indices: Sequence[int],
    test_indices: Sequence[int],
    dataset: ReservoirPatchDataset,
) -> Dict[str, float]:
    train_by_k: Dict[int, List[float]] = {}

    for i in train_indices:
        k, j = dataset.samples[i]
        train_by_k.setdefault(k, []).append(
            float(np.log10(dataset.permx[k, j, :].mean()))
        )

    global_mean = float(
        np.mean([v for vs in train_by_k.values() for v in vs])
    )

    preds = []
    trues = []

    for i in test_indices:
        k, j = dataset.samples[i]

        if k in train_by_k and train_by_k[k]:
            preds.append(float(np.mean(train_by_k[k])))
        else:
            preds.append(global_mean)

        trues.append(float(np.log10(dataset.permx[k, j, :].mean())))

    preds = np.asarray(preds, dtype=np.float64)
    trues = np.asarray(trues, dtype=np.float64)

    pred_perm = 10.0 ** preds
    true_perm = 10.0 ** trues

    return {
        "log_mae": float(mean_absolute_error(trues, preds)),
        "log_rmse": float(np.sqrt(mean_squared_error(trues, preds))),
        "log_r2": float(r2_score(trues, preds)),
        "perm_mae": float(mean_absolute_error(true_perm, pred_perm)),
        "perm_rmse": float(np.sqrt(mean_squared_error(true_perm, pred_perm))),
        "perm_r2": float(r2_score(true_perm, pred_perm)),
    }


# ============================================================
# PLOTS
# ============================================================

def save_training_plot(history: Dict[str, List[float]]) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Huber loss")
    plt.plot(history["val_loss"], label="Validation Huber loss")
    plt.xlabel("Epoch")
    plt.ylabel("Huber loss")
    plt.title("CNN training history (best validation seed)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_history.png", dpi=250)
    plt.close()


def save_prediction_plot(
    true_log: np.ndarray,
    pred_log: np.ndarray,
    true_perm: np.ndarray,
    pred_perm: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(true_log, pred_log, alpha=0.8)
    lo = min(true_log.min(), pred_log.min())
    hi = max(true_log.max(), pred_log.max())
    axes[0].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[0].set_xlabel("True log10(mean PERMX)")
    axes[0].set_ylabel("Predicted log10(mean PERMX)")
    axes[0].set_title("Best-seed test prediction - log scale")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(true_perm, pred_perm, alpha=0.8)
    lo = min(true_perm.min(), pred_perm.min())
    hi = max(true_perm.max(), pred_perm.max())
    axes[1].plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)
    axes[1].set_xlabel("True mean PERMX")
    axes[1].set_ylabel("Predicted mean PERMX")
    axes[1].set_title("Best-seed test prediction - original scale")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "test_prediction.png", dpi=250)
    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 72)
    print("SPE9 CNN - SPATIAL BLOCK-HOLDOUT PERMEABILITY PREDICTION")
    print("=" * 72)
    print(f"Device: {DEVICE}")

    print("\n[1/7] Loading real SPE9 data...")

    permx, poro = prepare_grid()

    print("\n[2/7] Building spatial dataset...")

    base_dataset = ReservoirPatchDataset(permx, poro)

    train_idx, val_idx, test_idx = make_spatial_indices(base_dataset)

    print("\n[3/7] Fitting preprocessing on training region only...")

    input_stats = normalize_input(base_dataset, train_idx)
    target_scaler = normalize_target(base_dataset, train_idx)

    print(
        "Input statistics:\n"
        f"  log10(PERMX): mean={input_stats[0]:.5f}, std={input_stats[1]:.5f}\n"
        f"  PORO:         mean={input_stats[2]:.5f}, std={input_stats[3]:.5f}"
    )

    print(
        "Target statistics (log10):\n"
        f"  mean={target_scaler.mean_[0]:.5f}, "
        f"std={target_scaler.scale_[0]:.5f}"
    )

    train_dataset = NormalizedReservoirDataset(base_dataset, train_idx, augment=True)
    val_dataset = NormalizedReservoirDataset(base_dataset, val_idx, augment=False)
    test_dataset = NormalizedReservoirDataset(base_dataset, test_idx, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    print("\n[4/7] Evaluating baselines...")

    mean_metrics = baseline_train_mean(train_idx, test_idx, base_dataset)
    same_k_metrics = baseline_same_k_train_mean(train_idx, test_idx, base_dataset)

    print(
        f"Train-mean baseline        | "
        f"log_R2={mean_metrics['log_r2']:.4f} | "
        f"log_RMSE={mean_metrics['log_rmse']:.4f} | "
        f"perm_R2={mean_metrics['perm_r2']:.4f}"
    )
    print(
        f"Same-K train-mean baseline | "
        f"log_R2={same_k_metrics['log_r2']:.4f} | "
        f"log_RMSE={same_k_metrics['log_rmse']:.4f} | "
        f"perm_R2={same_k_metrics['perm_r2']:.4f}"
    )

    print("\n[5/7] Training across seeds...")

    all_metrics: Dict[str, List[float]] = {
        "log_mae": [],
        "log_rmse": [],
        "log_r2": [],
        "perm_mae": [],
        "perm_rmse": [],
        "perm_r2": [],
    }

    best_overall_state = None
    best_overall_val = float("inf")
    best_history = None
    best_seed = None

    for run in range(N_SEEDS):
        seed = SEED + run

        print(f"\n--- Run {run + 1}/{N_SEEDS} (seed={seed}) ---")

        set_seed(seed)
        model = CNNReservoir()

        model, history = train_model(
            model,
            train_loader,
            val_loader,
            target_scaler,
        )

        metrics = evaluate(model, test_loader, target_scaler)

        for key in all_metrics:
            all_metrics[key].append(metrics[key])

        val_best = min(history["val_loss"])

        if val_best < best_overall_val:
            best_overall_val = val_best
            best_overall_state = copy.deepcopy(model.state_dict())
            best_history = history
            best_seed = seed

        print(
            f"Run {run + 1} test | "
            f"log_R2={metrics['log_r2']:.4f} | "
            f"perm_R2={metrics['perm_r2']:.4f} | "
            f"log_RMSE={metrics['log_rmse']:.4f}"
        )

    print("\n[6/7] Aggregating metrics across seeds...")

    print("\n" + "=" * 72)
    print("CNN TEST METRICS (mean +/- std over initialization seeds)")
    print("=" * 72)

    for key in all_metrics:
        arr = np.asarray(all_metrics[key])
        print(f"  {key:10s} : {arr.mean():.6f} +/- {arr.std():.6f}")

    print("\nBaselines (single deterministic value on the same test set):")
    print(
        f"  train-mean        | log_R2={mean_metrics['log_r2']:.4f} | "
        f"perm_R2={mean_metrics['perm_r2']:.4f}"
    )
    print(
        f"  same-K train-mean | log_R2={same_k_metrics['log_r2']:.4f} | "
        f"perm_R2={same_k_metrics['perm_r2']:.4f}"
    )
    print("=" * 72)

    if best_overall_state is None:
        raise RuntimeError("No model state was retained.")

    torch.save(
        {
            "model_state_dict": best_overall_state,
            "target_scaler_mean": target_scaler.mean_,
            "target_scaler_scale": target_scaler.scale_,
            "input_stats": input_stats,
            "grid_shape_kji": GRID_SHAPE_KJI,
            "patch_kj": PATCH_KJ,
            "train_j": list(TRAIN_J),
            "val_j": list(VAL_J),
            "test_j": list(TEST_J),
            "seed": SEED,
            "n_seeds": N_SEEDS,
            "best_seed": best_seed,
            "best_val_loss": best_overall_val,
            "criterion": "HuberLoss(delta=1.0)",
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
        },
        OUTPUT_DIR / "best_cnn_reservoir.pt",
    )

    print("\n[7/7] Saving plots and best-seed evaluation...")

    set_seed(best_seed if best_seed is not None else SEED)
    model = CNNReservoir()
    model.load_state_dict(best_overall_state)
    model.to(DEVICE)

    final_metrics = evaluate(model, test_loader, target_scaler)

    save_training_plot(best_history)
    save_prediction_plot(
        final_metrics["_true_log"],
        final_metrics["_pred_log"],
        final_metrics["_true_perm"],
        final_metrics["_pred_perm"],
    )

    print("\nSaved:")
    print(f"  {OUTPUT_DIR / 'best_cnn_reservoir.pt'}")
    print(f"  {OUTPUT_DIR / 'training_history.png'}")
    print(f"  {OUTPUT_DIR / 'test_prediction.png'}")


if __name__ == "__main__":
    main()
