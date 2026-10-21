"""
cnn_dynamic_spe9.py  (v3 - final benchmark version)

One-step prediction of dynamic property increments (dP, dSO, dSG, dSW) at the
cell level from static fields (log10 PERMX, PORO) and the dynamic state at
time t in a 5x5x5 neighbourhood, evaluated with spatially blocked CV.

Why v3 (motivated by the v2 SPE9 run, where ExtraTrees >= CNN)
---------------------------------------------------------------
The v2 result was: with identical inputs ExtraTrees beat the CNN, and
`rate_persistence` (previous increment scaled by dt) alone reached R2(dSW)=0.91.
v3 removes plausible *causes* on the CNN side and strengthens the baselines.
Whether they matter is an empirical question answered by the ablations; none of
these changes is guaranteed to help.

  1. Loss/metric mismatch: Huber (delta=1 in sigma units) down-weights exactly the
     heavy-tailed front cells that dominate RMSE. Default loss is now MSE in
     sigma-normalised units (= the NRMSE objective); `--loss huber` is kept.
  2. No temporal history: the CNN only saw the state at t. New inputs: the last
     `--n-hist` increments (variants `prev`, `hist`).
  3. Direct vs. residual formulation: variant `resid` learns a correction on top
     of rate persistence (prediction = base + CNN output), so the CNN has to add
     value beyond the strongest simple predictor.
  4. Baselines: + HistGradientBoosting (sklearn, no extra dependency), history
     versions of Ridge/ET/HGB, and small validation-block hyper-parameter
     selection for ET/HGB so baselines are not under-tuned.
  5. Optional active-cell loss weighting (`--active-w`, default 0).

Other
-----
* Time-aware baselines, real sparsity (hidden columns are removed at train,
  validation and test), train-only normalisation, cluster bootstrap over J-columns,
  Holm-corrected paired tests, split-conformal intervals (calibrated on the
  validation block, evaluated on the test block), saturation-sum / bound
  violations, boundary-forced rollout, per-channel tables.
* `--main` selects the pre-specified main model for paired tests (default
  `cnn_resid`). Choose it BEFORE looking at test results.
* Resume: every (protocol, fold, method-group) is checkpointed in
  <out>/<protocol>/ckpt; re-running the same command continues where it stopped.
  Delete the ckpt folder after changing the code or data.
* CPU presets: `--fast` (about 10x cheaper) and `--quick` (smoke test).
* `--test-max N` evaluates every method on the same random subset of N test
  samples per fold (consistent across all methods).
* Several datasets: `--npz-names a.npz b.npz` (same key layout) runs each one
  into its own output sub-folder. A second reservoir model is strongly
  recommended for any generality claim.

Notes for the paper (limitations)
---------------------------------
* Only interior cells (distance >= patch//2 from every grid border) are predicted.
* Conformal calibration uses the validation block, which is also used for early
  stopping / baseline tuning, so intervals are slightly optimistic.
* D4 augmentation assumes horizontal isotropy (kx=ky, dx=dy) and no well-induced
  asymmetry within a patch; ablation `+no_aug` checks its effect.
* Rollout is boundary-forced (not a full simulator); its history lags start from
  the true states.

Usage
-----
    python cnn_dynamic_spe9.py --toy --skip-cnn            # pipeline smoke test (no torch needed)
    python cnn_dynamic_spe9.py --toy --quick               # tiny CNN smoke test
    python cnn_dynamic_spe9.py                             # auto: --fast on CPU, full on GPU
    python cnn_dynamic_spe9.py --tune-cnn                  # per-fold CNN lr/dropout search on the val block only
    python cnn_dynamic_spe9.py --protocols spatial spatial_temporal   # add the strict temporal-extrapolation protocol
    python test_no_leakage.py                              # leakage / causality unit tests (no torch needed)
    python cnn_dynamic_spe9.py --medium                    # overnight CPU run (several hours)
    python cnn_dynamic_spe9.py --full                      # force the heavy configuration
    python cnn_dynamic_spe9.py --tta                       # full run (GPU recommended; TTA is opt-in)
    python cnn_dynamic_spe9.py --protocols spatial spatial_temporal --rollout-horizon 10

Multi-scenario study (leave-one-scenario-out, "loso")
-----------------------------------------------------
    python cnn_dynamic_spe9.py --protocols loso --npz-names ref.npz s00.npz ... s11.npz --out-dir outputs_loso
  Scenarios = the SAME reservoir (identical static fields and well layout) under different well controls
  (injection rate, producer rate schedule, shut-in wells), produced by make_spe9_scenarios.py + OPM Flow +
  build_dataset.py (which writes the operating controls as the `ctrl` array). Held-out scenario s is the test
  set, scenario (s+1) mod S is used for early stopping / conformal calibration / tuning, all others train.
  `--loso-folds N` runs N evenly spaced held-out scenarios; `--sub-per-scen M` keeps M random samples per
  scenario (memory). Controls are given to the CNNs and to the tree baselines (same information). Outputs:
  metrics_dynamic.csv (pooled), loso_by_scenario.csv, loso_paired_by_scenario.csv (scenario-level tests).
  Not covered by this protocol: a different reservoir (second field) and multi-step roll-out.

Variant syntax: base[+mod+mod]. Bases: full, prev, hist, resid, static_only.
Mods: no_poro, no_time, no_center (not with resid), no_aug, no_phys, no_maskaug,
no_hist (history length 1), res (ResNet-style blocks). Example: --ablations full prev resid resid+no_aug

Input .npz keys: permx, poro (K,J,I); pressure, so, sg, sw (T,K,J,I); times (T,)
Optional: well_mask (K,J,I), 1 at well cells (extra input channel);
          ctrl (T,K,J,I,C), operating controls at every step (extra input channels; all scenarios or none);
          active (K,J,I) bool, ACTNUM mask for fields with inactive cells (e.g. Norne): inactive cells are never
          predicted, are zeroed in the inputs and flagged by an extra channel; fill them with finite values
          (permx=1, poro=0, states=0). `--sub-per-scen M` caps the samples per scenario/dataset (large grids).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import pickle
import platform
import dataclasses
import random
import sys
import time
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(os.cpu_count() or 1))   # silences the joblib/wmic warning

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as sps
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    HAS_TORCH = True
except ImportError:
    torch = None
    nn = None
    F = None
    HAS_TORCH = False

CODE_VERSION = "3.0"


# ============================================================
# PATHS / CONFIG
# ============================================================

_PROJECT_ROOT = Path(__file__).resolve().parent
if not (_PROJECT_ROOT / "data").exists():
    for _cand in [
        Path.home() / "Documents" / "Zoom" / "reservoir_ai_project",
        Path("C:/Users/Zahara/Documents/Zoom/reservoir_ai_project"),
    ]:
        if (_cand / "data").exists():
            _PROJECT_ROOT = _cand
            break

GRID_SHAPE_KJI = (15, 25, 24)
NPZ_NAME = "spe9_dynamic_dataset.npz"
CHANNELS_OUT = ["dP", "dSO", "dSG", "dSW"]

_BASE = dict(poro=True, dyn=True, prev=0, time=True, center=True, aug="d4", drop=True, phys=True, resid=False,
             arch="plain")
BASE_SPECS: Dict[str, Dict] = {
    "full": dict(_BASE),
    "prev": dict(_BASE, prev=1),
    "hist": dict(_BASE, prev=-1),                 # -1 -> cfg.n_hist lags
    "resid": dict(_BASE, prev=-1, resid=True),    # base = rate persistence
    "static_only": dict(_BASE, dyn=False),
}
MODS: Dict[str, Dict] = {
    "no_poro": dict(poro=False),
    "no_time": dict(time=False),
    "no_center": dict(center=False),
    "no_aug": dict(aug="none"),
    "no_phys": dict(phys=False),
    "no_maskaug": dict(drop=False),
    "no_hist": dict(prev=1),
    "res": dict(arch="res"),                      # two residual 3D blocks instead of a plain conv stack
}

BASELINES = [
    "persistence", "train_mean", "time_mean", "time_layer_mean", "rate_persistence",
    "ridge", "extratrees", "hgb",
    "ridge_prev", "extratrees_prev", "hgb_prev",
    "extratrees_hist", "hgb_hist",
]

PRETTY = {
    "persistence": "Persistence (zero increment)",
    "train_mean": "Train mean",
    "time_mean": "Time mean",
    "time_layer_mean": "Time x layer mean",
    "rate_persistence": "Rate persistence",
    "ridge": "Ridge",
    "extratrees": "ExtraTrees",
    "hgb": "HistGradBoost",
    "ridge_prev": "Ridge (+prev)",
    "extratrees_prev": "ExtraTrees (+prev)",
    "hgb_prev": "HistGradBoost (+prev)",
    "extratrees_hist": "ExtraTrees (+hist)",
    "hgb_hist": "HistGradBoost (+hist)",
    "cnn_full": "3D-CNN (state only)",
    "cnn_prev": "3D-CNN (+prev)",
    "cnn_hist": "3D-CNN (+hist)",
    "cnn_resid": "3D-CNN residual (+hist)",
    "cnn_static_only": "3D-CNN static-only",
}


def pretty(m: str) -> str:
    if m in PRETTY:
        return PRETTY[m]
    if m.startswith("cnn_"):
        return "3D-CNN " + m[4:].replace("_", " ")
    return m


PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
           "#F0E442", "#000000", "#999999", "#882255", "#44AA99", "#117733",
           "#AA4499", "#332288", "#DDCC77", "#88CCEE"]


@dataclass
class Config:
    data_dir: str = str(_PROJECT_ROOT / "data")
    npz_names: List[str] = field(default_factory=lambda: [NPZ_NAME])
    out_dir: str = str(_PROJECT_ROOT / "outputs_dynamic")
    toy: bool = False
    toy_inactive: bool = False     # toy data with inactive (ACTNUM = 0) cells, to test the Norne-style masking

    patch: int = 5
    n_folds: int = 5
    val_width: int = 2
    protocols: List[str] = field(default_factory=lambda: ["spatial"])
    t_cut_frac: float = 0.7

    seed: int = 42
    n_ens: int = 5
    epochs: int = 60
    patience: int = 10
    batch_size: int = 128
    epoch_samples: int = 20000
    lr: float = 5e-4
    weight_decay: float = 1e-3
    dropout: float = 0.15
    width: int = 16
    max_val: int = 4000
    baseline_max_train: int = 30000
    enforce_sum: bool = True
    tta: bool = False            # 8x D4 test-time augmentation: use on GPU (--tta)
    loss: str = "mse"            # mse | huber
    active_w: float = 0.0
    n_hist: int = 2

    sparsity: List[float] = field(default_factory=lambda: [0.0, 0.25, 0.5])
    train_drop_max: float = 0.4
    ablations: List[str] = field(default_factory=lambda: [
        "full", "prev", "hist", "resid", "resid+no_hist", "resid+no_aug", "resid+no_phys", "static_only"])
    alpha: float = 0.10
    n_boot: int = 2000
    ref: str = "time_layer_mean"
    main: str = "cnn_resid"

    test_max: int = 0
    rollout_horizon: int = 10      # multi-step roll-out check (spatial protocol); 0 disables
    rollout_starts: int = 3

    resume: bool = True
    threads: int = 0
    skip_cnn: bool = False
    quick: bool = False
    fast: bool = False
    medium: bool = False         # overnight CPU preset (--medium): ~5 members, 40 epochs, width 16, 3 variants
    tune_cnn: bool = False         # small per-fold CNN hyper-parameter search on the val block only (--tune-cnn)
    full: bool = True           # force the heavy configuration even on CPU (--full)
    sub_per_scen: int = 0          # loso only: random subset of samples kept per scenario (0 = all; memory control)
    loso_folds: int = 0            # loso only: number of held-out scenarios actually run (0 = all, evenly spaced subset)


def get_device():
    if HAS_TORCH and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu") if HAS_TORCH else None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if HAS_TORCH:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_spec(name: str, cfg: Config) -> Dict:
    parts = name.split("+")
    if parts[0] not in BASE_SPECS:
        raise ValueError(f"Unknown variant base '{parts[0]}' (choose from {sorted(BASE_SPECS)}).")
    spec = dict(BASE_SPECS[parts[0]])
    for mod in parts[1:]:
        if mod not in MODS:
            raise ValueError(f"Unknown modifier '{mod}' (choose from {sorted(MODS)}).")
        spec.update(MODS[mod])
    if spec["prev"] == -1:
        spec["prev"] = max(1, cfg.n_hist)
    if spec["resid"] and not (spec["dyn"] and spec["center"] and spec["prev"] >= 1):
        raise ValueError(f"Variant '{name}': residual models need dynamic input, visible centre and history.")
    return spec


def cnn_name(variant: str) -> str:
    return "cnn_" + variant.replace("+", "_")


# ============================================================
# TOY DATASET (pipeline smoke test only -- NOT physics)
# ============================================================

def make_toy_npz(path: Path, T: int = 14, seed: int = 0, shape=GRID_SHAPE_KJI, inactive: bool = False) -> None:
    rng = np.random.default_rng(seed)
    K, J, I = shape
    white = rng.standard_normal(shape)
    fr = np.meshgrid(*[np.fft.fftfreq(n) for n in shape], indexing="ij")
    q = sum((2 * np.pi * c * f) ** 2 for c, f in zip((1.5, 4.0, 5.0), fr))
    grf = np.fft.ifftn(np.fft.fftn(white) * np.sqrt((1 + q) ** -2.0)).real
    grf = (grf - grf.mean()) / grf.std()
    logk = 1.5 + rng.normal(0, 0.3, K)[:, None, None] + 0.5 * grf
    permx = 10 ** logk
    poro = np.repeat(np.linspace(0.08, 0.2, K), J * I).reshape(shape)
    kk, jj, ii = np.meshgrid(np.arange(K), np.arange(J), np.arange(I), indexing="ij")
    dist = np.sqrt(jj ** 2 + ii ** 2 + (0.5 * (kk - 3)) ** 2)
    speed = np.sqrt(10 ** (logk - 1.5))
    times = np.linspace(0, 400, T)
    sw, so, sg, pr = [], [], [], []
    for t in times:
        w = 0.2 + 0.6 / (1 + np.exp(-(speed * t / 40.0 - dist) / 1.5))
        g = 0.05 * (1 - np.exp(-t / 120.0)) * np.exp(-kk / 3.0)
        sw.append(w)
        sg.append(g)
        so.append(1 - w - g)
        pr.append(3600 - 900 * (1 - np.exp(-t / 150.0)) * np.exp(-dist / 15.0) + 0.5 * rng.standard_normal(shape))
    extra = {}
    pr, so, sg, sw = (np.stack(a_).astype(np.float32) for a_ in (pr, so, sg, sw))
    permx, poro = permx.astype(np.float32), poro.astype(np.float32)
    if inactive:                                     # ACTNUM-like mask: random blobs + one dead layer strip; neutral filler
        act = rng.random(shape) > 0.12
        act &= ~((kk < 2) & (jj < 6))
        act[:, 3:6, 10:14] = False
        permx = np.where(act, permx, 1.0).astype(np.float32)
        poro = np.where(act, poro, 0.0).astype(np.float32)
        pr = np.where(act[None], pr, 0.0).astype(np.float32)
        sw = np.where(act[None], sw, 1.0).astype(np.float32)
        so = np.where(act[None], so, 0.0).astype(np.float32)
        sg = np.where(act[None], sg, 0.0).astype(np.float32)
        extra["active"] = act
    np.savez_compressed(path, permx=permx, poro=poro, pressure=pr, so=so, sg=sg, sw=sw,
                        times=times.astype(np.float32), **extra)


def make_toy_multi(outdir: Path, n_scen: int = 4, T: int = 14, seed: int = 0, shape=GRID_SHAPE_KJI) -> List[Path]:
    """Toy multi-scenario set: same static fields + well layout, different controls that really drive the dynamics.
    Pipeline smoke test only -- NOT physics."""
    outdir.mkdir(parents=True, exist_ok=True)
    K, J, I = shape
    base = outdir / "_base.npz"
    make_toy_npz(base, T=T, seed=seed, shape=shape)
    b = np.load(base)
    wm = np.zeros(shape, np.float32)
    wm[:, 0, 0] = 1
    wm[:, J - 1, I - 1] = 1
    wm[:, J // 2, I // 2] = 1
    kk, jj, ii = np.meshgrid(np.arange(K), np.arange(J), np.arange(I), indexing="ij")
    dist = np.sqrt(jj ** 2 + ii ** 2 + (0.5 * (kk - 3)) ** 2)
    speed = np.sqrt(10 ** (np.log10(b["permx"]) - 1.5))
    rng = np.random.default_rng(seed + 1)
    paths = []
    for s in range(n_scen):
        inj = float(rng.choice([0.4, 0.7, 1.0, 1.4]))
        prod = rng.choice([0.4, 0.8, 1.0], size=3)
        ctrl = np.zeros((T, K, J, I, 2), np.float32)
        times = np.linspace(0, 400, T)
        sw, so, sg, pr = [], [], [], []
        cum = 0.0
        for ti, t in enumerate(times):
            ph = min(2, ti * 3 // T)
            ctrl[ti, :, 0, 0, 0] = inj
            ctrl[ti, :, J - 1, I - 1, 1] = prod[ph]
            ctrl[ti, :, J // 2, I // 2, 1] = prod[ph]
            if ti > 0:
                cum += inj * (times[ti] - times[ti - 1])
            w = 0.2 + 0.6 / (1 + np.exp(-(speed * cum / 40.0 - dist) / 1.5))
            g = 0.05 * (1 - np.exp(-t / 120.0)) * np.exp(-kk / 3.0) * (0.5 + prod[ph])
            sw.append(w)
            sg.append(g)
            so.append(1 - w - g)
            pr.append(3600 - 900 * inj * (1 - np.exp(-t / 150.0)) * np.exp(-dist / 15.0) + 0.5 * rng.standard_normal(shape))
        pth = outdir / f"toy_s{s:02d}.npz"
        np.savez_compressed(pth, permx=b["permx"], poro=b["poro"], pressure=np.stack(pr).astype(np.float32),
                            so=np.stack(so).astype(np.float32), sg=np.stack(sg).astype(np.float32),
                            sw=np.stack(sw).astype(np.float32), times=times.astype(np.float32),
                            well_mask=wm, ctrl=ctrl)
        paths.append(pth)
    return paths


# ============================================================
# DATA
# ============================================================

class DynData:
    """One simulation run (single npz) or several scenarios of the SAME reservoir (list of npz, same grid/static
    fields/well layout, different well controls). Scenarios are stacked along the time axis; transitions that
    would cross a scenario boundary are never sampled, and every history/lag lookup is clamped to the start of
    its own scenario (see `Source.start`)."""

    def __init__(self, npz_path, patch: int, seed: int, sub_per_scen: int = 0) -> None:
        paths = [Path(q) for q in npz_path] if isinstance(npz_path, (list, tuple)) else [Path(npz_path)]
        ds = [np.load(q) for q in paths]
        need = {"permx", "poro", "pressure", "so", "sg", "sw", "times"}
        for q, d in zip(paths, ds):
            if not need <= set(d.files):
                raise KeyError(f"{q.name}: npz is missing keys: {sorted(need - set(d.files))}")
        d0 = ds[0]
        permx = d0["permx"].astype(np.float64)
        poro = d0["poro"].astype(np.float32)
        self.nS = len(ds)
        self.T_loc = int(d0["pressure"].shape[0])
        for q, d in zip(paths, ds):
            if d["pressure"].shape[0] != self.T_loc:
                raise ValueError(f"{q.name}: all scenarios must have the same number of report steps")
            if not (np.allclose(d["permx"], d0["permx"]) and np.allclose(d["poro"], d0["poro"])):
                raise ValueError(f"{q.name}: static fields differ between scenarios (multi-scenario mode = same reservoir)")
            if ("well_mask" in d.files) != ("well_mask" in d0.files) or \
                    ("well_mask" in d.files and not np.array_equal(d["well_mask"], d0["well_mask"])):
                raise ValueError(f"{q.name}: well_mask differs between scenarios")
            if ("ctrl" in d.files) != ("ctrl" in d0.files):
                raise ValueError(f"{q.name}: either all or none of the scenarios must provide `ctrl`")
            if ("active" in d.files) != ("active" in d0.files) or \
                    ("active" in d.files and not np.array_equal(d["active"], d0["active"])):
                raise ValueError(f"{q.name}: `active` (ACTNUM) mask differs between scenarios")
        D = np.concatenate([np.stack([d["pressure"], d["so"], d["sg"], d["sw"]], axis=-1) for d in ds],
                           axis=0).astype(np.float32)                                    # (S*T,K,J,I,4)
        self.times = np.concatenate([d["times"].astype(np.float64) for d in ds])        # within-scenario time
        self.D = D
        self.T = D.shape[0]
        self.K, self.J, self.I = permx.shape
        self.sc_of_step = np.repeat(np.arange(self.nS), self.T_loc)
        self.start = (self.sc_of_step * self.T_loc).astype(np.int64)                    # first step of own scenario
        self.tmax = float(self.times.max())
        diffs = [np.diff(self.times[s_ * self.T_loc:(s_ + 1) * self.T_loc]) for s_ in range(self.nS)]
        if D.shape[1:4] != permx.shape or poro.shape != permx.shape:
            raise ValueError("Shape mismatch between static and dynamic fields.")
        self.active = d0["active"].astype(bool) if "active" in d0.files else None   # (K,J,I) ACTNUM, None = all active
        if self.active is not None and self.active.shape != permx.shape:
            raise ValueError("`active` must have shape (K,J,I).")
        act = np.ones(permx.shape, bool) if self.active is None else self.active
        if not (np.all(np.isfinite(D)) and np.all(np.isfinite(permx)) and np.all(permx[act] > 0)):
            raise ValueError("Non-finite values (inactive cells must be filled with finite values) or "
                             "non-positive PERMX in active cells.")
        if not np.all(permx > 0):
            permx = np.where(permx > 0, permx, 1.0)                       # inactive filler only (never an input)
        if self.T_loc < 3 or any(np.any(x <= 0) for x in diffs):
            raise ValueError("`times` must be strictly increasing (within each scenario) with at least 3 entries.")
        self.dt_mean = float(np.concatenate(diffs).mean())

        self.static_raw = np.stack([np.log10(permx), poro.astype(np.float64)], axis=-1).astype(np.float32)
        self.well = d0["well_mask"].astype(np.float32) if "well_mask" in d0.files else None
        self.ctrl = None
        self.n_ctrl = 0
        if "ctrl" in d0.files:                                       # (T,K,J,I,C): operating controls of each step
            self.ctrl = np.concatenate([d["ctrl"] for d in ds], axis=0).astype(np.float32)
            if self.ctrl.shape[:4] != D.shape[:4] or not np.all(np.isfinite(self.ctrl)):
                raise ValueError("`ctrl` must have shape (T,K,J,I,C) and be finite.")
            self.n_ctrl = int(self.ctrl.shape[-1])

        sat = D[..., 1] + D[..., 2] + D[..., 3]
        self.sat_dev = float(np.abs(sat[:, act] - 1.0).max())
        self.sat_ok = self.sat_dev < 1e-3
        if self.active is not None:
            print(f"[data] ACTNUM mask: {int(self.active.sum())}/{self.active.size} active cells "
                  f"(inactive cells are excluded as samples, zeroed in the inputs, and flagged by an extra channel)")
        print(f"[data] scenarios={self.nS}, T={self.T_loc}/scenario, grid(K,J,I)={permx.shape}, "
              f"max|SO+SG+SW-1|={self.sat_dev:.2e} "
              f"-> physics projection {'enabled' if self.sat_ok else 'DISABLED (constraint not satisfied by data)'}")
        if self.well is None:
            print("[data] no `well_mask` in npz: well locations are NOT provided to the models "
                  "(add it in build_dataset.py for a stronger, fairer benchmark).")
        if self.ctrl is None and self.nS > 1:
            print("[data] WARNING: several scenarios but no `ctrl`: the models cannot see the well controls.")
        if self.ctrl is not None:
            print(f"[data] ctrl channels provided to models and tree baselines: C={self.n_ctrl}")

        self.h = patch // 2
        self.p = patch
        h = self.h
        t_idx = np.concatenate([s_ * self.T_loc + np.arange(self.T_loc - 1) for s_ in range(self.nS)])
        kc, jc, ic = np.meshgrid(np.arange(h, self.K - h), np.arange(h, self.J - h), np.arange(h, self.I - h),
                                 indexing="ij")
        kc, jc, ic = kc.ravel(), jc.ravel(), ic.ravel()
        if self.active is not None:
            m_ = self.active[kc, jc, ic]
            kc, jc, ic = kc[m_], jc[m_], ic[m_]
        C = len(kc)
        if C == 0:
            raise ValueError("No active interior cells.")
        if sub_per_scen > 0 and sub_per_scen < C * (self.T_loc - 1):
            rs = np.random.default_rng(seed + 999)                      # random subset per scenario (memory control)
            flat = []
            for s_ in range(self.nS):
                ids = np.sort(rs.choice(C * (self.T_loc - 1), sub_per_scen, replace=False))
                flat.append(s_ * self.T_loc * C + ids)                  # keeps (scenario, time, k, j, i) ordering
            flat = np.concatenate(flat)
            tpos, cpos = flat // C, flat % C                            # tpos indexes the (S*T_loc)-step axis
            self.ti = tpos.astype(np.int32)
        else:
            self.ti = np.repeat(t_idx, C).astype(np.int32)
            cpos = np.tile(np.arange(C), len(t_idx))
        self.k, self.j, self.i = kc[cpos].astype(np.int32), jc[cpos].astype(np.int32), ic[cpos].astype(np.int32)
        self.N = len(self.ti)
        self.tl = (self.ti % self.T_loc).astype(np.int32)           # local (within-scenario) step of each sample
        self.sc = (self.ti // self.T_loc).astype(np.int32)          # scenario of each sample
        self.Y = (D[self.ti + 1, self.k, self.j, self.i] - D[self.ti, self.k, self.j, self.i]).astype(np.float32)
        self.U = np.random.default_rng(seed + 12345).random((self.N, patch, patch)).astype(np.float32)
        self.centers_j = list(range(h, self.J - h)) if self.active is None else sorted(set(jc.tolist()))
        self.col_n = np.bincount(self.j, minlength=self.J)            # samples per J column (used to pick validation columns)
        print(f"[data] samples N={self.N}")

    def idx_scen(self, scens: Sequence[int]) -> np.ndarray:
        return np.where(np.isin(self.sc, list(scens)))[0]

    def hidden(self, idx: np.ndarray, level: float, center_visible: bool = True) -> np.ndarray:
        hid = self.U[idx] < level
        hid[:, self.h, self.h] = not center_visible
        return hid

    def no_hidden(self, n: int, center_visible: bool = True) -> np.ndarray:
        hid = np.zeros((n, self.p, self.p), dtype=bool)
        hid[:, self.h, self.h] = not center_visible
        return hid

    def idx_for(self, js: Sequence[int], tmin: int, tmax: int) -> np.ndarray:
        m = np.isin(self.j, list(js)) & (self.ti >= tmin) & (self.ti <= tmax)
        return np.where(m)[0]


# ============================================================
# SPATIAL BLOCK FOLDS (strict buffers)
# ============================================================

def _support(c, h):
    return set(range(c - h, c + h + 1))


def _union(cs, h):
    s: set = set()
    for c in cs:
        s |= _support(c, h)
    return s


def _runs(sc):
    out: List[List[int]] = []
    for c in sc:
        if out and c == out[-1][-1] + 1:
            out[-1].append(c)
        else:
            out.append([c])
    return out


VAL_MIN_SAMPLES = 3000


def make_folds(centers_j: Sequence[int], n_folds: int, val_width: int, h: int, col_weight=None):
    """`col_weight` (samples per J column; only given for data with inactive cells) makes the validation window prefer
    columns that actually contain samples (up to VAL_MIN_SAMPLES). It uses the mask only, never labels."""
    centers_j = list(centers_j)
    n = len(centers_j)
    bounds = np.linspace(0, n, n_folds + 1).round().astype(int)
    folds = []
    for f in range(n_folds):
        test = centers_j[bounds[f]:bounds[f + 1]]
        tsup = _union(test, h)
        elig = [c for c in centers_j if not (_support(c, h) & tsup)]
        best = None
        for run in _runs(elig):
            for s in range(0, len(run) - val_width + 1):
                val = run[s:s + val_width]
                vsup = _union(val, h)
                train = [c for c in elig if not (_support(c, h) & vsup)]
                dist = min(abs(v - t) for v in val for t in test)
                wv = 0 if col_weight is None else min(int(sum(col_weight[v] for v in val)), VAL_MIN_SAMPLES)
                key = (wv, len(train), dist)
                if best is None or key > best[0]:
                    best = (key, val, train)
        if best is None or len(best[2]) == 0:
            raise RuntimeError(f"Fold {f}: no valid split with strict buffers.")
        _, val, train = best
        s_tr, s_va, s_te = _union(train, h), _union(val, h), tsup
        assert not (s_tr & s_va) and not (s_tr & s_te) and not (s_va & s_te), "support leakage"
        folds.append({"train": sorted(train), "val": sorted(val), "test": sorted(test)})
    assert sorted(c for f in folds for c in f["test"]) == centers_j
    return folds


# ============================================================
# NORMALISATION (train columns and train times only)
# ============================================================

@dataclass
class Stats:
    st_m: np.ndarray
    st_s: np.ndarray
    dy_m: np.ndarray
    dy_s: np.ndarray
    ym: np.ndarray
    ys: np.ndarray
    rm: Optional[np.ndarray] = None      # residual (Y - rate-persistence base) mean / std
    rs: Optional[np.ndarray] = None


def _safe_std(a: np.ndarray) -> np.ndarray:
    s = a.astype(np.float64)
    return np.where(s < 1e-8, 1.0, s)


def compute_stats(data: DynData, train_js: Sequence[int], t_hi: int, tr_idx: np.ndarray,
                  steps: Optional[np.ndarray] = None) -> Stats:
    """`steps` (global step indices) restricts the dynamic statistics to the training scenarios (loso protocol)."""
    cols = sorted(_union(train_js, data.h))
    sraw = data.static_raw[:, cols, :, :]
    am = data.active[:, cols, :] if data.active is not None else None      # statistics over ACTIVE cells only
    sflat = sraw[am] if am is not None else sraw
    if steps is None:
        draw = data.D[: t_hi + 1][:, :, cols, :, :]
        dflat = draw[:, am] if am is not None else draw
        dy_m = dflat.reshape(-1, 4).mean(0).astype(np.float64)
        dy_s = _safe_std(dflat.reshape(-1, 4).std(0))
    else:                                            # loso: chunked float64 accumulation (memory-light)
        n_acc, s1, s2 = 0, np.zeros(4), np.zeros(4)
        for c0 in range(0, len(steps), 32):
            blk = data.D[np.asarray(steps)[c0:c0 + 32]][:, :, cols, :, :]
            fl = (blk[:, am] if am is not None else blk).reshape(-1, 4).astype(np.float64)
            n_acc += len(fl)
            s1 += fl.sum(0)
            s2 += (fl * fl).sum(0)
        dy_m = s1 / n_acc
        dy_s = _safe_std(np.sqrt(np.maximum(s2 / n_acc - dy_m ** 2, 0.0)))
    Ytr = data.Y[tr_idx]
    return Stats(
        st_m=sflat.reshape(-1, 2).mean(0).astype(np.float64),
        st_s=_safe_std(sflat.reshape(-1, 2).std(0)),
        dy_m=dy_m,
        dy_s=dy_s,
        ym=Ytr.mean(0).astype(np.float64),
        ys=_safe_std(Ytr.std(0)),
    )


class Source:
    """State container read by `gather`: the dataset itself or a rollout."""

    def __init__(self, static_raw, well, times, st: Stats, tmax: float, dt_mean: float,
                 starts: Optional[Sequence[int]] = None, ctrl: Optional[np.ndarray] = None,
                 active: Optional[np.ndarray] = None) -> None:
        self.static_n = ((static_raw - st.st_m) / st.st_s).astype(np.float32)
        self.act = None if active is None else active.astype(np.float32)           # (K,J,I) 1 = active cell
        self.inact = None if active is None else ~active.astype(bool)
        if self.inact is not None:
            self.static_n[self.inact] = 0.0
        self.well = well
        self.ctrl = ctrl
        self.times = np.asarray(times, dtype=np.float64)
        self.st = st
        self.tmax = float(tmax)
        self.dt_mean = float(dt_mean)
        T = len(self.times)
        K, J, I = static_raw.shape[:3]
        shp = (T, K, J, I, 4)
        self.dyn_raw = np.zeros(shp, np.float32)
        self.dyn_n = np.zeros(shp, np.float32)
        self.prev_raw = np.zeros(shp, np.float32)     # increment that arrived at step t: D[t]-D[t-1]
        self.prev_n = np.zeros(shp, np.float32)       # scaled by train std of increments, no shift (0 stays 0)
        bs = sorted(set(int(x) for x in (starts if starts is not None else [0])) | {0})
        self.block_starts = bs
        st_arr = np.zeros(T, np.int64)                # first step of the scenario each step belongs to
        ratio = np.zeros(T)                           # dt_cur / dt_prev (0 at the first step of every scenario)
        for b_i, b0 in enumerate(bs):
            b1 = bs[b_i + 1] if b_i + 1 < len(bs) else T
            st_arr[b0:b1] = b0
            dtb = np.diff(self.times[b0:b1])
            if len(dtb) > 1:
                ratio[b0 + 1:b1 - 1] = dtb[1:] / dtb[:-1]
        self.start = st_arr
        self.ratio = ratio

    @classmethod
    def from_full(cls, data: DynData, st: Stats) -> "Source":
        src = cls(data.static_raw, data.well, data.times, st, data.tmax, data.dt_mean,
                  starts=[s_ * data.T_loc for s_ in range(data.nS)], ctrl=data.ctrl, active=data.active)
        src.dyn_raw[:] = data.D
        src.dyn_n[:] = ((data.D - st.dy_m) / st.dy_s).astype(np.float32)
        src.prev_raw[1:] = data.D[1:] - data.D[:-1]
        for b0 in src.block_starts:                       # no increment "arrives" at the first step of a scenario
            src.prev_raw[b0] = 0.0
        src.prev_n[:] = (src.prev_raw / st.ys).astype(np.float32)
        if src.inact is not None:                         # inactive cells carry no information
            src.dyn_n[:, src.inact] = 0.0
            src.prev_n[:, src.inact] = 0.0
            src.prev_raw[:, src.inact] = 0.0
        return src

    def set_step(self, s: int, dyn: np.ndarray, prev: Optional[np.ndarray] = None) -> None:
        self.dyn_raw[s] = dyn
        self.dyn_n[s] = ((dyn - self.st.dy_m) / self.st.dy_s).astype(np.float32)
        if prev is None:
            prev = dyn - self.dyn_raw[s - 1] if s > 0 else np.zeros_like(dyn)
        self.prev_raw[s] = prev
        self.prev_n[s] = (prev / self.st.ys).astype(np.float32)
        if self.inact is not None:
            self.dyn_n[s][self.inact] = 0.0
            self.prev_n[s][self.inact] = 0.0
            self.prev_raw[s][self.inact] = 0.0


def rate_base_np(src: Source, ti, k, j, i) -> np.ndarray:
    """Rate persistence: previous increment of the cell scaled by dt_cur/dt_prev (0 at the first step)."""
    return (src.prev_raw[ti, k, j, i] * src.ratio[ti][:, None]).astype(np.float32)


def add_resid_stats(st: Stats, data: DynData, src: Source, tr: np.ndarray) -> None:
    base = rate_base_np(src, data.ti[tr], data.k[tr], data.j[tr], data.i[tr])
    res = data.Y[tr].astype(np.float64) - base
    st.rm = res.mean(0)
    st.rs = _safe_std(res.std(0))


def gather(src: Source, ti, k, j, i, hid, h: int, n_lag: int = 0) -> Dict[str, np.ndarray]:
    """Vectorised extraction of (B,p,p,p,*) neighbourhoods (axes K,J,I)."""
    off = np.arange(-h, h + 1)
    kk = k[:, None, None, None] + off[None, :, None, None]
    jj = j[:, None, None, None] + off[None, None, :, None]
    ii = i[:, None, None, None] + off[None, None, None, :]
    t4 = ti[:, None, None, None]
    prev = None
    if n_lag > 0:
        prev = np.concatenate(
            [src.prev_n[np.maximum(ti - l, src.start[ti])[:, None, None, None], kk, jj, ii] for l in range(n_lag)],
            axis=-1)
    return {
        "static": src.static_n[kk, jj, ii],
        "dyn": src.dyn_n[t4, kk, jj, ii],
        "prev": prev,                                                  # lag1, lag2, ... (4 channels each)
        "well": None if src.well is None else src.well[kk, jj, ii][..., None],
        "ctrl": None if src.ctrl is None else src.ctrl[t4, kk, jj, ii],               # (B,p,p,p,C) operating controls
        "act": None if src.act is None else src.act[kk, jj, ii][..., None],           # (B,p,p,p,1) 1 = active cell
        "obs": (~hid)[:, None, :, :, None].astype(np.float32),
        "t": (src.times[ti] / src.tmax).astype(np.float32),
        "dt": ((src.times[ti + 1] - src.times[ti]) / src.dt_mean).astype(np.float32),
    }


def build_x(parts: Dict[str, np.ndarray], spec: Dict) -> np.ndarray:
    B, p = parts["static"].shape[0], parts["static"].shape[1]
    obs = np.broadcast_to(parts["obs"], (B, p, p, p, 1)).astype(np.float32)
    ch = [parts["static"][..., 0:1]]
    if spec["poro"]:
        ch.append(parts["static"][..., 1:2])
    if parts["well"] is not None:
        ch.append(parts["well"])
    if parts.get("ctrl") is not None:
        ch.append(parts["ctrl"])
    if parts.get("act") is not None:
        ch.append(parts["act"])
    if spec["dyn"]:
        ch.append(parts["dyn"] * obs)
        if spec["prev"] > 0:
            ch.append(parts["prev"] * obs)
        ch.append(obs)
    if spec["time"]:
        for key in ("t", "dt"):
            ch.append(np.broadcast_to(parts[key][:, None, None, None, None], (B, p, p, p, 1)))
    x = np.concatenate(ch, axis=-1).astype(np.float32)
    return np.ascontiguousarray(x.transpose(0, 4, 1, 2, 3))     # (B,C,K,J,I)


def n_channels(spec: Dict, has_well: bool, n_ctrl: int = 0, has_act: bool = False) -> int:
    c = 1 + int(spec["poro"]) + int(has_well) + int(n_ctrl) + int(has_act)
    if spec["dyn"]:
        c += 4 + 4 * spec["prev"] + 1
    if spec["time"]:
        c += 2
    return c


def make_features(parts: Dict[str, np.ndarray], k: np.ndarray, K: int, n_lag: int, h: int) -> np.ndarray:
    B, p = parts["static"].shape[0], parts["static"].shape[1]
    obs = np.broadcast_to(parts["obs"], (B, p, p, p, 1)).astype(np.float64)
    w = np.maximum(obs.sum(axis=(1, 2, 3)), 1.0)                        # (B,1)

    def mstat(a):
        a = a.astype(np.float64)
        m = (a * obs).sum(axis=(1, 2, 3)) / w
        v = (((a - m[:, None, None, None, :]) ** 2) * obs).sum(axis=(1, 2, 3)) / w
        return m, np.sqrt(v)

    dm, ds = mstat(parts["dyn"])
    feats = [parts["dyn"][:, h, h, h, :] * obs[:, h, h, h, :], dm, ds]
    for l in range(n_lag):
        sl = slice(4 * l, 4 * l + 4)
        pm, _ = mstat(parts["prev"][..., sl])
        feats += [parts["prev"][:, h, h, h, sl] * obs[:, h, h, h, :], pm]
    st = parts["static"].astype(np.float64)
    feats += [st[:, h, h, h, :], st.mean(axis=(1, 2, 3)), st[..., 0].std(axis=(1, 2, 3))[:, None]]
    feats += [(k / max(K - 1, 1))[:, None], parts["t"][:, None], parts["dt"][:, None]]
    if parts.get("ctrl") is not None:                                    # operating controls (same info as the CNN)
        c_ = parts["ctrl"].astype(np.float64)
        feats += [c_[:, h, h, h, :], c_.mean(axis=(1, 2, 3)), c_.max(axis=(1, 2, 3))]
    if parts.get("act") is not None:                                     # fraction of active cells in the patch
        feats += [parts["act"].astype(np.float64).mean(axis=(1, 2, 3))]
    return np.concatenate(feats, axis=1).astype(np.float32)


# ============================================================
# BASELINE PREDICTORS: call(src, ti_src, ti_real, k, j, i, hid) -> (n,4) physical
# ============================================================

class Predictor:
    def __call__(self, src, ti_src, ti_real, k, j, i, hid):
        raise NotImplementedError


class Persistence(Predictor):
    def __call__(self, src, ti_src, ti_real, k, j, i, hid):
        return np.zeros((len(k), 4), np.float32)


class TrainMean(Predictor):
    def __init__(self, Ytr):
        self.m = Ytr.mean(0).astype(np.float32)

    def __call__(self, src, ti_src, ti_real, k, j, i, hid):
        return np.broadcast_to(self.m, (len(k), 4)).copy()


def _fill_time(tab):
    T = tab.shape[0]
    for t in range(1, T):
        m = np.isnan(tab[t])
        tab[t][m] = tab[t - 1][m]
    for t in range(T - 2, -1, -1):
        m = np.isnan(tab[t])
        tab[t][m] = tab[t + 1][m]
    return tab


class TimeMean(Predictor):
    def __init__(self, ti_tr, Ytr, T):
        sums = np.zeros((T, 4))
        cnt = np.zeros(T)
        np.add.at(sums, ti_tr, Ytr)
        np.add.at(cnt, ti_tr, 1.0)
        tab = np.where(cnt[:, None] > 0, sums / np.maximum(cnt[:, None], 1), np.nan)
        self.tab = _fill_time(tab).astype(np.float32)

    def __call__(self, src, ti_src, ti_real, k, j, i, hid):
        return self.tab[ti_real]


class TimeLayerMean(Predictor):
    def __init__(self, ti_tr, k_tr, Ytr, T, K, fallback: TimeMean):
        sums = np.zeros((T, K, 4))
        cnt = np.zeros((T, K))
        np.add.at(sums, (ti_tr, k_tr), Ytr)
        np.add.at(cnt, (ti_tr, k_tr), 1.0)
        tab = np.where(cnt[..., None] > 0, sums / np.maximum(cnt[..., None], 1), np.nan)
        tab = _fill_time(tab)
        tm = np.broadcast_to(fallback.tab[:, None, :], tab.shape)
        self.tab = np.where(np.isnan(tab), tm, tab).astype(np.float32)

    def __call__(self, src, ti_src, ti_real, k, j, i, hid):
        return self.tab[ti_real, k]


class RatePersistence(Predictor):
    def __init__(self, fallback: Predictor):
        self.fb = fallback

    def __call__(self, src, ti_src, ti_real, k, j, i, hid):
        out = rate_base_np(src, ti_src, k, j, i)
        first = ti_real == 0
        if first.any():
            out[first] = self.fb(src, ti_src, ti_real, k, j, i, hid)[first]
        return out


class MultiHGB:
    def __init__(self, seed: int, **kw):
        self.seed, self.kw = seed, kw

    def fit(self, X, Y):
        self.ms = [HistGradientBoostingRegressor(random_state=self.seed, **self.kw).fit(X, Y[:, c])
                   for c in range(Y.shape[1])]
        return self

    def predict(self, X):
        return np.stack([m.predict(X) for m in self.ms], axis=1)


class FeatureModel(Predictor):
    GRIDS = {
        "ridge": [{}],
        "et": [dict(min_samples_leaf=2), dict(min_samples_leaf=8)],
        "hgb": [dict(max_leaf_nodes=15, learning_rate=0.1, max_iter=200),
                dict(max_leaf_nodes=63, learning_rate=0.05, max_iter=300)],
    }

    def __init__(self, kind: str, n_lag: int, st: Stats, K: int, h: int, seed: int):
        self.kind, self.n_lag, self.st, self.K, self.h, self.seed = kind, n_lag, st, K, h, seed
        self.params = None
        self.model = None

    def _make(self, params):
        if self.kind == "ridge":
            return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 13)))
        if self.kind == "et":
            return ExtraTreesRegressor(n_estimators=100, max_features=0.6, n_jobs=-1,
                                       random_state=self.seed, **params)
        return MultiHGB(self.seed, **params)

    def _feat_many(self, src, ti, k, j, i, hid):
        out = []
        for s in range(0, len(k), 8192):
            sl = slice(s, s + 8192)
            parts = gather(src, ti[sl], k[sl], j[sl], i[sl], hid[sl], self.h, self.n_lag)
            out.append(make_features(parts, k[sl], self.K, self.n_lag, self.h))
        return np.concatenate(out)

    def fit(self, data: DynData, src: Source, tr, va_sub, levels, max_train: int, rng):
        sub = tr if len(tr) <= max_train else np.sort(rng.choice(tr, max_train, replace=False))
        lv = rng.choice(np.asarray(levels, dtype=np.float32), size=len(sub))
        hid = data.U[sub] < lv[:, None, None]
        hid[:, self.h, self.h] = False
        X = self._feat_many(src, data.ti[sub], data.k[sub], data.j[sub], data.i[sub], hid)
        Yn = (data.Y[sub] - self.st.ym) / self.st.ys
        Xv, Yv = [], []
        for L in levels:
            hv = data.hidden(va_sub, L, True)
            Xv.append(self._feat_many(src, data.ti[va_sub], data.k[va_sub], data.j[va_sub], data.i[va_sub], hv))
            Yv.append((data.Y[va_sub] - self.st.ym) / self.st.ys)
        Xv, Yv = np.vstack(Xv), np.vstack(Yv)
        best = None
        for params in self.GRIDS[self.kind]:
            m = self._make(params).fit(X, Yn)
            e = float(np.mean((m.predict(Xv) - Yv) ** 2))
            if best is None or e < best[0]:
                best = (e, m, params)
        _, self.model, self.params = best
        return self

    def __call__(self, src, ti_src, ti_real, k, j, i, hid):
        X = self._feat_many(src, ti_src, k, j, i, hid)
        return (self.model.predict(X) * self.st.ys + self.st.ym).astype(np.float32)


def build_baselines(data, src, st, tr, va_sub, cfg: Config, levels, fold_seed: int) -> Dict[str, Predictor]:
    rng = np.random.default_rng(fold_seed)
    Ytr, ti_tr, k_tr = data.Y[tr], data.tl[tr], data.k[tr]          # local (within-scenario) step for time tables
    tm = TimeMean(ti_tr, Ytr, data.T_loc)
    tlm = TimeLayerMean(ti_tr, k_tr, Ytr, data.T_loc, data.K, tm)
    preds: Dict[str, Predictor] = {
        "persistence": Persistence(), "train_mean": TrainMean(Ytr), "time_mean": tm,
        "time_layer_mean": tlm, "rate_persistence": RatePersistence(tlm),
    }
    nh = max(1, cfg.n_hist)
    for name, kind, n_lag in [("ridge", "ridge", 0), ("extratrees", "et", 0), ("hgb", "hgb", 0),
                              ("ridge_prev", "ridge", 1), ("extratrees_prev", "et", 1), ("hgb_prev", "hgb", 1),
                              ("extratrees_hist", "et", nh), ("hgb_hist", "hgb", nh)]:
        preds[name] = FeatureModel(kind, n_lag, st, data.K, data.h, fold_seed).fit(
            data, src, tr, va_sub, levels, cfg.baseline_max_train, rng)
    return preds


# ============================================================
# CNN
# ============================================================

D4_FULL = [(tr, fj, fi) for tr in (False, True) for fj in (False, True) for fi in (False, True)]

if HAS_TORCH:

    class ResBlock3D(nn.Module):
        def __init__(self, ch: int):
            super().__init__()
            self.c1, self.g1 = nn.Conv3d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch)
            self.c2, self.g2 = nn.Conv3d(ch, ch, 3, padding=1), nn.GroupNorm(4, ch)

        def forward(self, x):
            y = F.relu(self.g1(self.c1(x)))
            y = self.g2(self.c2(y))
            return F.relu(x + y)

    class DynamicCNN(nn.Module):
        def __init__(self, in_ch: int, out_ch: int = 4, width: int = 16, patch: int = 5, dropout: float = 0.15,
                     arch: str = "plain"):
            super().__init__()
            w, w2 = width, width * 2
            self.h = patch // 2
            if arch == "res":
                self.enc = nn.Sequential(
                    nn.Conv3d(in_ch, w2, 3, padding=1), nn.GroupNorm(4, w2), nn.ReLU(inplace=True),
                    ResBlock3D(w2), ResBlock3D(w2),
                )
            else:
                self.enc = nn.Sequential(
                    nn.Conv3d(in_ch, w, 3, padding=1), nn.GroupNorm(2, w), nn.ReLU(inplace=True),
                    nn.Conv3d(w, w2, 3, padding=1), nn.GroupNorm(4, w2), nn.ReLU(inplace=True),
                    nn.Conv3d(w2, w2, 3, padding=1), nn.GroupNorm(4, w2), nn.ReLU(inplace=True),
                )
            self.head = nn.Sequential(
                nn.Linear(2 * w2, 64), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(64, out_ch)
            )

        def forward(self, x):
            f = self.enc(x)                                    # (B,w2,5,5,5)
            centre = f[:, :, self.h, self.h, self.h]           # cell-level read-out
            glob = f.mean(dim=(2, 3, 4))
            return self.head(torch.cat([centre, glob], dim=1))

    def to_phys(out, st_t, project: bool, base=None):
        """Network output -> physical increments. Direct: out*ys+ym. Residual: base + out*rs + rm."""
        if base is None:
            y = out * st_t["ys"] + st_t["ym"]
        else:
            y = base + out * st_t["rs"] + st_t["rm"]
        if project:
            s = y[:, 1:4]
            y = torch.cat([y[:, :1], s - s.mean(dim=1, keepdim=True)], dim=1)
        return y

    def loss_fn(pred, y, st_t, kind: str, active_w: float):
        d = (pred - y) / st_t["ys"]                            # sigma-normalised error (NRMSE units)
        if kind == "huber":
            l = F.huber_loss(d, torch.zeros_like(d), reduction="none", delta=1.0).mean(dim=1)
        else:
            l = d.pow(2).mean(dim=1)
        if active_w > 0:
            yn = ((y - st_t["ym"]) / st_t["ys"]).abs().amax(dim=1)
            l = l * (1.0 + active_w * (yn > 2.0).float())
        return l.mean()

    def augment_batch(x, mode: str):
        if mode == "none":
            return x
        b = x.shape[0]
        dev = x.device
        tr = (torch.rand(b, device=dev) < 0.5).view(-1, 1, 1, 1, 1)
        x = torch.where(tr, x.transpose(3, 4), x)
        fj = (torch.rand(b, device=dev) < 0.5).view(-1, 1, 1, 1, 1)
        fi = (torch.rand(b, device=dev) < 0.5).view(-1, 1, 1, 1, 1)
        x = torch.where(fj, x.flip(3), x)
        x = torch.where(fi, x.flip(4), x)
        return x

    @torch.no_grad()
    def predict_members(models, x, mode: str, use_tta: bool, st_t, project: bool, base):
        combos = D4_FULL if (use_tta and mode == "d4") else [(False, False, False)]
        outs = []
        for m in models:
            m.eval()
            acc = 0.0
            for tr, fj, fi in combos:
                xx = x
                if tr:
                    xx = xx.transpose(3, 4)
                if fj:
                    xx = xx.flip(3)
                if fi:
                    xx = xx.flip(4)
                acc = acc + to_phys(m(xx), st_t, project, base)
            outs.append(acc / len(combos))
        return torch.stack(outs)

    def stats_tensors(st: Stats, device):
        d = {"ym": torch.as_tensor(st.ym, dtype=torch.float32, device=device),
             "ys": torch.as_tensor(st.ys, dtype=torch.float32, device=device)}
        if st.rm is not None:
            d["rm"] = torch.as_tensor(st.rm, dtype=torch.float32, device=device)
            d["rs"] = torch.as_tensor(st.rs, dtype=torch.float32, device=device)
        return d

    def cnn_predict(models, spec, src, st_t, ti, k, j, i, hid, h, cfg, device, project):
        out = []
        for s in range(0, len(k), 4096):
            sl = slice(s, s + 4096)
            parts = gather(src, ti[sl], k[sl], j[sl], i[sl], hid[sl], h, spec["prev"])
            x = torch.as_tensor(build_x(parts, spec), device=device)
            base = (torch.as_tensor(rate_base_np(src, ti[sl], k[sl], j[sl], i[sl]), device=device)
                    if spec["resid"] else None)
            out.append(predict_members(models, x, spec["aug"], cfg.tta, st_t, project, base).cpu().numpy())
        return np.concatenate(out, axis=1).astype(np.float64)            # (M,n,4)

    def train_member(data, src, st, st_t, tr, va_sub, cfg: Config, spec: Dict, seed: int, device, project: bool):
        set_seed(seed)
        rng = np.random.default_rng(seed)
        h, p = data.h, data.p
        n_lag, resid = spec["prev"], spec["resid"]
        model = DynamicCNN(n_channels(spec, data.well is not None, data.n_ctrl, data.active is not None), 4, cfg.width, cfg.patch, cfg.dropout,
                           spec["arch"]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
        drop_max = cfg.train_drop_max if spec["drop"] else 0.0

        # fixed validation set: all sparsity levels, stratified over time (random subsample of the block)
        xs, ys, bs = [], [], []
        for L in cfg.sparsity:
            hid = data.hidden(va_sub, L, spec["center"])
            for s in range(0, len(va_sub), 2048):
                sl = slice(s, s + 2048)
                idx = va_sub[sl]
                parts = gather(src, data.ti[idx], data.k[idx], data.j[idx], data.i[idx], hid[sl], h, n_lag)
                xs.append(build_x(parts, spec))
                ys.append(data.Y[idx])
                if resid:
                    bs.append(rate_base_np(src, data.ti[idx], data.k[idx], data.j[idx], data.i[idx]))
        x_va = torch.as_tensor(np.concatenate(xs), device=device)
        y_va = torch.as_tensor(np.concatenate(ys), device=device)
        b_va = torch.as_tensor(np.concatenate(bs), device=device) if resid else None

        hist = {"train_loss": [], "val_loss": []}
        best, best_state, bad = float("inf"), None, 0
        for _ in range(cfg.epochs):
            model.train()
            ep = rng.choice(tr, size=min(cfg.epoch_samples, len(tr)), replace=False)
            losses = []
            for s in range(0, len(ep), cfg.batch_size):
                idx = ep[s:s + cfg.batch_size]
                b = len(idx)
                ps = rng.random(b) * drop_max
                hid = rng.random((b, p, p)) < ps[:, None, None]
                hid[:, h, h] = not spec["center"]
                parts = gather(src, data.ti[idx], data.k[idx], data.j[idx], data.i[idx], hid, h, n_lag)
                x = augment_batch(torch.as_tensor(build_x(parts, spec), device=device), spec["aug"])
                y = torch.as_tensor(data.Y[idx], device=device)
                base = (torch.as_tensor(rate_base_np(src, data.ti[idx], data.k[idx], data.j[idx], data.i[idx]),
                                        device=device) if resid else None)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(to_phys(model(x), st_t, project, base), y, st_t, cfg.loss, cfg.active_w)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                losses.append(float(loss.item()))
            model.eval()
            with torch.no_grad():
                tot, cnt = 0.0, 0
                for s in range(0, len(x_va), 4096):
                    bb = None if b_va is None else b_va[s:s + 4096]
                    pred = to_phys(model(x_va[s:s + 4096]), st_t, project, bb)
                    n_ = len(pred)
                    tot += float(loss_fn(pred, y_va[s:s + 4096], st_t, cfg.loss, 0.0).item()) * n_
                    cnt += n_
                vl = tot / cnt
            sched.step(vl)
            hist["train_loss"].append(float(np.mean(losses)))
            hist["val_loss"].append(vl)
            if vl < best - 1e-9:
                best, best_state, bad = vl, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        model.load_state_dict(best_state)
        return model, hist

    def save_models(models, path: Path) -> None:
        torch.save([m.state_dict() for m in models], path)

    def load_models(path: Path, spec, data, cfg, device):
        states = torch.load(path, map_location=device)
        models = []
        for sd in states:
            m = DynamicCNN(n_channels(spec, data.well is not None, data.n_ctrl, data.active is not None), 4, cfg.width, cfg.patch, cfg.dropout,
                           spec["arch"]).to(device)
            m.load_state_dict(sd)
            models.append(m)
        return models

    class CNNEnsemble(Predictor):
        def __init__(self, models, spec, st_t, h, cfg, device, project):
            self.models, self.spec, self.st_t, self.h = models, spec, st_t, h
            self.cfg, self.device, self.project = cfg, device, project

        def members(self, src, ti_src, k, j, i, hid):
            return cnn_predict(self.models, self.spec, src, self.st_t, ti_src, k, j, i, hid,
                               self.h, self.cfg, self.device, self.project)

        def __call__(self, src, ti_src, ti_real, k, j, i, hid):
            return self.members(src, ti_src, k, j, i, hid).mean(0).astype(np.float32)


# ============================================================
# CONFORMAL, METRICS, STATISTICS
# ============================================================

def conformal_q(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(math.ceil((n + 1) * (1 - alpha)))
    return float(np.sort(scores)[min(k, n) - 1])


class CoverAcc:
    def __init__(self):
        self.n = 0
        self.h_abs = np.zeros(4)
        self.w_abs = np.zeros(4)
        self.h_nrm = np.zeros(4)
        self.w_nrm = np.zeros(4)
        self.has_nrm = False

    def add(self, yv, pv, yt, pt, sv, stt, alpha):
        qa = np.array([conformal_q(np.abs(yv[:, c] - pv[:, c]), alpha) for c in range(4)])
        self.h_abs += (np.abs(yt - pt) <= qa).sum(0)
        self.w_abs += 2 * qa * len(yt)
        if sv is not None and stt is not None:
            qn = np.array([conformal_q(np.abs(yv[:, c] - pv[:, c]) / (sv[:, c] + 1e-9), alpha) for c in range(4)])
            self.h_nrm += (np.abs(yt - pt) <= qn * stt).sum(0)
            self.w_nrm += (2 * qn * stt).sum(0)
            self.has_nrm = True
        self.n += len(yt)

    def state(self) -> Dict:
        return dict(n=self.n, h_abs=self.h_abs.copy(), w_abs=self.w_abs.copy(), h_nrm=self.h_nrm.copy(),
                    w_nrm=self.w_nrm.copy(), has_nrm=self.has_nrm)

    @classmethod
    def from_state(cls, d: Dict) -> "CoverAcc":
        o = cls()
        o.n, o.has_nrm = d["n"], d["has_nrm"]
        o.h_abs, o.w_abs, o.h_nrm, o.w_nrm = (d[k].copy() for k in ("h_abs", "w_abs", "h_nrm", "w_nrm"))
        return o

    def merge(self, o: "CoverAcc") -> None:
        self.n += o.n
        self.h_abs += o.h_abs
        self.w_abs += o.w_abs
        self.h_nrm += o.h_nrm
        self.w_nrm += o.w_nrm
        self.has_nrm = self.has_nrm or o.has_nrm


def active_thresholds(Y: np.ndarray, q: float = 0.9) -> np.ndarray:
    return np.quantile(np.abs(Y), q, axis=0)


def point_metrics(y, p, thr, valid):
    e = p - y
    out = {}
    for c, name in enumerate(CHANNELS_OUT):
        rmse = float(np.sqrt(np.mean(e[:, c] ** 2)))
        out[f"{name}_rmse"] = rmse
        out[f"{name}_mae"] = float(np.mean(np.abs(e[:, c])))
        sst = float(np.sum((y[:, c] - y[:, c].mean()) ** 2))
        out[f"{name}_r2"] = float(1 - np.sum(e[:, c] ** 2) / sst) if valid[c] and sst > 0 else float("nan")
        act = np.abs(y[:, c]) >= thr[c] if thr[c] > 0 else np.zeros(len(y), bool)
        out[f"{name}_rmse_active"] = float(np.sqrt(np.mean(e[act, c] ** 2))) if act.any() else float("nan")
    return out


class ClusterStats:
    """Per J-column sums enabling a fast cluster bootstrap."""

    def __init__(self, y: np.ndarray, groups: np.ndarray, sigma: np.ndarray, valid: np.ndarray):
        self.order = np.argsort(groups, kind="stable")
        gs = groups[self.order]
        self.starts = np.r_[0, np.flatnonzero(np.diff(gs)) + 1]
        self.n_g = np.diff(np.r_[self.starts, len(gs)]).astype(np.float64)
        self.G = len(self.starts)
        yo = y[self.order].astype(np.float64)
        self.Sy = np.add.reduceat(yo, self.starts, axis=0)
        self.Syy = np.add.reduceat(yo ** 2, self.starts, axis=0)
        self.sigma2 = sigma.astype(np.float64) ** 2
        self.valid = valid
        self.nv = int(valid.sum())

    def sse(self, y, p) -> np.ndarray:
        e2 = ((p - y).astype(np.float64) ** 2)[self.order]
        return np.add.reduceat(e2, self.starts, axis=0)                  # (G,4)

    def boot_counts(self, n_boot, rng):
        draws = rng.integers(0, self.G, size=(n_boot, self.G))
        return np.stack([np.bincount(r, minlength=self.G) for r in draws]).astype(np.float64)

    def metrics(self, SSE, weights):
        n = weights @ self.n_g
        mse = (weights @ SSE) / n[:, None]
        nmse = (mse[:, self.valid] / self.sigma2[self.valid]).mean(1)
        sst = (weights @ self.Syy) - (weights @ self.Sy) ** 2 / n[:, None]
        r2 = 1 - (weights @ SSE) / np.maximum(sst, 1e-300)
        return mse, np.sqrt(nmse), r2


def ci(a, lo=2.5, hi=97.5):
    return float(np.nanpercentile(a, lo)), float(np.nanpercentile(a, hi))


def holm(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj, run = np.empty(m), 0.0
    for rank, idx in enumerate(order):
        run = max(run, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, run)
    return adj.tolist()


# ============================================================
# ROLLOUT (boundary-forced autoregression inside the test block)
# ============================================================

def run_rollout(data: DynData, st: Stats, test_js: Sequence[int], t0: int, H: int,
                preds: Dict[str, Predictor], n_lag_max: int) -> Dict[str, np.ndarray]:
    """Interior test cells evolve with their own predictions; all other cells are
    driven by the true state at every step. Returns {name: squared errors (H,4)
    summed over centres} and the number of centres under key '_n'."""
    h = data.h
    kk, jj, ii = np.meshgrid(np.arange(h, data.K - h), np.asarray(sorted(test_js)),
                             np.arange(h, data.I - h), indexing="ij")
    kk, jj, ii = kk.ravel(), jj.ravel(), ii.ravel()
    if data.active is not None:
        m_ = data.active[kk, jj, ii]
        kk, jj, ii = kk[m_], jj[m_], ii[m_]
    n = len(kk)
    off = min(max(n_lag_max, 1), t0)                      # true history before the roll-out starts
    sl = slice(t0 - off, t0 + H + 1)
    Dt = data.D[sl]
    assert data.nS == 1, "roll-out is only defined for single-scenario data"
    dt_mean = data.dt_mean
    out: Dict[str, np.ndarray] = {}
    for name, pred in preds.items():
        src = Source(data.static_raw, data.well, data.times[sl], st, data.tmax, dt_mean,
                     ctrl=None if data.ctrl is None else data.ctrl[sl], active=data.active)
        first_prev = (data.D[t0 - off] - data.D[t0 - off - 1]) if t0 - off > 0 else np.zeros_like(Dt[0])
        src.set_step(0, Dt[0], first_prev)
        for s in range(1, off + 1):
            src.set_step(s, Dt[s])
        sse = np.zeros((H, 4))
        for s in range(H):
            idx = off + s
            ti_src = np.full(n, idx, dtype=np.int64)
            ti_real = np.full(n, t0 + s, dtype=np.int64)
            dpred = pred(src, ti_src, ti_real, kk, jj, ii, data.no_hidden(n, True))
            nxt = Dt[idx + 1].copy()
            nxt[kk, jj, ii] = src.dyn_raw[idx][kk, jj, ii] + dpred
            src.set_step(idx + 1, nxt)
            sse[s] = ((nxt[kk, jj, ii] - Dt[idx + 1][kk, jj, ii]).astype(np.float64) ** 2).sum(0)
        out[name] = sse
    out["_n"] = np.array([n])
    return out


# ============================================================
# PLOTS
# ============================================================

def _style():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 120, "savefig.dpi": 300})


def _save(fig, out: Path, name: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def main_name(methods, cfg: Config) -> Optional[str]:
    if cfg.main in methods:
        return cfg.main
    cn = [m for m in methods if m.startswith("cnn_")]
    return cn[0] if cn else None


def fig_bars(df0: pd.DataFrame, out: Path):
    _style()
    d = df0.sort_values("nrmse")
    fig, ax = plt.subplots(figsize=(7.8, 0.38 * len(d) + 1.5))
    for pos, (_, r) in enumerate(d.iterrows()):
        col = PALETTE[0] if r.method.startswith("cnn") else PALETTE[8]
        ax.barh(pos, r.nrmse, color=col, alpha=0.85)
        ax.errorbar(r.nrmse, pos, xerr=[[r.nrmse - r.nrmse_ci_lo], [r.nrmse_ci_hi - r.nrmse]],
                    color="k", capsize=3, lw=1)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([pretty(m) for m in d.method])
    ax.invert_yaxis()
    ax.set_xlabel("Macro NRMSE of increments (95% spatial-bootstrap CI)")
    _save(fig, out, "fig_nrmse_bars")


def fig_sparsity(df: pd.DataFrame, out: Path, main: Optional[str]):
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for i, m in enumerate(df.method.unique()):
        s = df[df.method == m].sort_values("level")
        ax.plot(s.level * 100, s.nrmse, marker="o", color=PALETTE[i % len(PALETTE)],
                label=pretty(m), lw=2.2 if m == main else 1)
    ax.set_xlabel("Hidden neighbour columns (%)")
    ax.set_ylabel("Macro NRMSE")
    ax.legend(fontsize=6, ncol=2)
    _save(fig, out, "fig_sparsity")


def fig_parity(data, oof, sel, out, name):
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for c, ch in enumerate(CHANNELS_OUT):
        y, p = data.Y[sel, c], oof[name][0.0][sel, c]
        axes[c].hexbin(y, p, gridsize=60, bins="log", mincnt=1, cmap="viridis")
        lo, hi = min(y.min(), p.min()), max(y.max(), p.max())
        axes[c].plot([lo, hi], [lo, hi], "r--", lw=1)
        axes[c].set_title(f"{ch} - {pretty(name)}", fontsize=9)
        axes[c].set_xlabel("True increment")
        axes[c].set_ylabel("Predicted increment")
    _save(fig, out, "fig_parity")


def fig_time_curve(data, oof, sel, out, names, sigma, valid):
    _style()
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    tis = np.unique(data.ti[sel])
    for i, m in enumerate(names):
        vals = []
        for t in tis:
            s = sel[data.ti[sel] == t]
            e2 = ((oof[m][0.0][s] - data.Y[s]) ** 2).mean(0)
            vals.append(np.sqrt((e2[valid] / sigma[valid] ** 2).mean()))
        ax.plot(data.times[tis], vals, marker="o", ms=3, color=PALETTE[i % len(PALETTE)], label=pretty(m))
    ax.set_xlabel("Time")
    ax.set_ylabel("Macro NRMSE")
    ax.legend(fontsize=7)
    _save(fig, out, "fig_error_vs_time")


def fig_maps(data, oof, sel, out, name):
    _style()
    ti_star = int(np.argmax([np.abs(data.Y[sel][data.ti[sel] == t, 3]).mean() if (data.ti[sel] == t).any() else -1
                             for t in range(data.T - 1)]))
    s_t = sel[data.ti[sel] == ti_star]
    k_star = int(np.argmax([np.abs(data.Y[s_t][data.k[s_t] == k, 3]).mean() if (data.k[s_t] == k).any() else -1
                            for k in range(data.K)]))
    s_tk = s_t[data.k[s_t] == k_star]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for r, c in enumerate([3, 0]):
        grids = []
        for vals in (data.Y[s_tk, c], oof[name][0.0][s_tk, c]):
            g = np.full((data.J, data.I), np.nan)
            g[data.j[s_tk], data.i[s_tk]] = vals
            grids.append(g)
        grids.append(np.abs(grids[1] - grids[0]))
        vmin, vmax = np.nanmin(grids[0]), np.nanmax(grids[0])
        for q, (g, t) in enumerate(zip(grids, ("True", pretty(name), "|error|"))):
            kw = dict(vmin=vmin, vmax=vmax, cmap="viridis") if q < 2 else dict(cmap="magma")
            im = axes[r, q].imshow(g, aspect="auto", **kw)
            axes[r, q].set_title(f"{CHANNELS_OUT[c]} - {t}", fontsize=9)
            axes[r, q].grid(False)
            fig.colorbar(im, ax=axes[r, q], fraction=0.046)
    fig.suptitle(f"time step {ti_star}, layer K={k_star} (J rows x I columns)", fontsize=9)
    _save(fig, out, "fig_maps")


def fig_training(hist, out):
    _style()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(hist["train_loss"], label="Train loss")
    ax.plot(hist["val_loss"], label="Validation loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (sigma-normalised)")
    ax.legend()
    _save(fig, out, "fig_training_history")


def fig_rollout(roll, sigma, valid, out):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    n = roll["_n"].sum()
    for i, m in enumerate([k for k in roll if k != "_n"]):
        e = np.sqrt(roll[m] / n)                                      # (H,4)
        ax.plot(np.arange(1, len(e) + 1), np.sqrt(((e[:, valid] / sigma[valid]) ** 2).mean(1)),
                marker="o", color=PALETTE[i % len(PALETTE)], label=pretty(m))
    ax.set_xlabel("Rollout horizon (steps)")
    ax.set_ylabel("Macro NRMSE of predicted state change vs truth")
    ax.legend(fontsize=7)
    _save(fig, out, "fig_rollout")


# ============================================================
# ANALYSIS
# ============================================================

def column_volume_metrics(data: DynData, oof, sel: np.ndarray, methods, level: float, cfg: Config, out: Path):
    """Reservoir-level check: porosity-weighted saturation change summed over each (column j, time) slab
    (cell volumes are not in the npz, so this is a pore-volume proxy) and column-mean pressure change.
    Requires complete coverage of every test column (test_max = 0), otherwise it is skipped."""
    j, ti, k, i = data.j[sel], data.ti[sel], data.k[sel], data.i[sel]
    key = j.astype(np.int64) * data.T + ti
    uniq, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    full = (data.K - 2 * data.h) * (data.I - 2 * data.h)
    if not np.all(cnt == full):
        print("  [volume metrics] skipped: test columns are subsampled (use --test-max 0 for complete coverage)")
        return
    poro = data.static_raw[k, j, i, 1].astype(np.float64)
    gj = np.bincount(inv, weights=j.astype(np.float64)) / cnt          # column id of every slab
    ug = np.unique(gj)
    gidx = np.searchsorted(ug, gj)
    G = len(ug)
    chans = [(0, "dP_mean", None), (1, "dSO_pv", poro), (2, "dSG_pv", poro), (3, "dSW_pv", poro)]
    rng = np.random.default_rng(cfg.seed + 4242)
    Wb = rng.multinomial(G, np.full(G, 1.0 / G), size=500).astype(np.float64)       # cluster bootstrap over columns
    rows = []
    for c, name, w in chans:
        wt = np.ones(len(sel)) if w is None else w
        denom = cnt.astype(np.float64) if w is None else np.ones(len(cnt))
        yt = np.bincount(inv, weights=wt * data.Y[sel, c], minlength=len(cnt)) / denom
        var_t = float(np.var(yt))
        if var_t < 1e-18:
            continue
        for m in methods:
            pm = np.bincount(inv, weights=wt * oof[m][level][sel, c], minlength=len(cnt)) / denom
            e2 = (pm - yt) ** 2
            sse_g = np.bincount(gidx, weights=e2, minlength=G)
            n_g = np.bincount(gidx, minlength=G).astype(np.float64)
            nr = float(np.sqrt(sse_g.sum() / n_g.sum() / var_t))
            nb = np.sqrt((Wb @ sse_g) / (Wb @ n_g) / var_t)
            rows.append({"method": m, "level": level, "quantity": name, "n_slabs": len(cnt),
                         "nrmse": nr, "nrmse_ci_lo": float(np.percentile(nb, 2.5)), "nrmse_ci_hi": float(np.percentile(nb, 97.5)),
                         "r2": float(1 - sse_g.sum() / (var_t * len(cnt))), "bias": float(np.mean(pm - yt))})
    if rows:
        pd.DataFrame(rows).to_csv(out / "column_volume_metrics.csv", index=False)
        print("  [volume metrics] column_volume_metrics.csv written (pore-volume proxy, cluster-bootstrap CI)")


def loso_summary(data: DynData, oof, sel, methods, level, main, sigma, valid, cfg: Config, out: Path):
    """Per-held-out-scenario NRMSE and a scenario-level paired comparison (main vs every other method)."""
    scs = sorted(set(int(x) for x in data.sc[sel]))
    rows, nm = [], {}
    for m in methods:
        nm[m] = []
        for s_ in scs:
            r = sel[data.sc[sel] == s_]
            e = (oof[m][level][r] - data.Y[r]).astype(np.float64)
            mse = (e ** 2).mean(0)
            val = float((mse[valid] / (sigma[valid] ** 2)).mean())
            nm[m].append(val)
            rows.append({"method": m, "scenario": s_, "n": len(r), "nrmse": float(np.sqrt(val)),
                         **{f"{c}_r2": float(1 - (e[:, ci] ** 2).sum() / max(((data.Y[r][:, ci] - data.Y[r][:, ci].mean()) ** 2).sum(), 1e-300))
                            for ci, c in enumerate(CHANNELS_OUT)}})
    pd.DataFrame(rows).to_csv(out / "loso_by_scenario.csv", index=False)
    if not main or main not in nm:
        return
    rng = np.random.default_rng(cfg.seed + 2024)
    out_rows = []
    for o in methods:
        if o == main:
            continue
        d = np.array(nm[main]) - np.array(nm[o])                  # negative = main better (NMSE scale)
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(cfg.n_boot)])
        try:
            pw = float(sps.wilcoxon(d, zero_method="wilcox").pvalue)
        except ValueError:
            pw = float("nan")
        out_rows.append({"model": main, "vs": o, "n_scenarios": len(d), "mean_delta_nmse": float(d.mean()),
                         "ci_lo": float(np.percentile(bs, 2.5)), "ci_hi": float(np.percentile(bs, 97.5)),
                         "n_scen_main_better": int((d < 0).sum()), "p_wilcoxon": pw})
    pdf_ = pd.DataFrame(out_rows)
    if len(pdf_):
        pdf_["p_wilcoxon_holm"] = holm(pdf_["p_wilcoxon"].fillna(1.0).tolist())
        pdf_.to_csv(out / "loso_paired_by_scenario.csv", index=False)
    print(f"  [loso] per-scenario results: loso_by_scenario.csv ; scenario-level paired tests: "
          f"loso_paired_by_scenario.csv (n={len(scs)} held-out scenarios)")


def analyze(data: DynData, oof, cover, hists, roll, cfg: Config, out: Path, methods, levels, protocol: str):
    sel = np.where(~np.isnan(oof["persistence"][levels[0]][:, 0]))[0]
    for m in methods:
        for L in levels:
            if np.isnan(oof[m][L][sel]).any():
                raise RuntimeError(f"Missing OOF predictions for {m} at level {L}.")
    y = data.Y[sel]
    groups = data.j[sel]
    sigma = data.Y.std(0)
    valid = sigma > 1e-9
    thr = active_thresholds(data.Y[sel])
    cs = ClusterStats(y, groups, sigma, valid)
    rng = np.random.default_rng(cfg.seed + 777)
    W = cs.boot_counts(cfg.n_boot, rng)
    W1 = np.ones((1, cs.G))
    ref = cfg.ref if cfg.ref in methods else "persistence"
    main = main_name(methods, cfg)
    sat_state = data.D[data.ti[sel], data.k[sel], data.j[sel], data.i[sel]][:, 1:4]

    def _skill(mse, mse_r):
        ok = valid[None, :] & (mse_r / cs.sigma2[None, :] > 1e-6)
        return np.nanmean(np.where(ok, 1 - mse / np.maximum(mse_r, 1e-12), np.nan), axis=1)

    rows, paired = [], []
    for L in levels:
        sse = {m: cs.sse(y, oof[m][L][sel]) for m in methods}
        mse_ref_b = cs.metrics(sse[ref], W)[0]
        mse_ref_pt = cs.metrics(sse[ref], W1)[0]
        nm_g = {m: (sse[m][:, valid] / cs.sigma2[valid]).sum(1) for m in methods}
        for m in methods:
            mse_b, nrmse_b, _ = cs.metrics(sse[m], W)
            mse_p, nrmse_p, _ = cs.metrics(sse[m], W1)
            skill_b = _skill(mse_b, mse_ref_b)
            skill_p = float(_skill(mse_p, mse_ref_pt)[0])
            row = {"protocol": protocol, "level": L, "method": m, "n": len(sel),
                   "nrmse": float(nrmse_p[0]), "skill_vs_ref": skill_p}
            row["nrmse_ci_lo"], row["nrmse_ci_hi"] = ci(nrmse_b)
            row["skill_ci_lo"], row["skill_ci_hi"] = ci(skill_b)
            row.update(point_metrics(y, oof[m][L][sel], thr, valid))
            p = oof[m][L][sel]
            row["sat_sum_violation"] = float(np.abs(p[:, 1:4].sum(1)).mean())
            ns = sat_state + p[:, 1:4]
            row["sat_bound_violation"] = float(np.mean(((ns < -1e-3) | (ns > 1 + 1e-3)).any(1)))
            ca = cover[m][L]
            if ca.n:
                row["cov_abs"] = float((ca.h_abs[valid] / ca.n).mean())
                row["width_abs_norm"] = float((ca.w_abs[valid] / ca.n / sigma[valid]).mean())
                if ca.has_nrm:
                    row["cov_norm"] = float((ca.h_nrm[valid] / ca.n).mean())
                    row["width_norm_norm"] = float((ca.w_nrm[valid] / ca.n / sigma[valid]).mean())
            rows.append(row)
        if main:
            for o in [m for m in methods if m != main]:
                tot = W @ cs.n_g
                diff_b = (W @ (nm_g[main] - nm_g[o])) / (tot * cs.nv)
                d_pt = float((nm_g[main] - nm_g[o]).sum() / (cs.n_g.sum() * cs.nv))
                lo, hi = ci(diff_b)
                p_boot = float(min(1.0, 2 * min(np.mean(diff_b >= 0), np.mean(diff_b <= 0)) + 1.0 / len(W)))
                dg = (nm_g[main] - nm_g[o]) / (cs.n_g * cs.nv)
                try:
                    p_w = float(sps.wilcoxon(dg, zero_method="wilcox").pvalue)
                except ValueError:
                    p_w = float("nan")
                paired.append({"protocol": protocol, "level": L, "model": main, "vs": o, "delta_nmse": d_pt,
                               "ci_lo": lo, "ci_hi": hi, "p_boot": p_boot, "p_wilcoxon": p_w})
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics_dynamic.csv", index=False)
    column_volume_metrics(data, oof, sel, methods, levels[0], cfg, out)
    pdf = pd.DataFrame(paired)
    if len(pdf):
        for L in levels:
            m_ = pdf.level == L
            pdf.loc[m_, "p_boot_holm"] = holm(pdf.loc[m_, "p_boot"].tolist())
            pdf.loc[m_, "p_wilcoxon_holm"] = holm(pdf.loc[m_, "p_wilcoxon"].fillna(1.0).tolist())
        pdf.to_csv(out / "paired_tests.csv", index=False)

    d0 = df[df.level == levels[0]].set_index("method")
    order = sorted(methods, key=lambda x: d0.loc[x, "nrmse"])
    lines = [r"\begin{tabular}{lcccccc}", r"\hline",
             r"Method & NRMSE [95\% CI] & Skill vs ref. & $R^2$ dP & $R^2$ dSW & Cov.\ (90\%) & Sat.-sum viol. \\", r"\hline"]
    for m in order:
        r = d0.loc[m]
        cov = r.get("cov_abs", np.nan)
        lines.append(f"{pretty(m)} & {r.nrmse:.3f} [{r.nrmse_ci_lo:.3f}, {r.nrmse_ci_hi:.3f}] & "
                     f"{r.skill_vs_ref:.3f} & {r.dP_r2:.3f} & {r.dSW_r2:.3f} & "
                     f"{'' if np.isnan(cov) else f'{cov:.2f}'} & {r.sat_sum_violation:.1e} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "results_table.tex").write_text("\n".join(lines), encoding="utf-8")

    # per-channel table (macro NRMSE can hide channel-specific behaviour)
    ch_cols = ["method", "nrmse"] + [f"{c}_r2" for c in CHANNELS_OUT] + [f"{c}_rmse_active" for c in CHANNELS_OUT]
    d0.reset_index().loc[:, ch_cols].to_csv(out / "results_channels.csv", index=False)
    lines = [r"\begin{tabular}{lccccc}", r"\hline",
             r"Method & NRMSE & $R^2$ dP & $R^2$ dSO & $R^2$ dSG & $R^2$ dSW \\", r"\hline"]
    for m in order:
        r = d0.loc[m]
        lines.append(f"{pretty(m)} & {r.nrmse:.3f} & " + " & ".join(f"{r[f'{c}_r2']:.3f}" for c in CHANNELS_OUT) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    (out / "results_channels.tex").write_text("\n".join(lines), encoding="utf-8")

    np.savez_compressed(out / "oof_dynamic.npz", Y=data.Y, sel=sel, ti=data.ti, k=data.k, j=data.j, i=data.i,
                        **{f"{m}@{L}": oof[m][L] for m in methods for L in levels})

    fig_bars(df[df.level == levels[0]], out)
    if len(levels) > 1:
        fig_sparsity(df, out, main)
    curve = [m for m in ("persistence", "time_layer_mean", "rate_persistence", "extratrees_prev", "hgb_prev", main)
             if m and m in methods]
    fig_time_curve(data, oof, sel, out, curve, sigma, valid)
    if main:
        fig_parity(data, oof, sel, out, main)
        fig_maps(data, oof, sel, out, main)
        if hists:
            fig_training(hists[0], out)
    if roll:
        fig_rollout(roll, sigma, valid, out)
    if protocol == "loso":
        loso_summary(data, oof, sel, methods, levels[0], main, sigma, valid, cfg, out)

    print("\n" + "=" * 78)
    print(f"[{protocol}] POOLED OUT-OF-FOLD RESULTS | sparsity {levels[0]:.2f} | N={len(sel)} | ref={ref} | main={main}")
    print("=" * 78)
    for m in order:
        r = d0.loc[m]
        print(f"  {pretty(m):30s} NRMSE={r.nrmse:.4f} [{r.nrmse_ci_lo:.4f},{r.nrmse_ci_hi:.4f}] "
              f"skill={r.skill_vs_ref:+.3f} R2(dP,dSW)=({r.dP_r2:.3f},{r.dSW_r2:.3f})")
    if main and len(pdf):
        nc = d0[[not m.startswith("cnn") for m in d0.index]].sort_values("nrmse")
        if len(nc):
            b = nc.index[0]
            row = pdf[(pdf.level == levels[0]) & (pdf.vs == b)].iloc[0]
            print(f"\n  main ({pretty(main)}) vs best non-CNN ({pretty(b)}; picked post hoc on OOF - descriptive only):")
            print(f"    delta NMSE = {row.delta_nmse:+.4f} [{row.ci_lo:+.4f}, {row.ci_hi:+.4f}]  "
                  f"(negative = main better)  Holm p (boot) = {row.p_boot_holm:.3f}")
    return df


# ============================================================
# EXPERIMENT
# ============================================================

_FP_KEYS = ["patch", "n_folds", "val_width", "seed", "n_ens", "epochs", "patience", "batch_size", "epoch_samples",
            "lr", "weight_decay", "dropout", "width", "max_val", "baseline_max_train", "enforce_sum", "tta", "loss",
            "active_w", "n_hist", "sparsity", "train_drop_max", "alpha", "t_cut_frac", "test_max",
            "rollout_horizon", "rollout_starts"]


def fingerprint(cfg: Config, protocol: str, tag: str) -> str:
    d = {k: getattr(cfg, k) for k in _FP_KEYS}
    d["rollout_horizon"], d["rollout_starts"] = 0, 3      # roll-out has its own cache key (keeps old checkpoints valid)
    if cfg.tune_cnn:
        d["tune_cnn"] = True
    if cfg.sub_per_scen > 0:
        d["sub_per_scen"] = cfg.sub_per_scen
    d.update(protocol=protocol, tag=tag, version=CODE_VERSION)
    return hashlib.md5(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:10]


def protocol_indices(data: DynData, f: Dict, protocol: str, t_cut_frac: float):
    """Train/val/test sample indices of one fold.
    spatial_temporal: train/val targets stop at D[t_cut]; test inputs start at t_cut+1 (strict temporal boundary)."""
    if protocol == "loso":                      # leave-one-scenario-out: whole scenarios are train / val / test
        assert data.nS >= 3, "loso needs at least 3 scenarios (train, validation, test)"
        return (data.T_loc - 2, data.idx_scen(f["train"]), data.idx_scen(f["val"]), data.idx_scen(f["test"]))
    assert data.nS == 1, "the spatial protocols are defined for a single scenario; use --protocols loso for several"
    if protocol == "spatial_temporal":
        t_cut = int(t_cut_frac * (data.T - 2))
        return (t_cut, data.idx_for(f["train"], 0, t_cut - 1), data.idx_for(f["val"], 0, t_cut - 1),
                data.idx_for(f["test"], t_cut + 1, data.T - 2))
    t_cut = data.T - 2
    return t_cut, *(data.idx_for(f[s_], 0, data.T - 2) for s_ in ("train", "val", "test"))


def make_loso_folds(n_scen: int, n_run: int = 0) -> List[Dict]:
    """Held-out scenario s is tested; scenario (s+1) mod S is used for early stopping / conformal calibration /
    tuning; every other scenario is training data. Scenario-level split: no sample of a test scenario is seen."""
    if n_scen < 3:
        raise ValueError("loso needs at least 3 scenarios")
    folds = []
    for s_ in range(n_scen):
        v_ = (s_ + 1) % n_scen
        folds.append({"train": [x for x in range(n_scen) if x not in (s_, v_)], "val": [v_], "test": [s_]})
    if 0 < n_run < n_scen:
        keep = sorted(set(np.linspace(0, n_scen - 1, n_run).round().astype(int).tolist()))
        folds = [folds[i] for i in keep]
    for f in folds:
        assert not (set(f["train"]) & set(f["val"])) and not (set(f["train"]) & set(f["test"])) \
            and not (set(f["val"]) & set(f["test"])), "scenario overlap between train/val/test"
    return folds


def run_protocol(data: DynData, cfg: Config, out: Path, protocol: str, folds, device, tag: str) -> pd.DataFrame:
    out.mkdir(parents=True, exist_ok=True)
    ckdir = out / "ckpt"
    ckdir.mkdir(exist_ok=True)
    fp = fingerprint(cfg, protocol, tag)
    levels = [float(x) for x in cfg.sparsity]
    use_cnn = HAS_TORCH and not cfg.skip_cnn
    variants = list(cfg.ablations) if use_cnn else []
    specs = {v: resolve_spec(v, cfg) for v in variants}
    mname = {v: cnn_name(v) for v in variants}
    methods = BASELINES + [mname[v] for v in variants]
    if use_cnn and cfg.main not in methods:
        print(f"WARNING: --main {cfg.main} is not among the trained methods {methods[len(BASELINES):]}; "
              f"paired tests will use the first CNN.")
    N = data.N
    oof = {m: {L: np.full((N, 4), np.nan, np.float32) for L in levels} for m in methods}
    sig = {m: {L: np.full((N, 4), np.nan, np.float32) for L in levels} for m in methods}
    cover = {m: {L: CoverAcc() for L in levels} for m in methods}
    hists: List[Dict] = []
    roll_acc: Dict[str, np.ndarray] = {}
    roll_n = 0
    rng_master = np.random.default_rng(cfg.seed)
    project = bool(cfg.enforce_sum and data.sat_ok)
    do_roll = cfg.rollout_horizon > 0 and protocol == "spatial"

    def ck(fi, group, suffix=".pkl"):
        return ckdir / f"{protocol}_f{fi}_{group}_{fp}{suffix}"

    for fi, f in enumerate(folds):
        t_cut, tr, va, te = protocol_indices(data, f, protocol, cfg.t_cut_frac)
        if cfg.test_max > 0 and len(te) > cfg.test_max:
            te = np.sort(np.random.default_rng(cfg.seed + 31 * fi + 7).choice(te, cfg.test_max, replace=False))
        print(f"\n=== [{protocol}] fold {fi + 1}/{len(folds)} | train={len(tr)} val={len(va)} test={len(te)} ===")
        if min(len(tr), len(va), len(te)) < 10:
            raise RuntimeError("Fold too small; adjust n_folds / t_cut_frac / test_max.")
        va_sub = np.sort(rng_master.choice(va, size=min(len(va), cfg.max_val), replace=False))
        if protocol == "loso":
            st = compute_stats(data, data.centers_j, data.T - 1, tr,
                               steps=np.where(np.isin(data.sc_of_step, f["train"]))[0])
        else:
            st = compute_stats(data, f["train"], t_cut if protocol == "spatial_temporal" else data.T - 1, tr)
        src = Source.from_full(data, st)
        if use_cnn:
            add_resid_stats(st, data, src, tr)
        fold_cov: Dict[Tuple[str, float], CoverAcc] = {}

        def evaluate(name, pred_fn, center=True, members_fn=None):
            for L in levels:
                hv = data.hidden(va_sub, L, center)
                ht = data.hidden(te, L, center)
                if members_fn is None:
                    pv = pred_fn(src, data.ti[va_sub], data.tl[va_sub], data.k[va_sub], data.j[va_sub],
                                 data.i[va_sub], hv)
                    pt = pred_fn(src, data.ti[te], data.tl[te], data.k[te], data.j[te], data.i[te], ht)
                    sv = stt = None
                else:
                    mv = members_fn(src, data.ti[va_sub], data.k[va_sub], data.j[va_sub], data.i[va_sub], hv)
                    mt = members_fn(src, data.ti[te], data.k[te], data.j[te], data.i[te], ht)
                    pv, pt = mv.mean(0), mt.mean(0)
                    sv = mv.std(0, ddof=1) if len(mv) > 1 else None
                    stt = mt.std(0, ddof=1) if len(mt) > 1 else None
                oof[name][L][te] = pt
                if stt is not None:
                    sig[name][L][te] = stt
                cf = CoverAcc()
                cf.add(data.Y[va_sub], pv, data.Y[te], pt, sv, stt, cfg.alpha)
                cover[name][L].merge(cf)
                fold_cov[(name, L)] = cf

        def snapshot(names):
            return {(m, L): dict(pred=oof[m][L][te].copy(), sig=sig[m][L][te].copy(), cov=fold_cov[(m, L)].state())
                    for m in names for L in levels}

        def restore(snap):
            for (m, L), d in snap.items():
                oof[m][L][te] = d["pred"]
                sig[m][L][te] = d["sig"]
                cf = CoverAcc.from_state(d["cov"])
                cover[m][L].merge(cf)
                fold_cov[(m, L)] = cf

        roll_ck = ck(fi, f"rollout_h{cfg.rollout_horizon}_s{cfg.rollout_starts}")
        roll_cached = do_roll and cfg.resume and roll_ck.exists()
        need_roll = do_roll and not roll_cached

        # ---- baselines ----
        base = None
        bl_ck = ck(fi, "baselines")
        if cfg.resume and bl_ck.exists():
            restore(pickle.load(open(bl_ck, "rb")))
            print("  baselines restored from checkpoint")
        else:
            t0 = time.time()
            base = build_baselines(data, src, st, tr, va_sub, cfg, levels, cfg.seed + fi)
            for m, fn in base.items():
                evaluate(m, fn)
            pickle.dump(snapshot(BASELINES), open(bl_ck, "wb"))
            print(f"  baselines done ({time.time() - t0:.0f}s)")

        # ---- CNN variants ----
        cnn_models: Dict[str, "CNNEnsemble"] = {}
        cfg_f = cfg
        if use_cnn:
            st_t = stats_tensors(st, device)
            if cfg.tune_cnn:
                tk = ck(fi, "tune")
                if cfg.resume and tk.exists():
                    hp = pickle.load(open(tk, "rb"))
                else:
                    mv_ = next((v for v in variants if mname[v] == cfg.main), variants[0])
                    cands = [(cfg.lr, cfg.dropout), (2 * cfg.lr, cfg.dropout), (0.5 * cfg.lr, cfg.dropout),
                             (cfg.lr, 0.05)]
                    scores = []
                    for lr_, do_ in cands:      # one member per candidate; selection uses the VAL block only
                        _, hh = train_member(data, src, st, st_t, tr, va_sub, dataclasses.replace(cfg, lr=lr_, dropout=do_),
                                             specs[mv_], cfg.seed + 500 + 1000 * fi, device, project and specs[mv_]["phys"])
                        scores.append(min(hh["val_loss"]))
                    b_ = int(np.argmin(scores))
                    hp = {"lr": cands[b_][0], "dropout": cands[b_][1], "cands": cands, "val_scores": scores}
                    pickle.dump(hp, open(tk, "wb"))
                print(f"  CNN tuning (val only): lr={hp['lr']:.2e} dropout={hp['dropout']:.2f}")
                cfg_f = dataclasses.replace(cfg, lr=hp["lr"], dropout=hp["dropout"])
            for v in variants:
                spec, mn = specs[v], mname[v]
                proj = project and spec["phys"]
                pk, mk = ck(fi, mn), ck(fi, mn, ".pt")
                models = None
                if cfg.resume and pk.exists() and mk.exists():
                    blob = pickle.load(open(pk, "rb"))
                    restore(blob["snap"])
                    if fi == 0 and mn == cfg.main and blob["hist"] is not None:
                        hists.append(blob["hist"])
                    if need_roll and v in ("full", "prev", "hist", "resid"):
                        models = load_models(mk, spec, data, cfg, device)
                    print(f"  {mn:22s} restored from checkpoint")
                else:
                    t1 = time.time()
                    models, first_hist = [], None
                    for mm in range(cfg.n_ens):
                        model, hist = train_member(data, src, st, st_t, tr, va_sub, cfg_f, spec,
                                                   cfg.seed + 1000 * fi + mm, device, proj)
                        models.append(model)
                        if first_hist is None:
                            first_hist = hist
                    ens = CNNEnsemble(models, spec, st_t, data.h, cfg, device, proj)
                    evaluate(mn, ens, spec["center"],
                             members_fn=lambda s_, ti, k, j, i, hid, e=ens: e.members(s_, ti, k, j, i, hid))
                    save_models(models, mk)
                    pickle.dump({"snap": snapshot([mn]), "hist": first_hist}, open(pk, "wb"))
                    if fi == 0 and mn == cfg.main:
                        hists.append(first_hist)
                    r = point_metrics(data.Y[te], oof[mn][levels[0]][te], active_thresholds(data.Y[te]),
                                      data.Y.std(0) > 1e-9)
                    print(f"  {mn:22s} fold dSW RMSE={r['dSW_rmse']:.6f} ({time.time() - t1:.0f}s)")
                if models is not None:
                    cnn_models[v] = CNNEnsemble(models, spec, st_t, data.h, cfg, device, proj)

        # ---- rollout ----
        if do_roll:
            if roll_cached:
                blob = pickle.load(open(roll_ck, "rb"))
                res_all, n_all = blob["roll"], blob["n"]
            else:
                if base is None:
                    base = build_baselines(data, src, st, tr, va_sub, cfg, levels, cfg.seed + fi)
                H = cfg.rollout_horizon
                starts = sorted(set(np.linspace(0, data.T - 1 - H, cfg.rollout_starts).round().astype(int).tolist()))
                rp = {k_: base[k_] for k_ in ("persistence", "time_layer_mean", "rate_persistence",
                                              "extratrees_prev", "hgb_prev")}
                for v, e in cnn_models.items():
                    rp[mname[v]] = e
                res_all, n_all = {}, 0
                for t0_ in starts:
                    if t0_ + H > data.T - 1 or t0_ < 0:
                        continue
                    res = run_rollout(data, st, f["test"], t0_, H, rp, max(1, cfg.n_hist))
                    n_all += int(res.pop("_n")[0])
                    for k_, v_ in res.items():
                        res_all[k_] = res_all.get(k_, 0) + v_
                pickle.dump({"roll": res_all, "n": n_all}, open(roll_ck, "wb"))
                print(f"  rollout done (H={H}, starts={starts})")
            roll_n += n_all
            for k_, v_ in res_all.items():
                roll_acc[k_] = roll_acc.get(k_, 0) + v_

    roll = None
    if roll_acc:
        roll = dict(roll_acc)
        roll["_n"] = np.array([roll_n])
    df = analyze(data, oof, cover, hists, roll, cfg, out, methods, levels, protocol)
    if roll:
        rr = [{"method": k_, "horizon": h_ + 1, **{CHANNELS_OUT[c]: float(np.sqrt(v[h_, c] / roll_n)) for c in range(4)}}
              for k_, v in roll.items() if k_ != "_n" for h_ in range(v.shape[0])]
        pd.DataFrame(rr).to_csv(out / "rollout_rmse.csv", index=False)
    return df


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _write_manifest(odir: Path, npz, cfg: Config, env: Dict) -> None:
    """Reproducibility record: hashes of the exact script and data used, versions and full configuration."""
    files = [Path(q) for q in npz] if isinstance(npz, (list, tuple)) else [Path(npz)]
    man = {"code_version": CODE_VERSION, "script": Path(__file__).name, "script_sha256": _sha256(Path(__file__)),
           "data_file": files[0].name if len(files) == 1 else [q.name for q in files],
           "data_sha256": _sha256(files[0]) if len(files) == 1 else {q.name: _sha256(q) for q in files},
           "env": env, "config": asdict(cfg)}
    (odir / "manifest.json").write_text(json.dumps(man, indent=2, default=str), encoding="utf-8")


def run_loso(cfg: Config, out: Path, env: Dict, device) -> None:
    """Leave-one-scenario-out over several scenarios of the same reservoir (different well controls)."""
    if cfg.toy:
        paths = make_toy_multi(out / "toy_multi", n_scen=4, seed=cfg.seed)
        print(f"[toy] synthetic multi-scenario pipeline-test data written to {out / 'toy_multi'} (NOT physical)")
    else:
        paths = [Path(cfg.data_dir) / n_ for n_ in cfg.npz_names]
        for q in paths:
            if not q.exists():
                raise FileNotFoundError(f"Missing {q}")
    if len(paths) < 3:
        raise ValueError("--protocols loso needs --npz-names with at least 3 scenario files")
    odir = out / "loso"
    odir.mkdir(parents=True, exist_ok=True)
    _write_manifest(odir, paths, cfg, env)
    print(f"\n##### LOSO over {len(paths)} scenarios: {[q.name for q in paths]}")
    data = DynData(paths, cfg.patch, cfg.seed, cfg.sub_per_scen)
    folds = make_loso_folds(data.nS, cfg.loso_folds)
    for fi, f in enumerate(folds):
        print(f"  fold {fi}: train scenarios={f['train']} val={f['val']} test={f['test']}")
    tag = "loso_" + hashlib.md5("|".join(q.name for q in paths).encode()).hexdigest()[:6] + f"_f{cfg.loso_folds}"
    run_protocol(data, cfg, odir, "loso", folds, device, tag)
    print(f"\nOutputs written to {out}")


def run_experiment(cfg: Config) -> None:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if HAS_TORCH and cfg.threads > 0:
        torch.set_num_threads(cfg.threads)
    device = get_device()
    print(f"Device: {device} | torch: {HAS_TORCH} | code v{CODE_VERSION}")
    if HAS_TORCH and not cfg.skip_cnn and device is not None and getattr(device, "type", "cpu") == "cpu" \
            and not (cfg.fast or cfg.quick or cfg.toy):
        print("WARNING: running the full configuration on CPU. Measured on one CPU laptop (v3, default settings): "
              "~70 min for ONE variant in ONE fold, i.e. more than a day for the default variant list. "
              "Prefer the default CPU preset (--fast), or run --full on a GPU (add --tta there). Checkpoints let you resume.")
    env = {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
           "torch": getattr(torch, "__version__", None)}
    (out / "config.json").write_text(json.dumps({"config": asdict(cfg), "env": env}, indent=2), encoding="utf-8")

    std_protocols = [p_ for p_ in cfg.protocols if p_ != "loso"]
    if "loso" in cfg.protocols:
        run_loso(cfg, out, env, device)
    jobs: List[Tuple[Path, Path, str]] = []
    if not std_protocols:
        print(f"\nOutputs written to {out}")
        return
    if cfg.toy:
        npz = out / "toy_dynamic.npz"
        make_toy_npz(npz, seed=cfg.seed, inactive=cfg.toy_inactive)
        print(f"[toy] synthetic pipeline-test data written to {npz} (NOT physical)")
        jobs.append((npz, out, "toy"))
    else:
        for name in cfg.npz_names:
            p = Path(cfg.data_dir) / name
            if not p.exists():
                raise FileNotFoundError(f"Missing {p}")
            jobs.append((p, out if len(cfg.npz_names) == 1 else out / p.stem, p.stem))

    for npz, odir, tag in jobs:
        print(f"\n##### dataset: {npz}")
        odir.mkdir(parents=True, exist_ok=True)
        _write_manifest(odir, npz, cfg, env)
        data = DynData(npz, cfg.patch, cfg.seed, cfg.sub_per_scen)
        folds = make_folds(data.centers_j, cfg.n_folds, cfg.val_width, data.h,
                             col_weight=None if data.active is None else data.col_n)
        for fi, f in enumerate(folds):
            print(f"  fold {fi}: train J={f['train']} val={f['val']} test={f['test']}")
        for proto in std_protocols:
            run_protocol(data, cfg, odir / proto, proto, folds, device, tag)
    print(f"\nOutputs written to {out}")


# ============================================================
# CLI
# ============================================================

def parse_args() -> Config:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = Config()
    for k, v in asdict(d).items():
        flag = "--" + k.replace("_", "-")
        if isinstance(v, bool):
            if v:
                ap.add_argument("--no-" + k.replace("_", "-"), dest=k, action="store_false")
            else:
                ap.add_argument(flag, dest=k, action="store_true")
        elif isinstance(v, list):
            ap.add_argument(flag, nargs="+", type=type(v[0]), default=v)
        else:
            ap.add_argument(flag, type=type(v), default=v)
    a = vars(ap.parse_args())
    a["data_dir"] = str(Path(a["data_dir"]).resolve())
    a["out_dir"] = str(Path(a["out_dir"]).resolve())
    return Config(**a)


def main() -> None:
    cfg = parse_args()
    default_abl = Config().ablations
    if cfg.medium:
        cfg.n_ens, cfg.epochs, cfg.patience, cfg.epoch_samples, cfg.width = 5, 40, 8, 20000, 16
        cfg.sparsity, cfg.tta, cfg.test_max, cfg.n_boot = [0.0, 0.5], False, 40000, 2000
        cfg.baseline_max_train, cfg.max_val = 30000, 4000
        if cfg.ablations == default_abl:
            cfg.ablations = ["full", "prev", "resid"]
    if (HAS_TORCH and not cfg.full and not (cfg.medium or cfg.fast or cfg.quick or cfg.toy or cfg.skip_cnn)
            and not torch.cuda.is_available()):
        cfg.fast = True
        print("[auto] No GPU found -> using the CPU-friendly --fast preset (add --full to force the heavy run).")
    if cfg.fast:
        cfg.rollout_horizon = min(cfg.rollout_horizon, 5)
        cfg.n_ens, cfg.epochs, cfg.patience, cfg.epoch_samples, cfg.width = 3, 25, 5, 8000, 8
        cfg.sparsity, cfg.tta, cfg.test_max, cfg.n_boot = [0.0, 0.5], False, 20000, 1000
        cfg.baseline_max_train, cfg.max_val = 15000, 2000
        if cfg.ablations == default_abl:
            cfg.ablations = ["full", "prev", "resid"]
    if cfg.quick:
        cfg.rollout_horizon = 0
        cfg.n_ens, cfg.epochs, cfg.patience, cfg.n_boot = 2, 5, 3, 300
        cfg.sparsity, cfg.tta = [0.0, 0.5], False
        cfg.epoch_samples, cfg.baseline_max_train, cfg.max_val, cfg.test_max = 3000, 6000, 1000, 6000
        cfg.protocols = ["spatial"]
        if cfg.ablations == default_abl:
            cfg.ablations = ["resid"]
    set_seed(cfg.seed)
    run_experiment(cfg)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    main()