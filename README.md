
Each module is independent (no local imports between them); they communicate
through files on disk (`*.npz`, `*.json`, `*.csv`).

## Quick start

    pip install -r requirements.txt

    # 1. place SPE9.DATA, PERMVALUES.DATA, TOPSVALUES.DATA, SPE9.GRDECL in data/
    #    (obtain from https://github.com/OPM/opm-data/tree/master/spe9)

    # 2. generate 12 well-control scenarios and run OPM Flow
    bash run_scenarios.sh 12

    # 3. run the ML benchmark (CPU-friendly preset)
    python cnn_dynamic_spe9.py --fast

    # 4. multi-scenario leave-one-out
    python cnn_dynamic_spe9.py --protocols loso \
        --npz-names ref.npz s00.npz s01.npz ... s11.npz

    # 5. Monte Carlo uncertainty (uses real permeability/thickness data)
    python uncertainty_analysis.py

    # 6. reduced-order digital twin (schedule-only, no history file needed)
    python digital_twin.py --project-dir .

    # 7. second field (Norne)
    bash run_norne.sh path/to/norne

    # 8. causality / leakage unit tests
    python test_no_leakage.py

## Files

| File | Role |
|---|---|
| `cnn_dynamic_spe9.py` | 3D-CNN surrogate + 13 baselines + all statistics |
| `build_dataset.py` | OPM Flow output → NPZ (SPE9) |
| `build_norne.py` | OPM Flow output → NPZ (Norne, active-cell expansion) |
| `inspect_flow_output.py` | Diagnostic — verifies Flow output before training |
| `make_spe9_scenarios.py` | Generate 12 scenario decks from SPE9.DATA |
| `digital_twin.py` | Reduced-order two-phase simulator (IMPES + EnKF) |
| `uncertainty_analysis.py` | Monte Carlo economic analysis (1000 realizations) |
| `test_no_leakage.py` | 14 causality and leakage unit tests |
| `rl_reservoir_optimizer.py` | Reinforcement-learning prototype (see Limitations) |
| `run_scenarios.sh` | Automated pipeline: decks → Flow → NPZ |
| `rebuild_npz.sh` | Rebuild NPZ from existing Flow output |
| `run_norne.sh` | Run the Norne benchmark deck + inspect |

## Reproducibility

Every run writes `manifest.json` with SHA-256 hashes of the script and
dataset, the full configuration, and environment versions. Runs are
checkpointed per (protocol, fold, method-group) and resume automatically.

## Limitations

See `docs/LIMITATIONS.md` for the full list. In short:

- The 3D-CNN is a **one-step surrogate**, not a replacement for a full
  simulator. Multi-step rollout is boundary-forced.
- Conformal intervals are slightly optimistic (calibrated on the same
  block used for early stopping).
- The reduced-order twin models two phases only (no gas, no Rs,
  no undersaturated oil).
- `rl_reservoir_optimizer.py` uses a **synthetic** environment with 4 wells
  and a simplified DQN. It is a **proof-of-concept**, not a production
  controller.
- The Norne NPZ does not include well locations or controls, so only the
  strict `spatial` protocol applies to that field.

## Acknowledgements

- **SPE9**: Killough, 1995 — 9th SPE Comparative Solution Project
- **Norne**: Equinor open data set
- **OPM Flow**: Open Porous Media initiative
- `resdata` library for restart-file I/O
