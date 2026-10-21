#!/usr/bin/env bash
# Usage (inside Ubuntu, in the project folder, with ~/v activated):  bash run_scenarios.sh [N=12]
# 1) reproduces the ORIGINAL SPE9 deck with per-step output -> ref.npz (sanity check vs your spe9_dynamic_dataset.npz)
# 2) generates N scenarios, runs Flow on each, builds s00.npz ... sNN.npz
set -e
N=${1:-12}
for f in SPE9.DATA PERMVALUES.DATA TOPSVALUES.DATA SPE9.GRDECL make_spe9_scenarios.py build_dataset.py; do
  [ -f "$f" ] || { echo "missing $f in $(pwd)"; exit 1; }
done
rm -rf scen ref
mkdir -p ref
sed -e "s/'BASIC=4'/'BASIC=2'/" -e "0,/^GRID[[:space:]]*\$/s//GRID\nINIT/" SPE9.DATA > ref/SPE9_REF.DATA
cp PERMVALUES.DATA TOPSVALUES.DATA SPE9.GRDECL ref/
echo "== reference run"; flow ref/SPE9_REF.DATA --output-dir=ref > ref/flow.log 2>&1
python build_dataset.py ref/SPE9_REF.DATA ref ref.npz 2>&1 | grep -v -i deprecat | grep -v warnings.warn
python make_spe9_scenarios.py SPE9.DATA scen "$N" 0
for d in scen/SCEN_*; do
  s=$(basename "$d"); k=${s#SCEN_}
  cp PERMVALUES.DATA TOPSVALUES.DATA SPE9.GRDECL "$d"/
  echo "== $s"; flow "$d/SPE9_$s.DATA" --output-dir="$d" > "$d/flow.log" 2>&1 || { echo "Flow FAILED for $s, see $d/flow.log"; continue; }
  python build_dataset.py "$d/SPE9_$s.DATA" "$d" "s$k.npz" 2>&1 | grep -v -i deprecat | grep -v warnings.warn
done
python - <<'PY'
import numpy as np, glob, os
cand = [p for p in glob.glob("*spe9_dynamic_dataset*.npz")] + glob.glob("data/*spe9_dynamic_dataset*.npz")
if cand and os.path.exists("ref.npz"):
    a, b = np.load(cand[0]), np.load("ref.npz")
    print("\nSANITY vs", cand[0])
    print(" permx equal:", np.allclose(a["permx"], b["permx"], rtol=1e-4), "| poro equal:", np.allclose(a["poro"], b["poro"], rtol=1e-4))
    for k in ["pressure", "so", "sg", "sw"]:
        print(f" {k}: max|diff|={np.abs(a[k]-b[k]).max():.4g}  (range {a[k].min():.3g}..{a[k].max():.3g})")
else:
    print("sanity check skipped (put spe9_dynamic_dataset.npz in this folder or data/)")
PY
ls -la s*.npz ref.npz