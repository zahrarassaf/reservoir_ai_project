#!/usr/bin/env bash
# Run the Norne benchmark deck with OPM Flow and inspect its output (Ubuntu, project folder, ~/v activated).
# Usage:  bash run_norne.sh [path/to/norne]      (default: ./norne  = the `norne` folder of opm-data, with INCLUDE inside)
set -e
NORNE=${1:-norne}
[ -f "$NORNE/NORNE_ATW2013.DATA" ] || { echo "missing $NORNE/NORNE_ATW2013.DATA"; exit 1; }
[ -d "$NORNE/INCLUDE" ] || { echo "missing $NORNE/INCLUDE (copy the WHOLE norne folder, about 31 MB)"; exit 1; }
rm -rf norne_run && mkdir -p norne_run && cp -r "$NORNE"/NORNE_ATW2013.DATA "$NORNE"/INCLUDE norne_run/
cd norne_run
echo "== starting Flow on Norne at $(date +%T) (this may take a while; log: norne_run/flow.log)"
START=$(date +%s)
flow NORNE_ATW2013.DATA --output-dir=out > flow.log 2>&1 || { echo "Flow FAILED"; tail -n 30 flow.log; exit 1; }
echo "== Flow finished in $(( $(date +%s) - START )) s"
grep -c "Restart file written" flow.log || true
tail -n 15 flow.log
ls -la out | head -20
cd ..
python inspect_flow_output.py norne_run/out NORNE_ATW2013 > norne_inspect.txt 2>&1 || true
echo "== inspection written to norne_inspect.txt (send me this file and the last 15 lines above)"
