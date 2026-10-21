"""
Generate N SPE9 scenario decks (different well controls) from the OPM SPE9.DATA.
Usage:  python make_spe9_scenarios.py <path/to/opm-data/spe9/SPE9.DATA> <outdir> [N=12] [seed=0]
Then run each deck with OPM Flow:  flow <outdir>/SCEN_xx/SPE9_SCEN_xx.DATA --output-dir=<outdir>/SCEN_xx/out
Scenario knobs (all recorded in <outdir>/scenarios.json -> use as `controls` / `well_mask` inputs):
  - injector rate, producer oil-rate schedule (3 phases), BHP limits,
  - random shut-in of 0-5 producers (well_mask), optional second injector.
The grid/rock/PVT are unchanged (same reservoir, different operations) -> leave-one-scenario-out protocol.
"""
import sys, os, re, json, random

def main():
    src, out = sys.argv[1], sys.argv[2]
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 12
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    rng = random.Random(seed)
    txt = open(src).read()
    head = txt[:txt.index("WCONINJE")]
    # SPE9 writes restart output only yearly (BASIC=4); the CNN needs every report step (BASIC=2).
    assert "'BASIC=4'" in head, "RPTRST 'BASIC=4' not found in deck"
    head = head.replace("'BASIC=4'", "'BASIC=2'")
    # the deck has no INIT keyword -> Flow writes no .INIT file (PERMX/PORO), which build_dataset.py reads
    import re as _re
    assert _re.search(r"^GRID\s*$", head, _re.M), "GRID section not found"
    head = _re.sub(r"^GRID\s*$", "GRID\nINIT", head, count=1, flags=_re.M)
    # keep WELSPECS/COMPDAT (in head); replace everything from WCONINJE on
    meta = []
    for s in range(N):
        inj = rng.choice([2500, 3500, 5000, 6500, 8000])
        q1, q2, q3 = rng.choice([800, 1200, 1500, 2000]), rng.choice([100, 300, 600]), rng.choice([800, 1200, 1500, 2000])
        bhp = rng.choice([800, 1000, 1500])
        t1, t2 = rng.choice([20, 30, 40]), rng.choice([4, 6, 10])
        shut = sorted(rng.sample(range(2, 27), rng.randint(0, 5)))
        sched = ["WCONINJE", f"  'INJE1' 'WATER' 'OPEN' 'RATE' {inj} 1* 4000 /", "/", "",
                 "WCONPROD", f"  'PRODU*' 'OPEN' 'ORAT' {q1} 4* {bhp} /", "/"]
        if shut:
            sched += ["", "WELOPEN"] + [f"  'PRODU{k}' 'SHUT' /" for k in shut] + ["/"]
        tot = 90
        t3 = tot - t1 - t2
        sched += ["", f"TSTEP\n{t1}*10 /", "", "WCONPROD", f"  'PRODU*' 'OPEN' 'ORAT' {q2} 4* {bhp} /", "/"]
        if shut:  # re-apply shut-ins after the blanket WCONPROD re-opens them
            sched += ["WELOPEN"] + [f"  'PRODU{k}' 'SHUT' /" for k in shut] + ["/"]
        sched += ["", f"TSTEP\n{t2}*10 /", "", "WCONPROD", f"  'PRODU*' 'OPEN' 'ORAT' {q3} 4* {bhp} /", "/"]
        if shut:
            sched += ["WELOPEN"] + [f"  'PRODU{k}' 'SHUT' /" for k in shut] + ["/"]
        sched += ["", f"TSTEP\n{t3}*10 /", "", "END", ""]
        d = os.path.join(out, f"SCEN_{s:02d}"); os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, f"SPE9_SCEN_{s:02d}.DATA")
        # INCLUDE paths in the original are relative -> copy needed files next to deck
        open(fn, "w").write(head + "\n".join(sched))
        meta.append(dict(scenario=s, inj_rate=inj, prod_rates=[q1, q2, q3], phase_steps=[t1, t2, t3],
                         bhp_min=bhp, shut_in=shut))
    json.dump(meta, open(os.path.join(out, "scenarios.json"), "w"), indent=1)
    print(f"wrote {N} decks to {out}. NOTE: copy PERMVALUES.DATA/TOPSVALUES.DATA/SPE9.GRDECL beside each deck "
          "(or keep INCLUDE paths absolute).")

if __name__ == "__main__":
    main()