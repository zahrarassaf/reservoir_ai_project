"""
Build a CNN-ready .npz from an OPM Flow run (SPE9 or one scenario).
Usage:
  pip install resdata numpy
  python build_dataset.py <DECK.DATA> <flow_output_dir> <out.npz>
Flow writes DECK.EGRID / DECK.INIT / DECK.UNRST in the output dir (needs RPTRST in the deck, which SPE9 has).
Optional 4th/5th arguments add the operating controls of a scenario (needed for multi-scenario runs):
  python build_dataset.py <DECK.DATA> <flow_output_dir> <out.npz> <scenarios.json> <index|ref>
  (`ref` = the original SPE9 schedule: 5000 STB/d injection; producers 1500 -> 100 (steps 30-36) -> 1500)
Keys written (same layout cnn_dynamic_spe9.py expects):
  permx, poro (K,J,I); pressure, so, sg, sw (T,K,J,I); times (T,) days;
  well_mask (K,J,I) 0/1 = completed cells; well_type (K,J,I) = +1 injector, -1 producer, 0 none;
  ctrl (T,K,J,I,2) (only with a scenario): ch0 = injection rate / 8000 at injector cells, ch1 = oil-rate target /
  2000 at OPEN producer cells (0 for shut-in wells); value at step t = control applied during interval t -> t+1.
NOTE: the well-location parser and the UNRST reader were written without a real Flow run to test against;
run a short check (T, grid shape, sum of saturations) before trusting the file.
"""
import sys
from pathlib import Path
import numpy as np


def parse_compdat(deck: Path):
    """Return {well: (i, j, k1, k2)} from the COMPDAT block (1-based indices)."""
    lines = [l.split("--")[0].strip() for l in deck.read_text().splitlines()]
    wells, on = {}, False
    for l in lines:
        if l == "COMPDAT":
            on = True
            continue
        if on:
            if l == "/":
                break
            t = l.split()
            if len(t) >= 5 and t[0].startswith("'"):
                wells[t[0].strip("'")] = tuple(int(x) for x in t[1:5])
    return wells


INJ_NORM, PROD_NORM = 8000.0, 2000.0
REF_SCENARIO = dict(inj_rate=5000, prod_rates=[1500, 100, 1500], phase_steps=[30, 6, 54], shut_in=[])


def build_ctrl(wells, params, T, shape):
    """Operating controls as spatial channels. Fixed physical normalisation constants (no data-dependent scaling)."""
    nz, ny, nx = shape
    ctrl = np.zeros((T, nz, ny, nx, 2), np.float32)
    t1, t2, t3 = params["phase_steps"]
    assert t1 + t2 + t3 == T - 1, f"phase_steps {params['phase_steps']} must sum to T-1={T - 1}"
    shut = {f"PRODU{k}" for k in params.get("shut_in", [])}
    for t in range(T):
        tt = min(t, T - 2)                                   # last step has no following interval: repeat
        ph = 0 if tt < t1 else (1 if tt < t1 + t2 else 2)
        for name, (i, j, k1, k2) in wells.items():
            if name.startswith("INJE"):
                ctrl[t, k1 - 1:k2, j - 1, i - 1, 0] = params["inj_rate"] / INJ_NORM
            elif name not in shut:
                ctrl[t, k1 - 1:k2, j - 1, i - 1, 1] = params["prod_rates"][ph] / PROD_NORM
    return ctrl


def main(deck, outdir, out, scen_json=None, scen_idx=None):
    from resdata.resfile import ResdataFile
    from resdata.grid import Grid

    deck, outdir = Path(deck), Path(outdir)
    stem = deck.stem
    grid = Grid(str(outdir / f"{stem}.EGRID"))
    g_ = lambda a, b: getattr(grid, a, None) or getattr(grid, b)
    nx, ny, nz = g_("get_nx", "getNX")(), g_("get_ny", "getNY")(), g_("get_nz", "getNZ")()
    init = ResdataFile(str(outdir / f"{stem}.INIT"))

    def arr(kw, f):
        return np.asarray(f[kw][0].numpy_view(), dtype=np.float32).reshape(nz, ny, nx)

    permx, poro = arr("PERMX", init), arr("PORO", init)
    rst = ResdataFile(str(outdir / f"{stem}.UNRST"))
    n = len(rst["PRESSURE"])
    P = np.stack([np.asarray(rst["PRESSURE"][k].numpy_view()).reshape(nz, ny, nx) for k in range(n)])
    SW = np.stack([np.asarray(rst["SWAT"][k].numpy_view()).reshape(nz, ny, nx) for k in range(n)])
    SG = np.stack([np.asarray(rst["SGAS"][k].numpy_view()).reshape(nz, ny, nx) for k in range(n)])
    SO = 1.0 - SW - SG
    times = np.array([float(x[0]) for x in rst["DOUBHEAD"]], np.float32)

    wm = np.zeros((nz, ny, nx), np.float32)
    wt = np.zeros((nz, ny, nx), np.int8)
    for name, (i, j, k1, k2) in parse_compdat(deck).items():
        wm[k1 - 1:k2, j - 1, i - 1] = 1
        wt[k1 - 1:k2, j - 1, i - 1] = 1 if name.startswith("INJE") else -1

    extra = {}
    if scen_idx is not None:
        import json
        params = REF_SCENARIO if str(scen_idx) == "ref" else json.load(open(scen_json))[int(scen_idx)]
        extra["ctrl"] = build_ctrl(parse_compdat(deck), params, n, (nz, ny, nx))
        print("controls:", {k: params[k] for k in ("inj_rate", "prod_rates", "phase_steps", "shut_in")})
    np.savez_compressed(out, permx=permx, poro=poro, pressure=P.astype(np.float32), so=SO.astype(np.float32),
                        sg=SG.astype(np.float32), sw=SW.astype(np.float32), times=times,
                        well_mask=wm, well_type=wt, grid_shape=np.array([nz, ny, nx]), **extra)
    print("wrote", out, "| T =", n, "| grid(K,J,I) =", (nz, ny, nx), "| well cells =", int(wm.sum()),
          "| max|sum sat-1| =", float(np.abs(SO + SG + SW - 1).max()))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], a[2], *(a[3:5] if len(a) >= 5 else (None, None)))
