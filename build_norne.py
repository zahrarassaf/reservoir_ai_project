"""
Build a CNN-ready .npz from an OPM Flow run of the Norne benchmark (second field for cnn_dynamic_spe9.py).
Usage (Ubuntu, venv with resdata):
  python build_norne.py <flow_output_dir> <out.npz> [case_stem]        # default stem: NORNE_ATW2013
Reads <stem>.EGRID / .INIT / .UNRST. Restart arrays hold ACTIVE cells only (44,431 of 113,344), so they are expanded
to the full (K,J,I) grid; inactive cells get neutral finite filler and the ACTNUM mask is stored as `active`
(the code never samples inactive cells, zeroes them in the inputs and flags them with an extra channel).
Keys written: permx, poro (K,J,I); pressure, so, sg, sw (T,K,J,I); times (T,) days; active (K,J,I) bool; grid_shape.
No well locations or controls are written (same information content as the original SPE9 file), so the benchmark
is the strict `spatial` protocol. SO is computed as 1 - SW - SG (the restart file has no SOIL).
PERMX is floored at 1e-3 mD in active cells (log10 is taken downstream); the number of floored cells is printed.
"""
import sys
from pathlib import Path
import numpy as np

PERM_FLOOR = 1e-3


def expand(vec, actnum, shape, fill):
    """Active-cell vector (Fortran/active order) -> full (K,J,I) array. Length nx*ny*nz is accepted as already full."""
    nz, ny, nx = shape
    v = np.asarray(vec, dtype=np.float64).ravel()
    out = np.full(nx * ny * nz, fill, np.float64)
    if v.size == out.size:
        return v.reshape(nz, ny, nx)
    act = actnum.ravel().astype(bool)
    if v.size != int(act.sum()):
        raise ValueError(f"array length {v.size} matches neither all cells ({out.size}) nor active cells ({int(act.sum())})")
    out[act] = v
    return out.reshape(nz, ny, nx)


def assemble(actnum, permx_a, poro_a, P, SW, SG, times, shape):
    """Pure-numpy core (unit-testable). P/SW/SG: lists of active-cell vectors, one per report step."""
    act = actnum.reshape(shape).astype(bool)
    permx = expand(permx_a, actnum, shape, 1.0)
    poro = expand(poro_a, actnum, shape, 0.0)
    n_floor = int((permx[act] < PERM_FLOOR).sum())
    permx = np.where(act, np.maximum(permx, PERM_FLOOR), 1.0)
    pres = np.stack([expand(p, actnum, shape, 0.0) for p in P])
    sw = np.stack([expand(s, actnum, shape, 1.0) for s in SW])
    sg = np.stack([expand(s, actnum, shape, 0.0) for s in SG])
    so = np.where(act[None], 1.0 - sw - sg, 0.0)
    t = np.asarray(times, dtype=np.float64)
    if not np.all(np.diff(t) > 0):
        raise ValueError("report times are not strictly increasing")
    return dict(permx=permx.astype(np.float32), poro=poro.astype(np.float32), pressure=pres.astype(np.float32),
                so=so.astype(np.float32), sg=sg.astype(np.float32), sw=sw.astype(np.float32),
                times=t.astype(np.float32), active=act, grid_shape=np.array(shape)), n_floor


def main(outdir, out, stem="NORNE_ATW2013"):
    from resdata.resfile import ResdataFile
    from resdata.grid import Grid
    d = Path(outdir)
    grid = Grid(str(d / f"{stem}.EGRID"))
    g_ = lambda a, b: getattr(grid, a, None) or getattr(grid, b)
    nx, ny, nz = g_("get_nx", "getNX")(), g_("get_ny", "getNY")(), g_("get_nz", "getNZ")()
    shape = (nz, ny, nx)
    try:
        actnum = np.asarray(list(grid.export_actnum()), dtype=np.int32)
    except Exception:
        actnum = np.array([1 if grid.active(global_index=g) else 0 for g in range(nx * ny * nz)], np.int32)
    init = ResdataFile(str(d / f"{stem}.INIT"))
    rst = ResdataFile(str(d / f"{stem}.UNRST"))
    n = len(rst["PRESSURE"])
    vec = lambda f, k, i: np.asarray(f[k][i].numpy_view(), dtype=np.float64)
    P = [vec(rst, "PRESSURE", k) for k in range(n)]
    SW = [vec(rst, "SWAT", k) for k in range(n)]
    SG = [vec(rst, "SGAS", k) for k in range(n)]
    times = [float(rst["DOUBHEAD"][k][0]) for k in range(n)]
    arrays, n_floor = assemble(actnum, vec(init, "PERMX", 0), vec(init, "PORO", 0), P, SW, SG, times, shape)
    np.savez_compressed(out, **arrays)
    act = arrays["active"]
    sat = arrays["so"] + arrays["sg"] + arrays["sw"]
    print("wrote", out, "| T =", n, "| grid(K,J,I) =", shape, "| active =", int(act.sum()), "/", act.size,
          "| PERMX floored in", n_floor, "active cells | max|sum sat-1| =", float(np.abs(sat[:, act] - 1).max()))
    print("report times (days): first", times[:3], "last", times[-3:], "| min/max step", float(np.diff(times).min()), float(np.diff(times).max()))
    print("saturation ranges on active cells: SW [%.3f,%.3f] SG [%.3f,%.3f] SO [%.3f,%.3f]" % (
        arrays["sw"][:, act].min(), arrays["sw"][:, act].max(), arrays["sg"][:, act].min(), arrays["sg"][:, act].max(),
        arrays["so"][:, act].min(), arrays["so"][:, act].max()))


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0], a[1], *(a[2:3] if len(a) > 2 else ()))