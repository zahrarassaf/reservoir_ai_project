"""Print what an OPM Flow run wrote, so a dataset builder can be written against the REAL output.
Usage: python inspect_flow_output.py <output_dir> <case_stem>"""
import sys
from pathlib import Path


def main(d, stem):
    d = Path(d)
    print("files:", sorted(p.name for p in d.iterdir()))
    from resdata.resfile import ResdataFile
    from resdata.grid import Grid
    g = Grid(str(d / f"{stem}.EGRID"))
    for nm in ("get_nx", "get_ny", "get_nz", "get_num_active", "getNX", "getNY", "getNZ", "getNumActive"):
        if hasattr(g, nm):
            try:
                print(f"grid.{nm}() =", getattr(g, nm)())
            except Exception as e:
                print(f"grid.{nm} failed:", e)
    init = ResdataFile(str(d / f"{stem}.INIT"))
    print("INIT keywords:", [k for k in init.keys()][:60])
    rst = ResdataFile(str(d / f"{stem}.UNRST"))
    ks = list(rst.keys())
    print("UNRST distinct keywords:", sorted(set(ks)))
    for k in ("PRESSURE", "SWAT", "SGAS", "SOIL", "RS", "RV", "INTEHEAD", "DOUBHEAD", "ICON", "IWEL", "ZWEL", "LOGIHEAD"):
        try:
            print(f"count[{k}] =", len(rst[k]))
        except Exception as e:
            print(f"count[{k}] -> not available ({type(e).__name__})")
    try:
        ih = rst["INTEHEAD"][0]
        vals = [int(x) for x in ih.numpy_view()]
        print("INTEHEAD[:50] =", vals[:50])
    except Exception as e:
        print("INTEHEAD failed:", e)
    try:
        print("first/last report day:", float(rst["DOUBHEAD"][0][0]), float(rst["DOUBHEAD"][len(rst['DOUBHEAD']) - 1][0]))
    except Exception as e:
        print("DOUBHEAD failed:", e)
    try:
        print("PRESSURE[0] size:", len(rst["PRESSURE"][0].numpy_view()), "(active cells per report step)")
    except Exception as e:
        print("PRESSURE size failed:", e)
    try:
        from resdata.summary import Summary
        s = Summary(str(d / f"{stem}.UNSMRY"))
        keys = list(s.keys())
        print("summary keys:", len(keys), "| first 40:", keys[:40])
        wk = [k for k in keys if k.startswith(("WOPR", "WWIR", "WGIR", "WWPR"))]
        print("sample well rate keys:", wk[:20])
        print("report steps in summary:", len(s.report_index_list()) if hasattr(s, "report_index_list") else "n/a")
    except Exception as e:
        print("summary read failed:", type(e).__name__, e)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
