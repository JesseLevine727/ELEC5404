"""Dataset generation for the MLP / KBNN models (PROJECT_PLAN.md section 4).

One sample = one Spectre transient of one comparison from reset, final latch (netlists/sa_tb_ade.scs):
ideal inputs vcm +- dvin/2, clock 1.2 ns evaluate / 0.8 ns reset, verification tolerances (sa_sim.TOL, 1 ps max
step). veriloga/sa_meter.va measures t_d and E inside the simulation and appends one line per sample, so no
waveforms are saved. Each set goes to Spectre as one paramset sweep.

Sets (data/dataset/points_<set>.csv in, <set>.csv out):
  train   2800  dvin 100 uV - 300 mV  scrambled Sobol, in order (every prefix is evenly spread)
  val      600  same range             uniform random
  test     600  same range             uniform random
  extrap   500  dvin 1 - 100 uV        uniform random (never trained on)
  slices   306  dvin 1 uV - 300 mV, 6 per decade, at 9 fixed (vcm, vdd, temp) conditions
  sym       50  -dvin at 25 test + 25 extrap points (symmetry check)
  tol       50  25 test + the 25 smallest-dvin extrap points at half the tolerances and max step
dvin log-uniform; vcm 0.4-0.6 V, vdd 0.9-1.1 V, temp 0-85 C uniform.

Usage: points | pilot [--analyze] | run <set>... | collect"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sa_sim import LATCH, circuit, run, measure, TOL
from cf import read_psf

DSET = os.path.join(LATCH, "data", "dataset")
REMOTE_VA = "/home/jesselevine/Desktop/StrongArm45/latch/veriloga/sa_meter.va"
TOL_HALF = "reltol=5e-7 vabstol=5e-13 iabstol=5e-14"
TRAN = "tran stop=2n maxstep=1p errpreset=conservative outputstart=1.99n"  # meter does the measuring; write almost nothing
TRAN_HALF = "tran stop=2n maxstep=0.5p errpreset=conservative outputstart=1.99n"
RANGES = dict(vcm=(0.4, 0.6), vdd=(0.9, 1.1), temp=(0.0, 85.0))
DV_MAIN, DV_EXTRAP = (100e-6, 0.3), (1e-6, 100e-6)
# nominal; each of vcm / vdd / temp at both ends; worst and best corners
SLICES = [(0.5, 1.0, 27), (0.4, 1.0, 27), (0.6, 1.0, 27), (0.5, 0.9, 27), (0.5, 1.1, 27),
          (0.5, 1.0, 0), (0.5, 1.0, 85), (0.4, 0.9, 85), (0.6, 1.1, 0)]
SETS = ["train", "val", "test", "extrap", "slices", "sym", "tol"]
NOMINAL = dict(td_ps=105.76, E_fJ=32.2)  # verify_01, final design: dvin 1 mV, vcm 0.5 V, 1.0 V, 27 C
HEADER = "dvin_V,vcm_V,vdd_V,temp_C,td_ps,E_fJ,sign,vlose_V,ok"


def scale(u, dv):
    """Unit-cube samples (n x 4) -> dvin (log-uniform over dv), vcm, vdd, temp."""
    lo, hi = np.log10(dv)
    out = np.empty_like(u)
    out[:, 0] = 10 ** (lo + u[:, 0] * (hi - lo))
    for j, k in enumerate(("vcm", "vdd", "temp"), 1):
        a, b = RANGES[k]
        out[:, j] = a + u[:, j] * (b - a)
    return out


def make_points():
    from scipy.stats import qmc
    pts = {"train": scale(qmc.Sobol(4, scramble=True, seed=5404).random_base2(12)[:2800], DV_MAIN),
           "val": scale(np.random.default_rng(1).random((600, 4)), DV_MAIN),
           "test": scale(np.random.default_rng(2).random((600, 4)), DV_MAIN),
           "extrap": scale(np.random.default_rng(3).random((500, 4)), DV_EXTRAP)}
    dvs = np.append(1e-6 * 10 ** (np.arange(33) / 6), 0.3)
    pts["slices"] = np.array([(dv, *c) for c in SLICES for dv in dvs])
    rng = np.random.default_rng(4)
    idx = {"sym": [("test", int(i)) for i in rng.choice(600, 25, replace=False)]
                  + [("extrap", int(i)) for i in rng.choice(500, 25, replace=False)],
           "tol": [("test", int(i)) for i in rng.choice(600, 25, replace=False)]
                  + [("extrap", int(i)) for i in np.argsort(pts["extrap"][:, 0])[:25]]}
    pts["sym"] = np.array([pts[s][i] for s, i in idx["sym"]]) * [-1, 1, 1, 1]
    pts["tol"] = np.array([pts[s][i] for s, i in idx["tol"]])
    os.makedirs(DSET, exist_ok=True)
    for name, p in pts.items():
        np.savetxt(os.path.join(DSET, f"points_{name}.csv"), p, delimiter=",", fmt="%.10g",
                   header="dvin_V,vcm_V,vdd_V,temp_C", comments="")
    json.dump(idx, open(os.path.join(DSET, "check_sources.json"), "w"))
    for name, p in pts.items():
        print(f"{name:7s} {len(p):5d} points, |dvin| {abs(p[:, 0]).min():.3g} - {abs(p[:, 0]).max():.3g} V")


def points(name):
    return np.loadtxt(os.path.join(DSET, f"points_{name}.csv"), delimiter=",", skiprows=1, ndmin=2)


def base():
    """Final latch with the meter's zero-volt branch between the supply source and the vdd net."""
    txt = circuit()
    assert txt.count("V0 (vdd 0) vsource") == 1
    return txt.replace("V0 (vdd 0) vsource", "V0 (vsup 0) vsource") + f'parameters pid=0\nahdl_include "{REMOTE_VA}"\n'


def simulate(name, pts, half_tol=False, saves=None):
    rows = "\n".join(f"{i} " + " ".join(f"{v:.10g}" for v in p) for i, p in enumerate(pts))
    opts = f"simOpts options {TOL_HALF if half_tol else TOL} temp=27 tnom=27 save={'selected' if saves else 'none'}"
    body = "\n".join(
        [f'MTR (clk outp outn vsup vdd) sa_meter vdd=vdd pid=pid dvin=dvin vcm=vcm file="raw/{name}/meter.txt"', opts]
        + ([saves] if saves else [])
        + [f"pts paramset {{\npid dvin vcm vdd temp\n{rows}\n}}",
           "sw sweep paramset=pts {", f"  tr {TRAN_HALF if half_tol else TRAN}", "}"])
    try:
        run(name, body, base=base())
    except RuntimeError as e:
        print(e)


def read_meter(name):
    """pid -> meter line (11 numbers after the pid)."""
    rows = {}
    for ln in open(os.path.join(LATCH, "raw", name, "meter.txt")):
        v = [float(x) for x in ln.split()]
        rows[int(v[0])] = v[1:]
    return rows


def metrics(pt, m):
    """One dataset row from the input point and its meter line."""
    dvin, vcm, vdd, temp = pt
    edv, evcm, evdd, etemp, tclk, tp, tn, tfall, energy, pkp, pkn = m
    # the echoed inputs prove the paramset row reached the circuit (temp included)
    echo = (abs(edv - dvin) <= 1e-8 * abs(dvin) and abs(evcm - vcm) < 1e-9 and abs(evdd - vdd) < 1e-9
            and abs(etemp - temp) < 1e-6)
    inwin = [(t, s) for t, s in ((tp, +1), (tn, -1)) if t > 0 and t < tfall]
    if not inwin or tclk < 0:
        return [np.nan, energy * 1e15, 0, np.nan, 0]
    t, sign = min(inwin)
    other = tn if sign > 0 else tp
    double = 0 < other < tfall  # the losing output also crossed vdd/2
    ok = echo and sign == np.sign(dvin) and not double
    return [(t - tclk) * 1e12, energy * 1e15, sign, pkn if sign > 0 else pkp, int(ok)]


def collect(name, run_name=None):
    pts, m = points(name), read_meter(run_name or f"ds_{name}")
    out = np.array([[*p, *(metrics(p, m[i]) if i in m else [np.nan, np.nan, 0, np.nan, 0])]
                    for i, p in enumerate(pts)])
    np.savetxt(os.path.join(DSET, f"{name}.csv"), out, delimiter=",", header=HEADER, comments="",
               fmt=["%.10g"] * 4 + ["%.6f", "%.5f", "%d", "%.5f", "%d"])
    bad = np.flatnonzero(out[:, 8] == 0)
    print(f"{name:7s} {len(out)} points, {len(m)} measured, {len(out) - len(bad)} ok"
          + (f"; NOT ok: rows {bad.tolist()[:20]}" if len(bad) else "")
          + f"; t_d {np.nanmin(out[:, 4]):.1f}-{np.nanmax(out[:, 4]):.1f} ps, E {np.nanmin(out[:, 5]):.1f}-{np.nanmax(out[:, 5]):.1f} fJ,"
          f" losing output <= {np.nanmax(out[:, 7]):.3f} V")
    return out


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "points":
        make_points()
    elif cmd == "pilot":
        # meter vs the waveform measurement used in verification, on a spread of points incl. the extremes
        P = np.array([(1e-3, 0.5, 1.0, 27), (0.3, 0.5, 1.0, 27), (-1e-3, 0.5, 1.0, 27), (1e-5, 0.5, 1.0, 27),
                      (1e-6, 0.4, 0.9, 85), (1e-6, 0.6, 1.1, 0), (0.3, 0.4, 0.9, 85), (0.3, 0.6, 1.1, 0),
                      (1e-4, 0.45, 0.95, 60), (1e-2, 0.55, 1.05, 10), (3e-6, 0.5, 1.0, 85), (3e-2, 0.4, 1.1, 0)])
        if "--analyze" not in sys.argv:
            simulate("ds_pilot", P, saves="save clk outp outn V0:p")
        m = read_meter("ds_pilot")
        import glob, re
        files = {int(re.search(r"sw-(\d+)", os.path.basename(f)).group(1)): f
                 for f in glob.glob(os.path.join(LATCH, "raw", "ds_pilot", "sw-*_tr.tran.tran"))}
        print(f"{'dvin':>8} {'vcm':>5} {'vdd':>5} {'T':>4} | {'td meter':>9} {'td wave':>9} {'diff fs':>8} |"
              f" {'E meter':>8} {'E wave':>8} | {'vlose m':>7} {'vlose w':>7} ok")
        for i, p in enumerate(P):
            r = metrics(p, m[i])
            w = measure(read_psf(files[i]), vdd=p[2])
            print(f"{p[0]:8.0e} {p[1]:5.2f} {p[2]:5.2f} {p[3]:4.0f} | {r[0]:9.3f} {w['td'] * 1e12:9.3f}"
                  f" {(r[0] - w['td'] * 1e12) * 1e3:8.1f} | {r[1]:8.3f} {w['energy'] * 1e15:8.3f} |"
                  f" {r[3]:7.3f} {w['vlose']:7.3f} {r[4]}")
    elif cmd == "run":
        for name in sys.argv[2:]:
            simulate(f"ds_{name}", points(name), half_tol=(name == "tol"))
    elif cmd == "collect":
        res = {n: collect(n) for n in SETS if os.path.exists(os.path.join(LATCH, "raw", f"ds_{n}", "meter.txt"))}
        idx = json.load(open(os.path.join(DSET, "check_sources.json")))
        for chk in ("sym", "tol"):
            if chk in res and all(s in res for s, _ in idx[chk]):
                d_td = np.array([res[chk][k, 4] - res[s][i, 4] for k, (s, i) in enumerate(idx[chk])])
                d_e = np.array([res[chk][k, 5] - res[s][i, 5] for k, (s, i) in enumerate(idx[chk])])
                print(f"{chk}: |t_d difference| max {np.nanmax(abs(d_td)) * 1e3:.1f} fs (median {np.nanmedian(abs(d_td)) * 1e3:.1f} fs),"
                      f" |E difference| max {np.nanmax(abs(d_e)) * 1e3:.2f} aJ")
        if "slices" in res:
            r = res["slices"][18]
            print(f"nominal (slices row 18, dvin {r[0]:g}): t_d {r[4]:.2f} ps (verified {NOMINAL['td_ps']}),"
                  f" E {r[5]:.2f} fJ (verified {NOMINAL['E_fJ']})")
