"""StrongARM experiments: compose Spectre netlists from the ADE netlist of SAL45/STRONGARM_TB and
measure decision delay and supply energy from the transient waveforms.

The ADE netlist (netlists/sa_tb_ade.scs, from skill/02_netlist_tb.il) carries the circuit and the
design variables vdd, vcm, dvin, cl, tdel, trf, tpw, tper; each experiment appends its own options
and analyses.
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LATCH = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(LATCH, "..", "common"))
import numpy as np
from cf import read_psf, crossing

ADE = os.path.join(LATCH, "netlists", "sa_tb_ade.scs")
# the explorations (explore_*.py) started from the SVT first-cut build, frozen here
FIRSTCUT = os.path.join(LATCH, "netlists", "sa_tb_ade_firstcut_svt.scs")
CF = os.path.join(LATCH, "..", "common", "cf.sh")
SAVES = "save outp outn clk I_SA.X I_SA.Y I_SA.P I_SA.Q I_SA.TAIL V0:p"
TOL = "reltol=1e-6 vabstol=1e-12 iabstol=1e-13"  # verify_01: same t_d as iabstol=1e-15 within 0.002 ps, more robust


def circuit():
    """ADE netlist up to its own options and info statements."""
    txt = open(ADE).read()
    return txt[: txt.index("simulatorOptions")]


def write(name, analyses, base=None):
    """Netlist = base circuit text (default: circuit()) + the experiment's statements."""
    with open(os.path.join(LATCH, "netlists", f"{name}.scs"), "w") as f:
        f.write(circuit() if base is None else base)
        f.write(f"// ---- experiment {name}\n{analyses.strip()}\n")


def run(name, analyses, base=None):
    write(name, analyses, base)
    out = subprocess.run([CF, "spectre", "latch", name], capture_output=True, text=True)
    print(out.stdout.strip())
    if out.returncode:
        print(out.stderr)
        raise RuntimeError(f"spectre {name} failed")


def measure(d, vdd=1.0):
    """Decision delay (clk 50 % to first output crossing vdd/2), decision sign and supply energy."""
    t = d["time"]
    tclk = crossing(t, d["clk"], vdd / 2, True)
    if np.isnan(tclk):  # truncated (failed) transient
        return dict(td=np.nan, sign=0, energy=np.nan, vlose=np.nan)
    after = t > tclk
    tp = crossing(t[after], d["outp"][after], vdd / 2, True)
    tn = crossing(t[after], d["outn"][after], vdd / 2, True)
    if np.isnan(tp) and np.isnan(tn):
        td, sign = np.nan, 0
    elif np.isnan(tn) or (not np.isnan(tp) and tp < tn):
        td, sign = tp - tclk, +1
    else:
        td, sign = tn - tclk, -1
    i = -d["V0:p"]                     # current delivered by the supply
    energy = vdd * np.trapezoid(i, t)  # whole simulated cycle: reset -> evaluate -> reset
    # peak of the losing output while evaluating (clk high): a high value means a double/false decision
    tfall = crossing(t[after], d["clk"][after], vdd / 2, False)
    ev = after & (t < (tfall if not np.isnan(tfall) else t[-1]))
    lose = d["outn"] if sign >= 0 else d["outp"]
    return dict(td=td, sign=sign, energy=energy, vlose=lose[ev].max() if ev.any() else np.nan)


# device groups for approximate width sweeps (netlist text edits, CDF geometry stays at the build values)
GROUPS = dict(win=["M1", "M2"], wtail=["M0"], wlatn=["M3", "M4"], wlatp=["M5", "M6"], wrst=["S1", "S2", "S3", "S4"],
              winvn=["MN"], winvp=["MP"])  # MN/MP: inside the INV45 subcircuit


def sweep_sizes(name, grid, dv=(1e-3, 1e-5), lvt=True, extra="", simulate=True):
    """Nested Spectre sweep over device widths (grid: group -> list of widths) and dvin. Returns one row per
    combination with t_d(1st dv), tau from the two dv points and E; failed transients give NaN."""
    import glob
    import itertools
    import re
    txt = circuit()
    defaults = []
    for g, vals in grid.items():
        for dev in GROUPS[g]:
            txt, n = re.subn(rf"(\n    {dev} \([^)]*\) g45[np]1[sl]vt) w=\([^)]*\)", rf"\1 w=({g})", txt)
            assert n == 1, dev
        defaults.append(f"{g}={vals[0]}")
    txt = txt.replace("parameters ", "parameters " + " ".join(defaults) + " ", 1)
    if lvt:
        txt = re.sub(r"(g45[np])1svt", r"\g<1>1lvt", txt)
    opens = "\n".join(f"{g}_sw sweep param={g} values=[{' '.join(v)}] {{" for g, v in grid.items())
    body = f"""simOpts options {TOL} temp=27 tnom=27 save=selected
save outp outn clk V0:p
{extra}
{opens}
dv_sw sweep param=dvin values=[{' '.join(f'{v:g}' for v in dv)}] {{
  tr tran stop=2n maxstep=1p errpreset=conservative
}}
{chr(10).join('}' for _ in grid)}
"""
    with open(os.path.join(LATCH, "netlists", f"{name}.scs"), "w") as f:
        f.write(txt + f"// ---- experiment {name}\n" + body)
    if simulate:
        out = subprocess.run([CF, "spectre", "latch", name], capture_output=True, text=True)
        print(out.stdout.strip().splitlines()[0])
    files = {tuple(int(i) for i in re.findall(r"_sw-(\d+)", os.path.basename(f))): f
             for f in glob.glob(os.path.join(LATCH, "raw", name, "*dv_sw-*_tr.tran.tran"))}
    rows = []
    for combo, idx in zip(itertools.product(*grid.values()), itertools.product(*[range(len(v)) for v in grid.values()])):
        m = [measure(read_psf(files[idx + (j,)])) if idx + (j,) in files
             else dict(td=np.nan, energy=np.nan, vlose=np.nan, sign=0) for j in range(len(dv))]
        tau = (m[-1]["td"] - m[0]["td"]) / np.log(dv[0] / dv[-1])
        rows.append(dict(zip(grid, combo), td_ps=m[0]["td"] * 1e12, tau_ps=tau * 1e12, E_fJ=m[0]["energy"] * 1e15,
                         dv=list(dv), td_all_ps=[x["td"] * 1e12 for x in m], vlose=[x["vlose"] for x in m],
                         sign=[x["sign"] for x in m]))
    return rows


def show(rows, keys, sort="td_ps", n=12):
    print(" ".join(f"{k:>6s}" for k in keys) + f" | {'t_d':>7s} {'tau':>6s} {'E fJ':>6s}")
    for r in sorted([r for r in rows if not np.isnan(r["tau_ps"])], key=lambda r: r[sort])[:n]:
        print(" ".join(f"{r[k]:>6s}" for k in keys) + f" | {r['td_ps']:7.1f} {r['tau_ps']:6.1f} {r['E_fJ']:6.1f}")
