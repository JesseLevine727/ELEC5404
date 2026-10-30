"""Exploration (approximate geometry for changed widths): decision memory vs reset-switch width and reset time.
Alternating input +100 mV / -delta (verify_05 testbench); the memory threshold is the smallest delta whose -delta
cycles all decide correctly. Also t_d(1 mV) and tau with ideal sources (relaxed clock) per width."""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sa_sim import LATCH, run, measure, TOL, GROUPS
from cf import read_psf, crossing
import verify_05_memory as v5

WRST = ["250n", "500n", "1u"]
TPER = ["0.6n", "0.7n", "0.8n"]           # tpw = 0.4n -> flat reset 160 / 260 / 360 ps
DELTA = ["10u", "100u", "1m", "3m", "10m"]
NCYC = 8


def with_wrst(txt):
    for dev in GROUPS["wrst"]:
        txt, n = re.subn(rf"(\n    {dev} \([^)]*\) g45[np]1[sl]vt) w=\([^)]*\)", rf"\1 w=(wrst)", txt)
        assert n == 1, dev
    return txt.replace("parameters ", "parameters wrst=250n ", 1)


if "--analyze" not in sys.argv:
    b = [f"simOpts options {TOL} temp=27 tnom=27 save=selected", "save dvn clk outp outn",
         "a_pw alter param=tpw value=0.4n",
         f"w_sw sweep param=wrst values=[{' '.join(WRST)}] {{",
         f"p_sw sweep param=tper values=[{' '.join(TPER)}] {{",
         f"d_sw sweep param=delta values=[{' '.join(DELTA)}] {{",
         f"  tr tran stop={0.2e-9 + NCYC * 0.8e-9:g} maxstep=1p errpreset=conservative",
         "}", "}", "}"]
    run("09_reset_memory", "\n".join(b), base=with_wrst(v5.base()))
    # speed with ideal sources for each width (default relaxed clock of the ADE netlist)
    import sa_sim
    rows = sa_sim.sweep_sizes("09_reset_speed", dict(wrst=WRST), dv=(1e-3, 1e-5))
    json.dump(rows, open(os.path.join(LATCH, "data", "09_reset_speed.json"), "w"), indent=1)

speed = {r["wrst"]: r for r in json.load(open(os.path.join(LATCH, "data", "09_reset_speed.json")))}
files = {tuple(int(i) for i in re.findall(r"_sw-(\d+)", os.path.basename(f))): f
         for f in glob.glob(os.path.join(LATCH, "raw", "09_reset_memory", "*d_sw-*_tr.tran.tran"))}
out = []
print(f"{'wrst':>5} | t_d(1mV) tau  E   | memory threshold (smallest passing delta) at flat reset 160 / 260 / 360 ps")
for i, w in enumerate(WRST):
    thr = []
    for j, per in enumerate(TPER):
        tper = float(per.rstrip("n")) * 1e-9
        ok_list = []
        for k, de in enumerate(DELTA):
            d = read_psf(files[(i, j, k)]); t = d["time"]
            good = True
            for c in range(1, NCYC, 2):
                tr = 0.2e-9 + 10e-12 + c * tper
                wdw = (t > tr) & (t < tr + 0.35e-9)
                tp, tn = crossing(t[wdw], d["outp"][wdw], 0.5, True), crossing(t[wdw], d["outn"][wdw], 0.5, True)
                good &= (not np.isnan(tn)) and (np.isnan(tp) or tn < tp)
            ok_list.append(good)
        passing = [DELTA[k] for k, g in enumerate(ok_list) if g]
        thr.append(passing[0] if passing and all(ok_list[DELTA.index(passing[0]):]) else ">" + DELTA[-1])
    s = speed[w]
    out.append(dict(wrst=w, td_ps=s["td_ps"], tau_ps=s["tau_ps"], E_fJ=s["E_fJ"], memory=dict(zip(["160ps", "260ps", "360ps"], thr))))
    print(f"{w:>5} | {s['td_ps']:6.1f} {s['tau_ps']:5.2f} {s['E_fJ']:5.1f} | " + "   ".join(f"{x:>6}" for x in thr))
json.dump(out, open(os.path.join(LATCH, "data", "09_reset.json"), "w"), indent=1)
