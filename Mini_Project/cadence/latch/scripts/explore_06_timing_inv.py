"""Exploration, all-LVT, core sizes from 05 (win 12u, wtail 8u, wlatn/wlatp 1u, wrst 250n):
(a) where the 1 mV delay goes (internal node timing), (b) output inverter sizing / trip point."""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_sim
sa_sim.ADE = sa_sim.FIRSTCUT
import numpy as np
from sa_sim import LATCH, circuit, run, sweep_sizes, show, measure, TOL, GROUPS
from cf import read_psf, crossing

CORE = dict(win="12u", wtail="8u", wlatn="1u", wlatp="1u", wrst="250n")

# (a) timing breakdown at 1 mV
txt = circuit()
for g, w in CORE.items():
    for dev in GROUPS[g]:
        txt, n = re.subn(rf"(\n    {dev} \([^)]*\) g45[np]1svt) w=\([^)]*\)", rf"\1 w=({w})", txt)
        assert n == 1
txt = re.sub(r"(g45[np])1svt", r"\g<1>1lvt", txt)
open(os.path.join(LATCH, "netlists", "06_timing.scs"), "w").write(txt + f"""// ---- experiment 06_timing
simOpts options {TOL} temp=27 tnom=27 save=selected
save outp outn clk I_SA.X I_SA.Y I_SA.P I_SA.Q I_SA.TAIL V0:p
tr tran stop=2n maxstep=0.5p errpreset=conservative
""")
import subprocess
print(subprocess.run([os.path.join(LATCH, "..", "common", "cf.sh"), "spectre", "latch", "06_timing"],
                     capture_output=True, text=True).stdout.strip().splitlines()[0])
d = read_psf(os.path.join(LATCH, "raw", "06_timing", "tr.tran.tran"))
t = d["time"]; tclk = crossing(t, d["clk"], 0.5, True); a = t > tclk
def tx(sig, lvl, rising=False):
    return (crossing(t[a], sig[a], lvl, rising) - tclk) * 1e12
pq = (d["I_SA.P"] + d["I_SA.Q"]) / 2; xy = (d["I_SA.X"] + d["I_SA.Y"]) / 2
dxy = d["I_SA.Y"] - d["I_SA.X"]
print(f"clk 50% -> tail below 0.1 V:     {tx(d['I_SA.TAIL'], 0.1):6.1f} ps   (tail min {d['I_SA.TAIL'][a].min():.3f} V)")
print(f"P/Q common mode reaches 0.5 V:   {tx(pq, 0.5):6.1f} ps")
print(f"X/Y common mode falls 0.3 V:     {tx(xy, 0.7):6.1f} ps")
print(f"|Y - X| reaches 10 mV / 0.1 V:   {tx(dxy, 0.01, True):6.1f} / {tx(dxy, 0.1, True):6.1f} ps")
print(f"X crosses 0.5 V:                 {tx(d['I_SA.X'], 0.5):6.1f} ps")
print(f"OUTP crosses 0.5 V (t_d):        {tx(d['outp'], 0.5, True):6.1f} ps")
np.savez(os.path.join(LATCH, "data", "06_timing.npz"), **{k.replace(".", "_").replace(":", "_"): v for k, v in d.items()})

# (b) output inverter sweep at the core sizes
grid = {k: [v] for k, v in CORE.items()} | dict(winvn=["250n", "500n", "1u"], winvp=["700n", "1.4u", "2.8u"])
rows = sweep_sizes("06_inv", grid)
json.dump(rows, open(os.path.join(LATCH, "data", "06_inv.json"), "w"), indent=1)
show(rows, ["winvn", "winvp"], n=9)
