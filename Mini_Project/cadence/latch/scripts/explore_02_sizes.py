"""Exploration: StrongARM speed vs device widths, SVT (02) or all-LVT (04, --lvt). Approximate: only w and
the model flavour change; junction/stress geometry stays at the first-cut CDF values.
Delay at 1 mV and 10 uV -> t_d(1 mV) and tau."""
import glob
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_sim
sa_sim.ADE = sa_sim.FIRSTCUT
import numpy as np
from sa_sim import LATCH, circuit, measure, TOL
from cf import read_psf
import subprocess

LVT = "--lvt" in sys.argv
GRID = (dict(win=["2u", "4u", "8u"], wtail=["2u", "4u", "8u"], wlatn=["500n", "1u", "2u"], wlatp=["500n", "1u", "2u"])
        if LVT else
        dict(win=["4u", "8u", "16u"], wtail=["4u", "8u", "16u"], wlatn=["1u", "2u", "4u"], wlatp=["1u", "2u", "4u"]))
DV = [1e-3, 1e-5]
NAME = "04_explore_sizes_lvt" if LVT else "02_explore_sizes"

if "--analyze" not in sys.argv:
    txt = circuit()
    for dev, par in (("M1", "win"), ("M2", "win"), ("M0", "wtail"), ("M3", "wlatn"), ("M4", "wlatn"),
                     ("M5", "wlatp"), ("M6", "wlatp")):
        txt, n = re.subn(rf"(\n    {dev} \([^)]*\) g45[np]1svt) w=\([^)]*\)", rf"\1 w=({par})", txt)
        assert n == 1, dev
    txt = txt.replace("parameters ", "parameters win=4u wtail=4u wlatn=1u wlatp=1u ", 1)
    if LVT:
        txt = re.sub(r"(g45[np])1svt", r"\g<1>1lvt", txt)
    sweeps = "\n".join(f"{k}_sw sweep param={k} values=[{' '.join(v)}] {{" for k, v in GRID.items())
    body = f"""simOpts options {TOL} temp=27 tnom=27 save=selected
save outp outn clk V0:p
{sweeps}
dv_sw sweep param=dvin values=[{' '.join(f'{v:g}' for v in DV)}] {{
  tr tran stop=2n maxstep=1p errpreset=conservative
}}
{chr(10).join('}' for _ in GRID)}
"""
    with open(os.path.join(LATCH, "netlists", f"{NAME}.scs"), "w") as f:
        f.write(txt + f"// ---- experiment {NAME}\n" + body)
    out = subprocess.run([os.path.join(LATCH, "..", "common", "cf.sh"), "spectre", "latch", NAME],
                         capture_output=True, text=True)
    print(out.stdout.strip())

# map files by their sweep indices; a failed transient leaves a gap instead of shifting everything
files = {tuple(int(i) for i in re.findall(r"_sw-(\d+)", os.path.basename(f))): f
         for f in glob.glob(os.path.join(LATCH, "raw", NAME, "*dv_sw-*_tr.tran.tran"))}
combos = list(itertools.product(*GRID.values()))
idx = list(itertools.product(*[range(len(v)) for v in GRID.values()]))
rows, failed = [], []
nan = dict(td=np.nan, energy=np.nan)
for k, combo in enumerate(combos):
    m = [measure(read_psf(files[idx[k] + (j,)])) if idx[k] + (j,) in files else nan for j in range(2)]
    if any(np.isnan(x["td"]) for x in m):
        failed.append(combo)
    tau = (m[1]["td"] - m[0]["td"]) / np.log(DV[0] / DV[1])
    rows.append(dict(zip(GRID, combo), td_1mV_ps=m[0]["td"] * 1e12, tau_ps=tau * 1e12,
                     E_fJ=m[0]["energy"] * 1e15))
json.dump(rows, open(os.path.join(LATCH, "data", f"{NAME}.json"), "w"), indent=1)
rows = [r for r in rows if not np.isnan(r["tau_ps"])]
rows.sort(key=lambda r: r["td_1mV_ps"])
print(f"{len(files)} transients; failed or undecided combinations: {failed}")
print("Fastest 12 by t_d(1 mV):")
print(f"{'win':>5s} {'wtail':>5s} {'wlatn':>5s} {'wlatp':>5s} | {'t_d 1mV':>8s} {'tau':>6s} {'E fJ':>6s}")
for r in rows[:12]:
    print(f"{r['win']:>5s} {r['wtail']:>5s} {r['wlatn']:>5s} {r['wlatp']:>5s} | {r['td_1mV_ps']:8.1f} {r['tau_ps']:6.1f} {r['E_fJ']:6.1f}")
print("Lowest tau 5:")
for r in sorted(rows, key=lambda r: r["tau_ps"])[:5]:
    print(f"{r['win']:>5s} {r['wtail']:>5s} {r['wlatn']:>5s} {r['wlatp']:>5s} | {r['td_1mV_ps']:8.1f} {r['tau_ps']:6.1f} {r['E_fJ']:6.1f}")
