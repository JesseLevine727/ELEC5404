"""Exploration: is the StrongARM starved by VCM = 0.5 V? Same first-cut sizes, (a) SVT with VCM 0.5 / 0.6 / 0.7 V,
(b) LVT input pair, LVT latch, all LVT at VCM = 0.5 V. Delay at 1 mV and 10 uV -> t_d(1 mV), tau."""
import glob
import json
import os
import re
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_sim
sa_sim.ADE = sa_sim.FIRSTCUT
import numpy as np
from sa_sim import LATCH, circuit, measure, TOL
from cf import read_psf

DV = [1e-3, 1e-5]
CASES = {"svt": [], "lvt_in": ["M1", "M2"], "lvt_latch": ["M3", "M4", "M5", "M6"],
         "lvt_all": ["M1", "M2", "M0", "M3", "M4", "M5", "M6", "S1", "S2", "S3", "S4"]}
results = {}
for case, devs in CASES.items():
    name = f"03_{case}"
    txt = circuit()
    for d in devs:
        txt, n = re.subn(rf"(\n    {d} \([^)]*\) g45)([np])1svt", r"\g<1>\g<2>1lvt", txt)
        assert n == 1, (case, d)
    vcms = [0.5, 0.6, 0.7] if case == "svt" else [0.5]
    body = f"""simOpts options {TOL} temp=27 tnom=27 save=selected
save outp outn clk V0:p
vcm_sw sweep param=vcm values=[{' '.join(map(str, vcms))}] {{
dv_sw sweep param=dvin values=[{' '.join(f'{v:g}' for v in DV)}] {{
  tr tran stop=2n maxstep=1p errpreset=conservative
}}
}}
"""
    open(os.path.join(LATCH, "netlists", f"{name}.scs"), "w").write(txt + f"// ---- experiment {name}\n" + body)
    out = subprocess.run([os.path.join(LATCH, "..", "common", "cf.sh"), "spectre", "latch", name],
                         capture_output=True, text=True)
    print(out.stdout.strip().splitlines()[0])
    files = sorted(glob.glob(os.path.join(LATCH, "raw", name, "*dv_sw-*_tr.tran.tran")))
    for k, vcm in enumerate(vcms):
        m = [measure(read_psf(files[2 * k + j])) for j in range(2)]
        tau = (m[1]["td"] - m[0]["td"]) / np.log(DV[0] / DV[1])
        results[f"{case} vcm={vcm}"] = dict(td_1mV_ps=m[0]["td"] * 1e12, tau_ps=tau * 1e12, E_fJ=m[0]["energy"] * 1e15)
json.dump(results, open(os.path.join(LATCH, "data", "03_vcm_lvt.json"), "w"), indent=1)
print(f"{'case':22s} {'t_d 1mV':>8s} {'tau':>6s} {'E fJ':>6s}")
for k, r in results.items():
    print(f"{k:22s} {r['td_1mV_ps']:8.1f} {r['tau_ps']:6.1f} {r['E_fJ']:6.1f}")
