"""Verification 01 on the ADE netlist of the final schematic (LVT, sa_sizes.il):
(a) delta-Vin from 300 mV to 1 nV (and -1 mV), three solver-tolerance sets: log law, t_d(1 mV), tau, energy, losing-output peak;
(b) operating range VDD x T x VCM at 1 mV and 10 uV: t_d(1 mV) and tau."""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sa_sim import LATCH, run, measure, SAVES
from cf import read_psf

DV = [0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 1e-7, 1e-8, 1e-9, -1e-3]
I1M = DV.index(1e-3)
TOLS = {"tight": "reltol=1e-6 vabstol=1e-12 iabstol=1e-15",
        "medium": "reltol=1e-5 vabstol=1e-9 iabstol=1e-12",
        "tight_iabs": "reltol=1e-6 vabstol=1e-12 iabstol=1e-13"}
dvs = " ".join(f"{v:g}" for v in DV)
out = {}


def files_by_index(name, pattern):
    return {tuple(int(i) for i in re.findall(r"_sw-(\d+)", os.path.basename(f))): f
            for f in glob.glob(os.path.join(LATCH, "raw", name, pattern))}


def tau_between(td, dv, a, b):
    return -(td[b] - td[a]) / np.log(abs(dv[b]) / abs(dv[a]))


# (a) delta-Vin sweep, three tolerance sets (one netlist each: Spectre keeps only the last options statement)
for k in TOLS:
    if "--analyze" not in sys.argv:
        try:
            run(f"v01_dvin_{k}", f"""simOpts options {TOLS[k]} temp=27 tnom=27 save=selected
{SAVES}
dv_sw sweep param=dvin values=[{dvs}] {{
  tr tran stop=2n maxstep=1p errpreset=conservative
}}""")
        except RuntimeError as e:
            print(e)
    fs = files_by_index(f"v01_dvin_{k}", "dv_sw-*_tr.tran.tran")
    m = [measure(read_psf(fs[(j,)])) if (j,) in fs else dict(td=np.nan, sign=0, energy=np.nan, vlose=np.nan)
         for j in range(len(DV))]
    td = np.array([x["td"] for x in m]) * 1e12
    out[k] = dict(dv=DV, td_ps=list(td), sign=[x["sign"] for x in m], E_fJ=[x["energy"] * 1e15 for x in m],
                  vlose=[x["vlose"] for x in m])
    print(f"--- {k}: {TOLS[k]}  ({len(fs)}/{len(DV)} transients)")
    print("  dvin    : " + " ".join(f"{v:8.0e}" for v in DV))
    print("  t_d (ps): " + " ".join(f"{v:8.2f}" for v in td))
    print("  sign    : " + " ".join(f"{x['sign']:8d}" for x in m))
    taus = [tau_between(td, DV, j, j + 1) for j in range(I1M, len(DV) - 2)]
    print(f"  tau per step 1 mV -> 1 nV (ps): " + " ".join(f"{x:.2f}" for x in taus))
    print(f"  E at 1 mV {m[I1M]['energy'] * 1e15:.2f} fJ, at 300 mV {m[0]['energy'] * 1e15:.2f} fJ;"
          f" max losing output {np.nanmax(out[k]['vlose']):.3f} V; t_d(-1 mV) {td[-1]:.2f} ps")
ref = np.array(out["tight"]["td_ps"])
for k in TOLS:
    if k != "tight":
        print(f"max |t_d({k}) - t_d(tight)| = {np.nanmax(np.abs(np.array(out[k]['td_ps']) - ref)):.3f} ps")

# (b) operating range
VDD, TEMP, VCM, DVR = [0.9, 1.0, 1.1], [0, 27, 85], [0.4, 0.5, 0.6], [1e-3, 1e-5]
if "--analyze" not in sys.argv and "--skip-pvt" not in sys.argv:
    try:
        run("v01_pvt", f"""simOpts options {TOLS['tight_iabs']} tnom=27 save=selected
save outp outn clk V0:p
t_sw sweep param=temp values=[{' '.join(map(str, TEMP))}] {{
vdd_sw sweep param=vdd values=[{' '.join(map(str, VDD))}] {{
vcm_sw sweep param=vcm values=[{' '.join(map(str, VCM))}] {{
dv_sw sweep param=dvin values=[{' '.join(f'{v:g}' for v in DVR)}] {{
  tr tran stop=2n maxstep=1p errpreset=conservative
}}
}}
}}
}}""")
    except RuntimeError as e:
        print(e)
fs = files_by_index("v01_pvt", "*dv_sw-*_tr.tran.tran")
pvt = []
print(f"--- operating range ({len(fs)}/{len(VDD) * len(TEMP) * len(VCM) * len(DVR)} transients)")
print(f"{'T':>3} {'VDD':>4} {'VCM':>4} | {'t_d 1mV':>8} {'tau':>6} {'E fJ':>6} {'M':>5}")
for it, T in enumerate(TEMP):
    for iv, vdd in enumerate(VDD):
        for ic, vcm in enumerate(VCM):
            m = [measure(read_psf(fs[(it, iv, ic, j)]), vdd) if (it, iv, ic, j) in fs else dict(td=np.nan, energy=np.nan)
                 for j in range(2)]
            td1, tau = m[0]["td"] * 1e12, (m[1]["td"] - m[0]["td"]) / np.log(DVR[0] / DVR[1]) * 1e12
            pvt.append(dict(T=T, vdd=vdd, vcm=vcm, td_1mV_ps=td1, tau_ps=tau, E_fJ=m[0]["energy"] * 1e15))
            print(f"{T:3d} {vdd:4.1f} {vcm:4.1f} | {td1:8.1f} {tau:6.2f} {m[0]['energy'] * 1e15:6.1f} {(300 - td1) / tau:5.1f}")
out["pvt"] = pvt
json.dump(out, open(os.path.join(LATCH, "data", "v01_nominal.json"), "w"), indent=1)
