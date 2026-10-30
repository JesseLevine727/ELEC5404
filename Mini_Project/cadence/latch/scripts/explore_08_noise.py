"""Exploration on the final ADE netlist (approximate geometry for changed widths): input-referred noise vs
input-pair / tail width, plus a noise-bandwidth check. Per candidate: deterministic t_d(1 mV), tau (1 mV / 10 uV), E,
and a short transient-noise S-curve (dv = 0.3/0.6/0.9 mV, NCYC cycles each, sigma fitted with mu = 0)."""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import ndtr
from sa_sim import LATCH, circuit, run, measure, GROUPS
from cf import read_psf

NAME = "08_noise_sizes"
NCYC, TPER = 300, 2e-9
DVN = [0.3e-3, 0.6e-3, 0.9e-3]
CANDS = [("base", "12u", "8u", "100G"), ("base_200G", "12u", "8u", "200G"), ("win16", "16u", "8u", "100G"),
         ("win24", "24u", "8u", "100G"), ("tail4", "12u", "4u", "100G"), ("win24_tail4", "24u", "4u", "100G")]

if "--analyze" not in sys.argv:
    txt = circuit()
    for g in ("win", "wtail"):
        for dev in GROUPS[g]:
            txt, n = re.subn(rf"(\n    {dev} \([^)]*\) g45[np]1[sl]vt) w=\([^)]*\)", rf"\1 w=({g})", txt)
            assert n == 1, dev
    txt = txt.replace("parameters ", "parameters win=12u wtail=8u ", 1)
    stmts = ["simOpts options reltol=1e-5 vabstol=1e-9 iabstol=1e-12 temp=27 tnom=27 save=selected",
             "save outp outn clk V0:p"]
    for k, (name, win, wtail, fmax) in enumerate(CANDS):
        stmts += [f"a{k}w alter param=win value={win}", f"a{k}t alter param=wtail value={wtail}",
                  f"det{k} sweep param=dvin values=[1e-3 1e-5] {{",
                  f"  trd{k} tran stop=2n maxstep=1p errpreset=conservative", "}",
                  f"noi{k} sweep param=dvin values=[{' '.join(f'{v:g}' for v in DVN)}] {{",
                  f"  trn{k} tran stop={200e-12 + NCYC * TPER:g} maxstep=2p errpreset=moderate noisefmax={fmax} noiseseed={11 + k}",
                  "}"]
    try:
        run(NAME, "\n".join(stmts), base=txt)
    except RuntimeError as e:
        print(e)

raw = os.path.join(LATCH, "raw", NAME)
res = []
print(f"{'cand':12s} {'win':>4s} {'tail':>4s} {'fmax':>5s} | {'t_d 1mV':>7s} {'tau':>5s} {'E fJ':>5s} | {'P(+) at 0.3/0.6/0.9 mV':>24s} | sigma_n")
for k, (name, win, wtail, fmax) in enumerate(CANDS):
    m = [measure(read_psf(os.path.join(raw, f"det{k}-{j:03d}_trd{k}.tran.tran"))) for j in range(2)]
    td1, tau = m[0]["td"] * 1e12, (m[1]["td"] - m[0]["td"]) / np.log(100) * 1e12
    npl, ntot = [], []
    for j in range(len(DVN)):
        d = read_psf(os.path.join(raw, f"noi{k}-{j:03d}_trn{k}.tran.tran")); t = d["time"]
        te = 200e-12 + 10e-12 + 1.2e-9 + np.arange(NCYC) * TPER
        i = np.searchsorted(t, te[te < t[-1]])
        npl.append(int((d["outp"][i] > d["outn"][i]).sum())); ntot.append(len(i))
    npl, ntot, dv = np.array(npl), np.array(ntot), np.array(DVN)
    f = lambda ls: -(npl * np.log(ndtr(dv / np.exp(ls))) + (ntot - npl) * np.log(1 - ndtr(dv / np.exp(ls)))).sum()
    sig = np.exp(minimize_scalar(f, bounds=(np.log(5e-5), np.log(5e-3)), method="bounded").x)
    res.append(dict(cand=name, win=win, wtail=wtail, fmax=fmax, td_1mV_ps=td1, tau_ps=tau, E_fJ=m[0]["energy"] * 1e15,
                    n_plus=npl.tolist(), n_tot=ntot.tolist(), sigma_mV=sig * 1e3))
    print(f"{name:12s} {win:>4s} {wtail:>4s} {fmax:>5s} | {td1:7.1f} {tau:5.2f} {m[0]['energy'] * 1e15:5.1f} | "
          + " ".join(f"{a / b:7.3f}" for a, b in zip(npl, ntot)) + f" | {sig * 1e3:.3f} mV")
json.dump(res, open(os.path.join(LATCH, "data", f"{NAME}.json"), "w"), indent=1)
