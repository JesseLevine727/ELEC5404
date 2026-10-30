"""Verification 03: input-referred noise by transient noise (Spectre trannoise). For each fixed delta-Vin the latch
runs NCYC clock cycles; the fraction of '+' decisions is fitted to P = Phi((dv - mu) / sigma_n) by maximum likelihood.
Usage: verify_03_noise.py [--fmax 100G] [--analyze]"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.optimize import minimize
from scipy.special import ndtr
from sa_sim import LATCH, run

DV = [0, 0.2e-3, 0.4e-3, 0.6e-3, 0.8e-3, 1.2e-3]
NCYC, TPER = 400, 2e-9
FMAX = sys.argv[sys.argv.index("--fmax") + 1] if "--fmax" in sys.argv else "100G"
NAME = f"v03_noise_{FMAX}"

if "--analyze" not in sys.argv:
    try:
        run(NAME, f"""simOpts options reltol=1e-5 vabstol=1e-9 iabstol=1e-12 temp=27 tnom=27 save=selected
save outp outn clk
dv_sw sweep param=dvin values=[{' '.join(f'{v:g}' for v in DV)}] {{
  tr tran stop={200e-12 + NCYC * TPER:g} maxstep=2p errpreset=moderate noisefmax={FMAX} noiseseed=7 noisescale=1
}}""")
    except RuntimeError as e:
        print(e)

files = {int(re.search(r"dv_sw-(\d+)", f).group(1)): f for f in glob.glob(os.path.join(LATCH, "raw", NAME, "dv_sw-*_tr.tran.tran"))}
from cf import read_psf
n_plus, n_tot = [], []
for j, dv in enumerate(DV):
    d = read_psf(files[j]); t = d["time"]
    # sample just before each falling clock edge (end of evaluation)
    te = 200e-12 + 10e-12 + 1.2e-9 + np.arange(NCYC) * TPER
    te = te[te < t[-1]]
    i = np.searchsorted(t, te)
    plus = d["outp"][i] > d["outn"][i]
    undecided = np.abs(d["outp"][i] - d["outn"][i]) < 0.5
    n_plus.append(int(plus.sum())); n_tot.append(len(i))
    print(f"dv {dv * 1e3:5.2f} mV: {plus.sum():4d}/{len(i)} '+' decisions  (P = {plus.mean():.3f}; undecided {undecided.sum()})")
n_plus, n_tot, dvs = np.array(n_plus), np.array(n_tot), np.array(DV)


def nll(p):
    mu, ls = p
    q = np.clip(ndtr((dvs - mu) / np.exp(ls)), 1e-12, 1 - 1e-12)
    return -(n_plus * np.log(q) + (n_tot - n_plus) * np.log(1 - q)).sum()


r = minimize(nll, [0.0, np.log(0.4e-3)], method="Nelder-Mead", options=dict(xatol=1e-9, fatol=1e-9))
mu, sig = r.x[0], np.exp(r.x[1])
# profile-likelihood 95 % interval for sigma
grid = sig * np.linspace(0.6, 1.6, 401)
prof = np.array([minimize(lambda m: nll([m[0], np.log(g)]), [mu], method="Nelder-Mead").fun for g in grid])
ok = grid[prof - r.fun < 1.92]
print(f"fit ({FMAX}): sigma_n = {sig * 1e3:.3f} mV rms (95 % interval {ok.min() * 1e3:.3f} .. {ok.max() * 1e3:.3f}),"
      f" mu = {mu * 1e3:+.3f} mV")
json.dump(dict(dv=DV, n_plus=n_plus.tolist(), n_tot=n_tot.tolist(), sigma_mV=sig * 1e3, mu_mV=mu * 1e3,
               sigma_ci_mV=[ok.min() * 1e3, ok.max() * 1e3], fmax=FMAX),
          open(os.path.join(LATCH, "data", f"{NAME}.json"), "w"), indent=1)
