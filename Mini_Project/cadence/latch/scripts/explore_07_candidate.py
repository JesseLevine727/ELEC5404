"""Exploration, all-LVT candidate (win 12u, wtail 8u, wrst 250n): latch 1u vs 2u, output inverter 500n/700n vs
250n/1.4u, delta-Vin from 300 mV to 1 nV. Checks the log law and the losing-output peak (double decisions)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_sim
sa_sim.ADE = sa_sim.FIRSTCUT
import numpy as np
from sa_sim import LATCH, sweep_sizes

DV = [0.3, 0.1, 0.03, 0.01, 3e-3, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]
grid = dict(win=["12u"], wtail=["8u"], wrst=["250n"], wlatn=["1u", "2u"], wlatp=["1u", "2u"],
            winvn=["500n", "250n"], winvp=["700n", "1.4u"])
rows = sweep_sizes("07_candidate", grid, dv=DV, simulate="--analyze" not in sys.argv)
json.dump(rows, open(os.path.join(LATCH, "data", "07_candidate.json"), "w"), indent=1)
lv = np.log(DV)
for r in rows:
    td = np.array(r["td_all_ps"])
    # local tau between neighbouring decades (from 1 mV down)
    taus = [-(td[j + 1] - td[j]) / (lv[j + 1] - lv[j]) for j in range(5, len(DV) - 1)]
    print(f"latn {r['wlatn']:>3} latp {r['wlatp']:>3} inv {r['winvn']:>4}/{r['winvp']:<4} | t_d(1mV) {td[5]:6.1f}"
          f" t_d(300mV) {td[0]:5.1f} | tau 1m..1n: " + " ".join(f"{x:5.1f}" for x in taus)
          + f" | max losing out {np.nanmax(r['vlose']):.3f} V | E {r['E_fJ']:.1f}")
