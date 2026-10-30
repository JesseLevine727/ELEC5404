"""Exploration, all-LVT: refine around the 04 optimum (wider input pair, reset switch width)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sa_sim
sa_sim.ADE = sa_sim.FIRSTCUT
import numpy as np
from sa_sim import LATCH, sweep_sizes, show

GRID = dict(win=["8u", "12u", "16u"], wtail=["6u", "8u", "12u"], wlatn=["1u", "2u"], wlatp=["1u", "2u"],
            wrst=["250n", "500n"])
rows = sweep_sizes("05_refine_lvt", GRID)
json.dump(rows, open(os.path.join(LATCH, "data", "05_refine_lvt.json"), "w"), indent=1)
print("failed:", [tuple(r[k] for k in GRID) for r in rows if np.isnan(r["tau_ps"])])
print("Fastest by t_d(1 mV):"); show(rows, list(GRID))
print("Lowest tau:"); show(rows, list(GRID), "tau_ps", 6)
