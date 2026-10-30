"""Verification 02: input-referred offset from device mismatch (Monte Carlo, GPDK045 section mc, variations=mismatch).
Each MC sample runs one transient in which veriloga/offset_search.va binary-searches delta-Vin over 13 clock
cycles (resolution 16 mV / 2^12 = 4 uV). Offset = -(final dv). Usage: verify_02_offset.py [numruns] [--analyze]"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sa_sim import LATCH, circuit, run
from cf import read_psf

N = int(next((a for a in sys.argv[1:] if a.isdigit()), 30))
NAME = f"v02_offset_mc{N}"
NCYC = 13
REMOTE_VA = "/home/jesselevine/Desktop/StrongArm45/latch/veriloga/offset_search.va"


def mc_base():
    """Testbench with mismatch models and the inputs driven by vcm +- dvn/2 (dvn from the search controller)."""
    txt = circuit().replace("section=tt", "section=mc")
    lines = [ln for ln in txt.splitlines() if not ln.startswith(("VINP ", "VINN "))]
    assert len(lines) == len(txt.splitlines()) - 2
    return "\n".join(lines) + f"""
ahdl_include "{REMOTE_VA}"
VCMS (cm 0) vsource dc=vcm
EP (inp cm dvn 0) vcvs gain=0.5
EN (inn cm dvn 0) vcvs gain=-0.5
"""


if __name__ == "__main__":
    if "--analyze" not in sys.argv:
        stop = 200e-12 + NCYC * 2e-9 + 0.2e-9
        try:
            run(NAME, f"""OS (clk outp outn dvn) offset_search step0=16m
simOpts options reltol=1e-5 vabstol=1e-9 iabstol=1e-12 temp=27 tnom=27 save=selected
save dvn outp outn clk
mc montecarlo numruns={N} seed=5404 variations=mismatch donominal=no savefamilyplots=yes {{
  tr tran stop={stop:g} maxstep=2p errpreset=moderate
}}""", base=mc_base())
        except RuntimeError as e:
            print(e)
    files = sorted(glob.glob(os.path.join(LATCH, "raw", NAME, "*tr.tran.tran")))
    vos = []
    for f in files:
        d = read_psf(f)
        vos.append(-d["dvn"][-1])
    vos = np.array(vos) * 1e3
    json.dump(dict(files=[os.path.basename(f) for f in files], vos_mV=list(vos)),
              open(os.path.join(LATCH, "data", f"{NAME}.json"), "w"), indent=1)
    s = vos.std(ddof=1)
    # 95 % interval of sigma from the chi-square distribution of the sample variance
    chi = np.random.default_rng(0).chisquare(len(vos) - 1, 200000)
    lo, hi = s * np.sqrt((len(vos) - 1) / np.percentile(chi, [97.5, 2.5]))
    print(f"{len(vos)} MC samples: offset mean {vos.mean():+.3f} mV, sigma {s:.3f} mV (95 % interval {lo:.3f} .. {hi:.3f})")
    print(f"min {vos.min():+.2f} mV, max {vos.max():+.2f} mV")
    print("first samples:", " ".join(f"{v:+.2f}" for v in vos[:10]))
