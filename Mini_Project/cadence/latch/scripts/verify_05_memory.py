"""Verification 05: decision memory (hysteresis) from incomplete reset. Ideal differential input from a pulse source
alternating each cycle between +100 mV and -delta (changed mid-reset). If the latch remembers the previous '+'
decision, the -delta cycles decide '+' when delta is below the memory threshold. Also records X/Y/P/Q at the end
of each reset. Clocks: relaxed (1.2/0.8 ns), SAR-like (0.4/0.2 ns), SAR-short (0.4/0.13 ns).  Usage: [--analyze]"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sa_sim import LATCH, circuit, run, TOL
from cf import read_psf, crossing

DELTA = ["1u", "3u", "10u", "30u", "100u", "300u", "1m"]
CLOCKS = {"relaxed": ("1.2n", "2n"), "sar": ("0.4n", "0.6n"), "sar_short": ("0.4n", "0.53n")}
NCYC = 8
BIG = "100m"


def base():
    txt = circuit()
    lines = [ln for ln in txt.splitlines() if not ln.startswith(("VINP ", "VINN "))]
    assert len(lines) == len(txt.splitlines()) - 2
    # dvn = +BIG on even cycles, -delta on odd cycles, switching in the middle of each reset phase
    return "\n".join(lines) + f"""
parameters delta=10u tmid0=220p+(tpw+tper)/2
VCMS (cm 0) vsource dc=vcm
EP (inp cm dvn 0) vcvs gain=0.5
EN (inn cm dvn 0) vcvs gain=-0.5
VDV (dvn 0) vsource type=pulse val0={BIG} val1=-delta delay=tmid0 rise=20p fall=20p width=tper-20p period=2*tper
"""


if __name__ == "__main__":
    if "--analyze" not in sys.argv:
        blocks = [f"simOpts options {TOL} temp=27 tnom=27 save=selected",
                  "save dvn clk outp outn I_SA.X I_SA.Y I_SA.P I_SA.Q I_SA.TAIL"]
        for ck, (pw, per) in CLOCKS.items():
            stop = 0.2e-9 + NCYC * float(per.rstrip("n")) * 1e-9
            blocks += [f"a_{ck}_pw alter param=tpw value={pw}", f"a_{ck}_per alter param=tper value={per}",
                       f"{ck}_d sweep param=delta values=[{' '.join(DELTA)}] {{",
                       f"  {ck}_tr tran stop={stop:g} maxstep=1p errpreset=conservative", "}"]
        try:
            run("v05_memory", "\n".join(blocks), base=base())
        except RuntimeError as e:
            print(e)

    res = {}
    for ck, (pw, per) in CLOCKS.items():
        tper = float(per.rstrip("n")) * 1e-9
        files = {int(re.search(rf"{ck}_d-(\d+)", os.path.basename(f)).group(1)): f
                 for f in glob.glob(os.path.join(LATCH, "raw", "v05_memory", f"{ck}_d-*_{ck}_tr.tran.tran"))}
        print(f"=== clock {ck}: evaluate {pw}, period {per} (reset ~{(tper - float(pw.rstrip('n')) * 1e-9 - 40e-12) * 1e12:.0f} ps flat)")
        for i, de in enumerate(DELTA):
            if i not in files:
                print(f"  delta {de}: missing"); continue
            d = read_psf(files[i]); t = d["time"]
            rows = []
            for k in range(NCYC):
                tr = 0.2e-9 + 10e-12 + k * tper
                w = (t > tr) & (t < tr + 0.9 * (tper - 0.05e-9))
                tp, tn = crossing(t[w], d["outp"][w], 0.5, True), crossing(t[w], d["outn"][w], 0.5, True)
                dec = 0 if (np.isnan(tp) and np.isnan(tn)) else (+1 if (np.isnan(tn) or tp < tn) else -1)
                pre = tr - 15e-12
                lv = {n: np.interp(pre, t, d[f"I_SA.{n}"]) for n in "XYPQ"} | {n: np.interp(pre, t, d[n]) for n in ("outp", "outn")}
                rows.append(dict(cycle=k, dv=np.interp(pre, t, d["dvn"]), dec=dec,
                                 td_ps=(np.nanmin([tp, tn]) - tr) * 1e12, **lv))
            odd = [r for r in rows if r["cycle"] % 2 == 1]
            wrong = sum(r["dec"] != -1 for r in odd)
            r1 = rows[1]
            res[f"{ck} {de}"] = rows
            print(f"  delta {de:>5}: -delta cycles decided correctly {len(odd) - wrong}/{len(odd)};"
                  f" before cycle 1: X {r1['X']:.4f} Y {r1['Y']:.4f} P {r1['P']:.4f} Q {r1['Q']:.4f} V; outp/outn {r1['outp']:.3f}/{r1['outn']:.3f} V;"
                  f" t_d(-delta) {np.mean([r['td_ps'] for r in odd]):.1f} ps")
    json.dump(res, open(os.path.join(LATCH, "data", "v05_memory.json"), "w"), indent=1, default=float)
