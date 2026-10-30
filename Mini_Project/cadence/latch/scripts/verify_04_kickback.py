"""Verification 04: kickback onto a SAR-like capacitive input. The ideal input sources are replaced by a floating
capacitor C_DAC per side (CDAC top plate), precharged to vcm +- dvin/2 by an initial condition. Three back-to-back
comparisons per transient. Reports, per case:
  - common-mode kickback peak during evaluation
  - differential kickback before the decision (input-referred error the comparator itself sees) and at the decision time
  - decision delay vs the ideal-source delay
  - residual differential/common-mode error at the start of the next comparisons (what corrupts the next SAR bit)
Clocks: relaxed (1.2 ns evaluate / 0.8 ns reset) and SAR-like (0.4 ns / 0.2 ns).  Usage: [--analyze]"""
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from sa_sim import LATCH, circuit, run, TOL
from cf import read_psf, crossing

CDAC = ["100f", "300f", "1p"]
DV = [1e-3, 0.1]
CLOCKS = {"relaxed": dict(tpw="1.2n", tper="2n", stop=0.2e-9 + 3 * 2e-9),
          "sar": dict(tpw="0.4n", tper="0.6n", stop=0.2e-9 + 3 * 0.6e-9)}
NCYC = 3
TD_IDEAL = {1e-3: 105.76, 0.1: 60.79}  # ps, ideal sources (verify_01 on the final design, tight tolerances)


def base():
    txt = circuit()
    lines = [ln for ln in txt.splitlines() if not ln.startswith(("VINP ", "VINN "))]
    assert len(lines) == len(txt.splitlines()) - 2
    return "\n".join(lines) + """
parameters cdac=300f
CDP (inp 0) capacitor c=cdac
CDN (inn 0) capacitor c=cdac
ic inp=vcm+dvin/2 inn=vcm-dvin/2
"""


if "--analyze" not in sys.argv:
    blocks = [f"simOpts options {TOL} temp=27 tnom=27 save=selected",
              "save inp inn clk outp outn I_SA.P I_SA.Q V0:p"]
    for ck, c in CLOCKS.items():
        blocks += [f"a_{ck}_pw alter param=tpw value={c['tpw']}", f"a_{ck}_per alter param=tper value={c['tper']}",
                   f"{ck}_c sweep param=cdac values=[{' '.join(CDAC)}] {{",
                   f"{ck}_d sweep param=dvin values=[{' '.join(f'{v:g}' for v in DV)}] {{",
                   f"  {ck}_tr tran stop={c['stop']:g} maxstep=1p errpreset=conservative ic=all",
                   "}", "}"]
    try:
        run("v04_kickback", "\n".join(blocks), base=base())
    except RuntimeError as e:
        print(e)


def cycle_metrics(d, tper):
    t, vin_p, vin_n = d["time"], d["inp"], d["inn"]
    diff, cm = vin_p - vin_n, (vin_p + vin_n) / 2
    rise = [crossing(t[(t > a) & (t < a + tper)], d["clk"][(t > a) & (t < a + tper)], 0.5, True)
            for a in 0.1e-9 + np.arange(NCYC) * tper]
    at = lambda sig, tt: np.interp(tt, t, sig)
    out = []
    for k, tr in enumerate(rise):
        win = (t > tr) & (t < tr + 0.9 * tper)
        tp = crossing(t[win], d["outp"][win], 0.5, True)
        tn = crossing(t[win], d["outn"][win], 0.5, True)
        td = np.nanmin([tp, tn]) - tr
        pre = win & (t < tr + td)
        d0, c0 = at(diff, tr - 15e-12), at(cm, tr - 15e-12)  # before the 20 ps clock edge starts
        out.append(dict(td_ps=td * 1e12, sign=1 if (np.isnan(tn) or tp < tn) else -1,
                        diff_start_mV=d0 * 1e3, cm_start_mV=c0 * 1e3,
                        cm_kick_peak_mV=(cm[win] - c0)[np.argmax(np.abs(cm[win] - c0))] * 1e3,
                        diff_kick_pre_mV=(diff[pre] - d0)[np.argmax(np.abs(diff[pre] - d0))] * 1e3 if pre.any() else np.nan,
                        diff_kick_at_td_mV=(at(diff, tr + td) - d0) * 1e3,
                        diff_kick_peak_mV=(diff[win] - d0)[np.argmax(np.abs(diff[win] - d0))] * 1e3))
    return out


res = {}
for ck, c in CLOCKS.items():
    tper = float(c["tper"].rstrip("n")) * 1e-9
    files = {tuple(int(i) for i in re.findall(r"_[cd]-(\d+)", os.path.basename(f))): f
             for f in glob.glob(os.path.join(LATCH, "raw", "v04_kickback", f"{ck}_c-*_{ck}_d-*_{ck}_tr.tran.tran"))}
    print(f"=== clock {ck}: evaluate {c['tpw']}, period {c['tper']} ({len(files)} transients)")
    print(f"{'Cdac':>5} {'dVin':>6} {'cyc':>3} | {'t_d ps':>7} {'(ideal)':>7} | {'start diff mV':>13} {'start cm mV':>11} |"
          f" {'CM kick':>8} {'dKick<td':>8} {'dKick@td':>8} {'dKick pk':>8}  (mV)")
    for i, cd in enumerate(CDAC):
        for j, dv in enumerate(DV):
            m = cycle_metrics(read_psf(files[(i, j)]), tper)
            res[f"{ck} {cd} {dv:g}"] = m
            for k, x in enumerate(m):
                print(f"{cd:>5} {dv * 1e3:5.0f}m {k:3d} | {x['td_ps']:7.1f} {TD_IDEAL[dv]:7.1f} | {x['diff_start_mV']:13.6f}"
                      f" {x['cm_start_mV']:11.4f} | {x['cm_kick_peak_mV']:8.2f} {x['diff_kick_pre_mV']:8.4f}"
                      f" {x['diff_kick_at_td_mV']:8.3f} {x['diff_kick_peak_mV']:8.2f}")
            r = [m[k]["diff_start_mV"] - m[0]["diff_start_mV"] for k in range(1, NCYC)]
            print(f"{'':16s} residual differential error at cycles 1, 2: " + ", ".join(f"{x * 1e3:+.3f} uV" for x in r))
json.dump(res, open(os.path.join(LATCH, "data", "v04_kickback.json"), "w"), indent=1)
