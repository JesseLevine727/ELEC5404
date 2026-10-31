"""Experiment 09: SAR metastability error rate (PROJECT_PLAN.md section 7, Fig 9, Table 5), with meta.py.
For each of the 9 Spectre reference conditions (VCM = VDD/2; VDD 0.9 / 1.0 / 1.1 V x T 0 / 27 / 85 C):
  - the rate vs fs of every network of the three frozen families (5 seeds each), by the threshold sum
  - the Spectre reference: the same calculation with the delay curve simulated directly (data/meta/meta_curves.csv,
    10 fV - VDD; below ~1 pV Spectre's delay is uncertain by ~1 ps, so the reference is trusted down to 1e-10)
  - the sampling rate at which each crosses 1e-3 / 1e-6 / 1e-9 / 1e-12, and each model's rate where the
    reference is 1e-6 and 1e-9; also with the other 9 delays from Spectre ("extrapolation only"), which removes
    the in-range errors and the 0.3 V clamp and leaves the stuck, extrapolated comparison
Also: the KBNN closed form (vs its table), f_max(1e-9) vs VDD for the models, and a brute-force GPU Monte Carlo
cross-check at the nominal condition (Table 5 timing). Writes results/exp_09_meta.npz and exp_09_meta.csv."""
import csv
import os
import time

import numpy as np
import torch

from nets import RESULTS, T_UNIT, features, rebuild, pick
import meta

LATCH_META = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "cadence", "latch", "data", "meta")
FAMS = ["kbnn_lin", "mlp_relu", "mlp_tanh"]
FS = np.linspace(30e6, 700e6, 13401)  # 50 kHz steps
TARGETS = (1e-3, 1e-6, 1e-9, 1e-12)
REF_FLOOR = 1e-10  # below this the reference relies on Spectre delays under ~1 pV
VDDS, TEMPS = (0.9, 1.0, 1.1), (0, 27, 85)

final = torch.load(os.path.join(RESULTS, "final.pt"), weights_only=False)
models = {fam: rebuild(final[fam]) for fam in FAMS}
sp = list(csv.DictReader(open(os.path.join(LATCH_META, "meta_curves.csv"))))
out, rows = {"fs": FS}, []
for vdd in VDDS:
    for temp in TEMPS:
        z, vfs = meta.condition(vdd, temp)
        tag = f"vdd{vdd:g}_t{temp}"
        pts = sorted((float(r["dvin_V"]), float(r["td_ps"])) for r in sp
                     if abs(float(r["vdd_V"]) - vdd) < 1e-9 and abs(float(r["temp_C"]) - temp) < 1e-6 and r["ok"] == "1")
        dv, td = map(np.array, zip(*pts))
        dv, keep = np.unique(dv, return_index=True)  # the grid's 10^0 and the appended VDD coincide at VDD = 1 V
        td = td[keep]
        sp_tab = meta.spectre_table(dv, td)
        ref = meta.rate(sp_tab, FS, vfs)
        out[f"{tag}_spectre"] = ref
        row = dict(vdd=vdd, temp=temp, spectre_points=len(dv), spectre_dv_min=dv[0])
        f_ref = {t: meta.fs_at(FS, ref, t) for t in TARGETS}
        for t in TARGETS:
            row[f"spectre_fs_{t:.0e}"] = f_ref[t] / 1e6 if t >= REF_FLOOR else np.nan
        for fam in FAMS:
            tabs = meta.model_tables(models[fam], z)
            r = np.array([meta.rate(tab, FS, vfs) for tab in tabs])  # (seeds, fs)
            out[f"{tag}_{fam}"] = r
            for t in TARGETS:
                f = np.array([meta.fs_at(FS, ri, t) for ri in r]) / 1e6
                row[f"{fam}_fs_{t:.0e}"], row[f"{fam}_fs_{t:.0e}_min"], row[f"{fam}_fs_{t:.0e}_max"] = np.median(f), f.min(), f.max()
            # extrapolation only: the other 9 delays from Spectre, the stuck comparison from the model
            rx = np.array([meta.rate(tab, FS, vfs, rest_table=sp_tab) for tab in tabs])
            out[f"{tag}_{fam}_xonly"] = rx
            for t in (1e-6, 1e-9):
                v = rx[:, np.searchsorted(FS, f_ref[t])] if np.isfinite(f_ref[t]) else np.full(len(rx), np.nan)
                row[f"{fam}_xonly_rate_at_ref_{t:.0e}"] = np.median(v)
                row[f"{fam}_xonly_rate_at_ref_{t:.0e}_min"], row[f"{fam}_xonly_rate_at_ref_{t:.0e}_max"] = v.min(), v.max()
            for t in (1e-6, 1e-9):  # the model's rate where the reference is t
                v = r[:, np.searchsorted(FS, f_ref[t])] if np.isfinite(f_ref[t]) else np.full(len(r), np.nan)
                row[f"{fam}_rate_at_ref_{t:.0e}"] = np.median(v)
                row[f"{fam}_rate_at_ref_{t:.0e}_min"], row[f"{fam}_rate_at_ref_{t:.0e}_max"] = v.min(), v.max()
            if fam == "kbnn_lin":  # closed form vs the table, every seed
                with torch.no_grad():
                    q = models[fam].parts(features([1e-3], z[None]))
                rr = meta.RS * vfs
                dev = []
                for s, tab in enumerate(tabs):
                    rest = np.where(meta.RS > 0, tab(np.where(rr > 0, rr, 1.0)), 0).sum(1)
                    rc = meta.rate_kbnn_closed(float(q["t0"][s, 0]) * T_UNIT, float(q["tau"][s, 0]) * T_UNIT, rest, FS, vfs)
                    ok = (r[s] > 1e-15) & (r[s] < 1e-2)
                    dev.append(np.abs(np.log(rc[ok] / r[s][ok])).max())
                row["kbnn_closed_vs_table_max_ln"] = max(dev)
        rows.append(row)
        print(f"\nVDD {vdd} V, {temp} C ({len(dv)} Spectre points from {dv[0]:.0e} V). fs (MS/s) at rate"
              f" 1e-3 / 1e-6 / 1e-9 / 1e-12; model = median of 5 seeds [min-max]:")
        print("  Spectre   " + " / ".join(f"{row[f'spectre_fs_{t:.0e}']:7.2f}" for t in TARGETS))
        for fam in FAMS:
            print(f"  {fam:9s} " + " / ".join(f"{row[f'{fam}_fs_{t:.0e}']:7.2f}" for t in TARGETS)
                  + f"   [1e-9: {row[f'{fam}_fs_1e-09_min']:.2f}-{row[f'{fam}_fs_1e-09_max']:.2f}]"
                  + "   rate where Spectre is 1e-6 / 1e-9: "
                  + " / ".join(f"{row[f'{fam}_rate_at_ref_{t:.0e}']:.1e}" for t in (1e-6, 1e-9))
                  + " (extrapolation only: " + " / ".join(f"{row[f'{fam}_xonly_rate_at_ref_{t:.0e}']:.1e}" for t in (1e-6, 1e-9)) + ")")
        print(f"  KBNN closed form vs table: max |ln ratio| {row['kbnn_closed_vs_table_max_ln']:.1e}")

# f_max(1e-9) vs VDD for the models (continuous; Spectre only at the 9 conditions above)
vgrid = np.linspace(0.9, 1.1, 21)
fs2 = FS[::5]  # 250 kHz steps; fs_at interpolates between them
out["vdd_grid"] = vgrid
for temp in TEMPS:
    for fam in FAMS:
        f = np.zeros((len(vgrid), len(models[fam].config["seeds"])))
        for i, vdd in enumerate(vgrid):
            z, vfs = meta.condition(vdd, temp)
            f[i] = [meta.fs_at(fs2, meta.rate(tab, fs2, vfs), 1e-9) / 1e6 for tab in meta.model_tables(models[fam], z)]
        out[f"fmax_t{temp}_{fam}"] = f

# brute-force Monte Carlo cross-check at the nominal condition, representative networks (Table 5 timing)
z, vfs = meta.condition(1.0, 27)
fs_mc = np.linspace(230e6, 280e6, 51)
for fam, n in (("kbnn_lin", 1_000_000_000), ("mlp_relu", 200_000_000), ("mlp_tanh", 200_000_000)):
    net = pick(models[fam], [final[fam]["rep"]])
    r_mc, fails, sec = meta.mc_rate(net, z, vfs, fs_mc, n)
    tab = meta.model_tables(net, z)[0]
    t = time.perf_counter()
    r_th = meta.rate(tab, fs_mc, vfs)
    t_th = (time.perf_counter() - t) / len(fs_mc)
    out[f"mc_{fam}"], out[f"mc_{fam}_th"], out[f"mc_{fam}_fails"] = r_mc, r_th, fails
    ok = (fails >= 100) & (r_th < 1e-2)  # the threshold sum is for rare events (one stuck comparison)
    dev = np.abs(r_mc[ok] / r_th[ok] - 1)
    print(f"\nMC {fam}: {n:.0e} conversions ({n * meta.N_BITS:.0e} delays) in {sec:.1f} s; threshold sum {1e3 * t_th:.2f} ms"
          f" per fs. Where MC has >= 100 failures and the rate is < 1e-2 (rates {r_mc[ok].min():.1e}-{r_mc[ok].max():.1e}):"
          f" |MC / threshold sum - 1| <= {dev.max():.3f}; counting noise 1/sqrt(failures) <= {1 / np.sqrt(fails[ok].min()):.3f}")
    for k in np.flatnonzero(ok | ((r_th >= 1e-2) & (r_th < 0.9))):
        print(f"  {fs_mc[k] / 1e6:6.1f} MS/s  threshold sum {r_th[k]:9.3e}  MC {r_mc[k]:9.3e}  ({fails[k]} failures)")
    out[f"mc_{fam}_seconds"], out[f"mc_{fam}_n"], out["fs_mc"] = sec, n, fs_mc
    out[f"th_{fam}_seconds_per_fs"] = t_th

np.savez(os.path.join(RESULTS, "exp_09_meta.npz"), **out)
with open(os.path.join(RESULTS, "exp_09_meta.csv"), "w") as fh:
    keys = list(dict.fromkeys(k for r in rows for k in r))
    fh.write(",".join(keys) + "\n")
    for r in rows:
        fh.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], (float, np.floating)) else str(r[k]) for k in keys) + "\n")
