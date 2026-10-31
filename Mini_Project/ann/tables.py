"""Report tables from the saved results, so no number is typed by hand. Writes tables/tables.tex (booktabs table
environments to \\input), tables/tables.md (the same as markdown) and tables/preview.tex (compiles them alone).
  Table 1  variables, ranges and data sets              (dataset CSVs, data/meta)
  Table 2  model structures                             (results/final.pt)
  Table 3  errors of the frozen models, every split     (results/final_eval.csv)
  Table 4  training and evaluation cost                 (results/exp_08_cost.csv)
  Table 5  cost of a metastability error rate           (results/exp_09_meta.npz)
  Table 6  metastability predictions vs Spectre         (results/exp_09_meta.csv)"""
import csv
import os
import re

import numpy as np
import torch

from nets import DSET, RESULTS, rebuild
import meta

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
FAMS = ["mlp_relu", "mlp_tanh", "kbnn_lin"]
NAME = {"mlp_relu": "ReLU MLP", "mlp_tanh": "tanh MLP (control)", "kbnn_lin": "KBNN"}
SPECTRE_S = 0.3  # s per comparison (dataset README)
tex, md = [], []


def table(label, caption, head, rows, align=None, note=None):
    """One table in both formats. Cells are (tex, md) pairs or plain strings used for both."""
    esc = lambda t: re.sub(r"(?<!\\)%", r"\\%", t)  # a bare % would start a LaTeX comment
    cell = lambda c, i: (c[i] if isinstance(c, tuple) else c) if i else esc(c[0] if isinstance(c, tuple) else c)
    caption, note = (esc(caption[0]), caption[1]), (note and (esc(note[0]), note[1]))
    align = align or "l" + "r" * (len(head) - 1)
    tex.append(f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption[0]}}}\n\\label{{{label}}}\n\\small\n"
               f"\\begin{{tabular}}{{{align}}}\n\\toprule\n" + " & ".join(cell(h, 0) for h in head) + " \\\\\n\\midrule\n"
               + "".join(("\\midrule\n" if r == "---" else " & ".join(cell(c, 0) for c in r) + " \\\\\n") for r in rows)
               + "\\bottomrule\n\\end{tabular}\n"
               + (f"\\par\\vspace{{2pt}}{{\\footnotesize {note[0]}}}\n" if note else "") + "\\end{table}\n")
    md.append(f"**{caption[1]}**\n\n| " + " | ".join(cell(h, 1) for h in head) + " |\n|" + "---|" * len(head) + "\n"
              + "".join("| " + " | ".join(cell(c, 1) for c in r) + " |\n" for r in rows if r != "---")
              + (f"\n{note[1]}\n" if note else "") + "\n")


def both(t, m=None):
    return (t, m if m is not None else t)


# ---- Table 1: variables and data sets
def rng(name, col):
    a = np.loadtxt(os.path.join(DSET, f"{name}.csv"), delimiter=",", skiprows=1)
    return a[:, col].min(), a[:, col].max()


main = [rng(n, 4) for n in ("train", "val", "test")]
td_lo, td_hi = min(r[0] for r in main), max(r[1] for r in main)
mainE = [rng(n, 5) for n in ("train", "val", "test")]
E_lo, E_hi = min(r[0] for r in mainE), max(r[1] for r in mainE)
x_td, x_E = rng("extrap", 4), rng("extrap", 5)
n = {s: len(np.loadtxt(os.path.join(DSET, f"{s}.csv"), delimiter=",", skiprows=1)) for s in ("train", "val", "test", "extrap", "slices")}
n_ref = len(np.loadtxt(os.path.join(DSET, "..", "meta", "meta_curves.csv"), delimiter=",", skiprows=1))
table("tab:data", ("Model variables and data sets. Every point is one Spectre transient of one comparison from reset "
                   "(GPDK045, tt, LVT).", "Table 1. Variables and data sets"),
      ["Variable / set", "Range or size", "Sampling / use"],
      [[both("$\\Delta V_\\text{in}$ (input)", "ΔV_in (input)"), both("100\\,\\textmu V -- 300\\,mV", "100 µV – 300 mV"), "log-uniform"],
       ["VCM (input)", "0.4 -- 0.6 V", "uniform"], ["VDD (input)", "0.9 -- 1.1 V", "uniform"],
       [both("$T$ (input)", "T (input)"), both("0 -- 85\\,\\textdegree C", "0 – 85 °C"), "uniform"],
       [both("$t_d$ (output)", "t_d (output)"), f"{td_lo:.0f} -- {td_hi:.0f} ps", "clock 50 % to output VDD/2"],
       [both("$E$ (output)", "E (output)"), f"{E_lo:.0f} -- {E_hi:.0f} fJ", "supply energy per cycle"],
       "---",
       ["train", f"{n['train']}", "scrambled Sobol; training"],
       ["val", f"{n['val']}", "uniform random; early stopping, sizing"],
       ["test", f"{n['test']}", "uniform random; final accuracy (used once)"],
       ["extrap", f"{n['extrap']}", both(f"$\\Delta V_\\text{{in}}$ 1--100\\,\\textmu V, $t_d$ {x_td[0]:.0f}--{x_td[1]:.0f} ps; used once",
                                         f"ΔV_in 1–100 µV, t_d {x_td[0]:.0f}–{x_td[1]:.0f} ps; used once")],
       ["slices", f"{n['slices']}", both("9 conditions $\\times$ 34 $\\Delta V_\\text{in}$ (1\\,\\textmu V -- 300\\,mV); Figs 4, 8",
                                         "9 conditions × 34 ΔV_in (1 µV – 300 mV); Figs 4, 8")],
       ["reference curves", f"{n_ref}", both("VCM = VDD/2, 9 conditions, 10\\,fV -- VDD; Fig 9",
                                             "VCM = VDD/2, 9 conditions, 10 fV – VDD; Fig 9")]],
      align="lrl")

# ---- Table 2: model structures
final = torch.load(os.path.join(RESULTS, "final.pt"), weights_only=False)
w = {f: rebuild(final[f]).n_params[0] for f in FAMS}
table("tab:models", ("Structures of the frozen models (smallest network with training and validation RMS error "
                     "$\\le$ 0.5\\,\\%). Layers counted as in the course, input layer included.",
                     "Table 2. Model structures"),
      ["", "ReLU MLP", "tanh MLP (control)", "KBNN"],
      [["Layers", "3", "3", "4"],
       ["Inputs", both("$\\log_{10}|\\Delta V_\\text{in}|$, VCM, VDD, $T$", "log10|ΔV_in|, VCM, VDD, T"),
        both("same", "same"), both("VCM, VDD, $T$; $|\\Delta V_\\text{in}|\\to L, v$", "VCM, VDD, T; |ΔV_in| → L, v")],
       ["Hidden layer", f"{final['mlp_relu']['size']} ReLU", f"{final['mlp_tanh']['size']} tanh",
        f"2 blocks $\\times$ {final['kbnn_lin']['size']} ReLU" if True else ""],
       ["Linear outputs", both("$t_d$, $E$", "t_d, E"), both("$t_d$, $E$", "t_d, E"),
        both("$t_0, \\tau, E_0, s$; $g, q$", "t0, τ, E0, s; g, q")],
       ["Knowledge layer", "--", "--", both("$t_d = t_0 + \\tau L + [g(v) - g(0)]$", "t_d = t0 + τL + [g(v) − g(0)]")],
       ["", "", "", both("$E = E_0 + s L + [q(v) - q(0)]$", "E = E0 + sL + [q(v) − q(0)]")],
       ["Weights", f"{w['mlp_relu']}", f"{w['mlp_tanh']}", f"{w['kbnn_lin']}"]],
      align="llll",
      note=("All: relative-MSE loss over ($t_d$, $E$); full-batch Adam, 60\\,000 epochs, learning rate $3\\times10^{-3}\\to3\\times10^{-5}$ "
            "(cosine); best-validation weights; 5 seeds; RTX 5080, float32. KBNN: softplus on $\\tau$ and $s$; "
            "$L = \\ln(1\\,\\text{mV}/|\\Delta V_\\text{in}|)$, $v = |\\Delta V_\\text{in}|/0.3$\\,V.",
            "All: relative-MSE loss over (t_d, E); full-batch Adam, 60 000 epochs, learning rate 3e-3 → 3e-5 (cosine); "
            "best-validation weights; 5 seeds; RTX 5080, float32. KBNN: softplus on τ and s; L = ln(1 mV/|ΔV_in|), v = |ΔV_in|/0.3 V."))

# ---- Table 3: errors of the frozen models
ev = list(csv.DictReader(open(os.path.join(RESULTS, "final_eval.csv"))))
rows = []
for f in FAMS:
    r = [x for x in ev if x["family"] == f]
    if rows:
        rows.append("---")
    for k, split in enumerate(("train", "val", "test", "extrap")):
        cells = []
        for key in ("td_avg", "td_max", "E_avg", "E_max"):
            v = np.array([float(x[f"{split}_{key}"]) for x in r])
            fmt = ".2f" if key.endswith("avg") else ".1f"
            cells.append(both(f"{v.mean():{fmt}} $\\pm$ {v.std():{fmt}}", f"{v.mean():{fmt}} ± {v.std():{fmt}}"))
        label = {"train": "training", "val": "validation", "test": "test", "extrap": both("extrap. (1--100\\,\\textmu V)", "extrap. (1–100 µV)")}[split]
        rows.append([NAME[f] if k == 0 else "", label, *cells])
table("tab:errors", ("Errors of the frozen models, $|$\\% error$|$, mean $\\pm$ standard deviation over 5 seeds. "
                     "The test and extrapolation sets were evaluated once, after the sizes were fixed.",
                     "Table 3. Errors of the frozen models (|% error|, mean ± sd over 5 seeds)"),
      ["Model", "Set", both("$t_d$ avg", "t_d avg"), both("$t_d$ max", "t_d max"), both("$E$ avg", "E avg"), both("$E$ max", "E max")],
      rows, align="llrrrr")

# ---- Table 4: training and evaluation cost
cost = {r["family"]: r for r in csv.DictReader(open(os.path.join(RESULTS, "exp_08_cost.csv")))}
c = lambda f, k: float(cost[f][k])
fm = lambda x: f"{x:,.0f}".replace(",", "\\,")
table("tab:cost", ("Training and evaluation cost (RTX 5080, float32). Epochs to a target: median over the 5 seeds "
                   "trained together that reach it.", "Table 4. Training and evaluation cost"),
      ["", "ReLU MLP", "tanh MLP", "KBNN"],
      [["Weights", *[f"{int(c(f, 'params'))}" for f in FAMS]],
       ["Time per epoch (ms)", *[f"{1e3 * c(f, 'batch5_train_s') / 60000:.2f}" for f in FAMS]],
       ["60k epochs, 1 / 5 networks (s)", *[f"{c(f, 'single_train_s'):.1f} / {c(f, 'batch5_train_s'):.1f}" for f in FAMS]],
       ["Epochs to val. RMS 1 %", *[both(fm(c(f, 'batch5_epochs_to_1pct')), f"{c(f, 'batch5_epochs_to_1pct'):,.0f}") for f in FAMS]],
       ["Epochs to val. RMS 0.5 %", *[both(f"{fm(c(f, 'batch5_epochs_to_0.5pct'))} ({int(c(f, 'batch5_reached_0.5pct'))}/5)",
                                           f"{c(f, 'batch5_epochs_to_0.5pct'):,.0f} ({int(c(f, 'batch5_reached_0.5pct'))}/5)") for f in FAMS]],
       ["Time to 0.5 % (s)", *[f"{c(f, 'batch5_train_s') / 60000 * c(f, 'batch5_epochs_to_0.5pct'):.1f}" for f in FAMS]],
       [both("One prediction, CPU (\\textmu s)", "One prediction, CPU (µs)"), *[f"{1e6 * c(f, 'cpu_single_s'):.0f}" for f in FAMS]],
       ["Per prediction, batch of $10^5$: CPU / GPU (ns)" if True else "", *[f"{1e9 * c(f, 'cpu_batch_s_per_point'):.0f} / {1e9 * c(f, 'gpu_batch_s_per_point'):.1f}" for f in FAMS]],
       [both("Spectre, per comparison", "Spectre, per comparison"), both(f"$\\approx$ {SPECTRE_S} s", f"≈ {SPECTRE_S} s"), "", ""]],
      align="lrrr")
md[-1] = md[-1].replace("$10^5$", "10^5")

# ---- Table 5: cost of a metastability rate
h = np.load(os.path.join(RESULTS, "exp_09_meta.npz"))
mc = {f: float(h[f"mc_{f}_seconds"]) * 1e9 / float(h[f"mc_{f}_n"]) for f in FAMS}
th = max(float(h[f"th_{f}_seconds_per_fs"]) for f in FAMS)
years = 1e11 * meta.N_BITS * SPECTRE_S / 3.156e7
table("tab:metacost", ("Cost of one metastability error rate at the nominal condition. A rate of $10^{-9}$ to $\\pm$10\\,\\% "
                       "needs about $10^{11}$ conversions by brute force.", "Table 5. Cost of a metastability error rate"),
      ["Route", "Cost"],
      [[both("Brute-force Spectre ($10^{12}$ comparisons)", "Brute-force Spectre (10^12 comparisons)"),
        both(f"$\\approx 10^{{{np.log10(years):.0f}}}$ years", f"≈ 10^{np.log10(years):.0f} years")],
       [both("GPU Monte Carlo with a model, $10^9$ conversions", "GPU Monte Carlo with a model, 10^9 conversions"),
        f"{mc['mlp_tanh']:.0f} / {mc['mlp_relu']:.0f} / {mc['kbnn_lin']:.0f} s (tanh / ReLU / KBNN)"],
       ["Threshold sum with a model (1023 thresholds)", both(f"$<$ 1 ms", "< 1 ms")],
       ["KBNN closed form", both("$\\approx$ 5\\,\\textmu s", "≈ 5 µs")],
       [both("Threshold sum with Spectre delays (85 simulations)", "Threshold sum with Spectre delays (85 simulations)"),
        both("$\\approx$ 40 s of Spectre per condition", "≈ 40 s of Spectre per condition")]],
      align="lr")

# ---- Table 6: metastability predictions vs Spectre
mr = list(csv.DictReader(open(os.path.join(RESULTS, "exp_09_meta.csv"))))


def ratio(x, tex_fmt=True):
    if x < 1e-20:
        return both("0", "0 (no errors)")
    if x >= 0.1 and x < 100:
        return f"{x:.2g}"
    e = int(np.floor(np.log10(x)))
    m = x / 10 ** e
    return both(f"${m:.1f}\\times10^{{{e}}}$", f"{m:.1f}e{e}")


rows = []
for r in mr:
    sp = float(r["spectre_fs_1e-09"])
    rows.append([both(f"{float(r['vdd']):.1f} V, {float(r['temp']):.0f}\\,\\textdegree C", f"{float(r['vdd']):.1f} V, {float(r['temp']):.0f} °C"),
                 f"{sp:.1f}",
                 *[both(f"{100 * (float(r[f + '_fs_1e-09']) / sp - 1):+.1f}".replace("-", "$-$").replace("+", "$+$"),
                        f"{100 * (float(r[f + '_fs_1e-09']) / sp - 1):+.1f} %") for f in ("kbnn_lin", "mlp_relu", "mlp_tanh")],
                 *[ratio(float(r[f + "_rate_at_ref_1e-09"]) / 1e-9) for f in ("kbnn_lin", "mlp_relu", "mlp_tanh")]])
table("tab:meta", ("Metastability of the 10-bit asynchronous SAR (sampling 25\\,\\% of the period, 200 ps overhead per "
                   "comparison, VCM = VDD/2). Left: sampling rate for $10^{-9}$ errors per conversion, Spectre reference and "
                   "model error in \\%. Right: each model's predicted rate divided by Spectre's, where Spectre gives "
                   "$10^{-9}$ (0: the model predicts no errors). Median of 5 seeds.",
                   "Table 6. Metastability: f_max for 1e-9 and predicted rate ÷ Spectre's where Spectre gives 1e-9"),
      [both("VDD, $T$", "VDD, T"), both("$f_{\\max}$ (MS/s)", "Spectre f_max (MS/s)"), both("KBNN \\%", "f_max err: KBNN"),
       both("ReLU \\%", "ReLU MLP"), both("tanh \\%", "tanh MLP"), both("KBNN", "rate ÷: KBNN"), both("ReLU", "ReLU MLP"), both("tanh", "tanh MLP")],
      rows, align="lrrrrrrr")

os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "tables.tex"), "w").write("% generated by ann/tables.py; needs \\usepackage{booktabs, amsmath, textcomp}\n\n" + "\n".join(tex))
open(os.path.join(OUT, "tables.md"), "w").write("<!-- generated by ann/tables.py -->\n\n" + "".join(md))
open(os.path.join(OUT, "preview.tex"), "w").write(
    "\\documentclass[10pt]{article}\n\\usepackage[margin=2cm]{geometry}\n\\usepackage{booktabs, amsmath, textcomp}\n"
    "\\begin{document}\n\\input{tables.tex}\n\\end{document}\n")
print("tables/tables.tex, tables/tables.md, tables/preview.tex")
