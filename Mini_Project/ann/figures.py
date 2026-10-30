"""Report figures from the frozen models (results/final.pt) and the saved experiment results. Writes figs/<name>.pdf
(for the report) and figs/<name>.png (preview), and figs/fig8_params.csv. No training; nothing is chosen here.
  sizing     train and validation error vs weights (exp_04, exp_06): the answer to "how many hidden neurons"
  fig4       t_d and E vs dvin at the nine Spectre slices, Spectre points and model curves (comparison 2)
  fig5       error vs dvin from the test set into the extrapolation region (comparison 3, key result)
  fig6       error vs number of training points (exp_07, comparison 4)
  fig7       validation error vs epoch (exp_08, comparison 5)
  fig8       KBNN parameters tau, t0, s, E0 vs log-law fits to Spectre (comparison 6)
Usage: python3 figures.py [names...]   (default: all)"""
import csv
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullFormatter
import numpy as np
import torch

from nets import DSET, RESULTS, T_UNIT, E_UNIT, V_REF, features, load, rebuild, pick, rel_mse

FIGS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figs")
ORDER = ["kbnn_lin", "mlp_relu", "mlp_tanh"]  # also the colour order: identity follows the model in every figure
COLOR = {"kbnn_lin": "#2a78d6", "mlp_relu": "#eb6834", "mlp_tanh": "#1baf7a"}
DASH = {"kbnn_lin": "-", "mlp_relu": (0, (4, 1.5)), "mlp_tanh": (0, (1, 1.2))}  # second channel besides colour
MARK = {"kbnn_lin": "o", "mlp_relu": "s", "mlp_tanh": "D"}
INK, INK2, MUTED, GRID, AXIS, SHADE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f0efec"
E_D = 0.5
EXTRAP_EDGE = 1e-4  # smallest dvin in the training set

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "axes.grid": True, "axes.grid.which": "major",
    "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-", "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200, "pdf.fonttype": 42,
})

final = torch.load(os.path.join(RESULTS, "final.pt"), weights_only=False)
MODELS = {fam: rebuild(final[fam]) for fam in ORDER}
REP = {fam: final[fam]["rep"] for fam in ORDER}


def label(fam):
    f = final[fam]
    name = dict(kbnn_lin="KBNN", mlp_relu="ReLU MLP", mlp_tanh="tanh MLP (control)")[fam]
    per = " per block" if fam.startswith("kbnn") else ""
    return f"{name}, {f['size']} neurons{per}"


def save(fig, name):
    os.makedirs(FIGS, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGS, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  figs/{name}.pdf")


def shade_extrap(ax, text=True, top=True):
    ax.axvspan(1e-7, EXTRAP_EDGE, color=SHADE, lw=0, zorder=0)
    if text:
        ax.text(0.02, 0.97 if top else 0.03, "extrapolation", transform=ax.transAxes, ha="left",
                va="top" if top else "bottom", color=INK2, fontsize=7)


def log_ticks(ax, ticks):
    """Labelled ticks at the given values on a log y axis (instead of decades only)."""
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def predict(fam, dvin, z):
    """(t_d ps, E fJ) of every seed's network, shape (seeds, N, 2)."""
    with torch.no_grad():
        p = MODELS[fam](features(dvin, z)).cpu().numpy()
    return p * np.array([T_UNIT, E_UNIT])


def read_csv(name):
    return list(csv.DictReader(open(os.path.join(RESULTS, f"{name}.csv"))))


def slices():
    """The nine slice conditions, in file order: (z, label, dvin, t_d, E)."""
    a = np.loadtxt(os.path.join(DSET, "slices.csv"), delimiter=",", skiprows=1)
    names = ["nominal", "", "", "", "", "", "", "slowest corner", "fastest corner"]
    out = []
    for k in range(9):
        r = a[34 * k: 34 * (k + 1)]
        z = r[0, 1:4]
        title = f"VCM {z[0]:.1f} V, VDD {z[1]:.1f} V, {z[2]:.0f} °C" + (f" ({names[k]})" if names[k] else "")
        out.append((z, title, r[:, 0], r[:, 4], r[:, 5]))
    return out


# ---------------------------------------------------------------------------------------------------------------
def fig_sizing():
    tr, va = load("train"), load("val")
    rows = {}  # (family, weights) -> (train %, val %), mean over seeds; one hidden layer, Adam 60k epochs
    for exp in ("exp_06_target", "exp_04_select"):  # exp_06 wins where both have the same size
        saved = torch.load(os.path.join(RESULTS, f"{exp}.pt"), weights_only=False)
        for fam in ORDER:
            m = rebuild(saved[fam])
            w = np.array(m.config.get("widths") or m.config["widths_p"])
            with torch.no_grad():
                e = [100 * rel_mse(m(d), d["y"]).sqrt().cpu().numpy() for d in (tr, va)]
            for h in sorted(set(w)):
                i = w == h
                rows.setdefault((fam, m.n_params[np.flatnonzero(i)[0]]), (e[0][i].mean(), e[1][i].mean()))
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7, 2.7))
    for fam in ORDER:
        r = sorted((p, *v) for (f, p), v in rows.items() if f == fam)
        p, a, b = (np.array(c) for c in zip(*r))
        c = COLOR[fam]
        ax.plot(p, b, color=c, ls=DASH[fam], marker=MARK[fam], ms=3.5, label=label(fam).split(",")[0].replace(" (control)", "") + ", validation")
        ax.plot(p, a, color=c, ls="none", marker=MARK[fam], ms=3.5, mfc="white", mew=0.9)
        k = list(p).index(MODELS[fam].n_params[0])
        ax.plot(p[k], b[k], "o", ms=10, mfc="none", mec=INK, mew=0.8, zorder=5)
        bx.plot(p, b / a, color=c, ls=DASH[fam], marker=MARK[fam], ms=3.5)
    ax.axhline(E_D, color=MUTED, lw=0.8)
    ax.text(0.98, E_D * 1.06, "target E_d = 0.5 %", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            color=INK2, fontsize=7)
    ax.plot([], [], ls="none", marker="o", ms=3.5, mfc="white", mec=INK2, mew=0.9, label="hollow: training")
    ax.plot([], [], ls="none", marker="o", ms=9, mfc="none", mec=INK, mew=0.8, label="chosen size")
    ax.set(xscale="log", yscale="log", xlabel="weights", ylabel="RMS relative error (%)",
           title="(a) Error vs size, one hidden layer")
    ax.legend(loc="lower left", fontsize=6.5)
    bx.axhline(1, color=MUTED, lw=0.8)
    bx.set(xscale="log", xlabel="weights", ylabel="validation / training error",
           title="(b) Overlearning: validation error ÷ training error")
    bx.text(0.03, 0.95, "> 1: memorizing training points", transform=bx.transAxes, va="top", color=INK2, fontsize=7)
    fig.tight_layout()
    save(fig, "fig_sizing")


def fig4():
    sl = slices()
    for out, col, unit, name in ((0, 3, "t_d (ps)", "fig4_td_slices"), (1, 4, "E (fJ)", "fig4b_E_slices")):
        fig, axs = plt.subplots(3, 3, figsize=(7, 6.6), sharex=True)
        dense = np.logspace(-6, math.log10(0.3), 200)
        for ax, (z, title, dv, td, E) in zip(axs.flat, sl):
            shade_extrap(ax, text=ax is axs.flat[0])
            for fam in ORDER:
                p = predict(fam, dense, np.tile(z, (len(dense), 1)))[REP[fam], :, out]
                ax.plot(dense, p, color=COLOR[fam], ls=DASH[fam], lw=1.2, label=label(fam))
            ax.plot(dv, (td, E)[out], "o", ms=2.6, color=INK, mec="white", mew=0.4, zorder=5, label="Spectre")
            ax.set_xscale("log")
            ax.set_title(title, fontsize=7)
        for ax in axs[-1]:
            ax.set_xlabel("|ΔV_in| (V)")
        for ax in axs[:, 0]:
            ax.set_ylabel(unit)
        h, l = axs.flat[0].get_legend_handles_labels()  # Spectre first, then the models in ORDER
        fig.legend(h[-1:] + h[:-1], l[-1:] + l[:-1], loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.965))
        save(fig, name)


def fig5():
    te, ex = load("test"), load("extrap")
    dv = np.concatenate([np.abs(ex["raw"][:, 0]), np.abs(te["raw"][:, 0])])
    edges = np.concatenate([np.logspace(-6, -4, 7), np.logspace(-4, math.log10(0.31), 11)[1:]])
    mid = np.sqrt(edges[:-1] * edges[1:])
    b = np.digitize(dv, edges) - 1
    fig, axs = plt.subplots(2, 2, figsize=(7, 4.6), sharex=True)
    for fam in ORDER:
        with torch.no_grad():
            err = np.concatenate([(MODELS[fam](d) - d["y"]).cpu().numpy() for d in (ex, te)], 1)
        err = np.abs(err) * np.array([T_UNIT, E_UNIT])  # (seeds, N, 2) in ps and fJ
        for out in range(2):
            mean = np.array([err[:, b == k, out].mean(1).mean() for k in range(len(mid))])
            worst = np.array([err[:, b == k, out].max(1).mean() for k in range(len(mid))])
            for row, v in ((0, mean), (1, worst)):
                axs[row, out].plot(mid, v, color=COLOR[fam], ls=DASH[fam], marker=MARK[fam], ms=3, label=label(fam))
    for ax in axs.flat:
        shade_extrap(ax, text=ax is axs[0, 0], top=False)
        ax.set(xscale="log", yscale="log", xlim=(1e-6, 0.31))
    axs[1, 0].set_ylim(bottom=0.6)
    for ax in axs[:, 0]:
        ax.axhline(1, color=MUTED, lw=0.8)
        ax.text(0.98, 0.93, "1 ps", transform=ax.get_yaxis_transform(), ha="right", va="top", color=INK2, fontsize=7)
    axs[0, 0].set(ylabel="mean |error| (ps)", title="(a) t_d, mean error")
    axs[1, 0].set(ylabel="worst |error| (ps)", title="(c) t_d, worst error", xlabel="|ΔV_in| (V)")
    axs[0, 1].set(ylabel="mean |error| (fJ)", title="(b) E, mean error")
    axs[1, 1].set(ylabel="worst |error| (fJ)", title="(d) E, worst error", xlabel="|ΔV_in| (V)")
    axs[0, 1].legend(loc="upper right")
    fig.tight_layout()
    save(fig, "fig5_extrapolation")


def fig6():
    rows = read_csv("exp_07_data_eff")
    fig, axs = plt.subplots(1, 3, figsize=(7, 2.5))
    keys = (("test_td_avg_ps", "(a) t_d, test set (in range)", "mean |error| (ps)", (0.5, 1, 2, 5)),
            ("extrap_td_avg_ps", "(b) t_d, 1–100 µV (extrapolation)", "mean |error| (ps)", (1, 2, 5, 10, 20)),
            ("extrap_E_avg", "(c) E, 1–100 µV (extrapolation)", "mean |error| (%)", (0.2, 0.5, 1, 2, 5)))
    for ax, (key, title, ylabel, ticks) in zip(axs, keys):
        for fam in ORDER:
            r = [x for x in rows if x["family"] == fam]
            ns = sorted({int(x["n_train"]) for x in r})
            v = np.array([[float(x[key]) for x in r if int(x["n_train"]) == n] for n in ns])
            ax.fill_between(ns, v.min(1), v.max(1), color=COLOR[fam], alpha=0.15, lw=0)
            ax.plot(ns, v.mean(1), color=COLOR[fam], ls=DASH[fam], marker=MARK[fam], ms=3, label=label(fam))
        ax.set(xscale="log", yscale="log", xlabel="training points N", ylabel=ylabel, title=title)
        log_ticks(ax, ticks)
        if key.endswith("ps"):
            ax.axhline(1, color=MUTED, lw=0.8)
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save(fig, "fig6_data_efficiency")


def fig7():
    h = np.load(os.path.join(RESULTS, "exp_08_cost_hist.npz"))
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for fam in ORDER:
        step, v = h[f"{fam}_batch5_step"], h[f"{fam}_batch5_val_rms"]
        ax.fill_between(step + 1, v.min(1), v.max(1), color=COLOR[fam], alpha=0.15, lw=0)
        ax.plot(step + 1, np.median(v, 1), color=COLOR[fam], ls=DASH[fam], lw=1.2, label=label(fam))
    ax.axhline(E_D, color=MUTED, lw=0.8)
    ax.text(0.02, E_D * 1.08, "E_d = 0.5 %", transform=ax.get_yaxis_transform(), ha="left", va="bottom", color=INK2,
            fontsize=7)
    ax.set(xscale="log", yscale="log", xlabel="epoch (one full-batch Adam step)", ylabel="validation RMS error (%)",
           ylim=(0.3, 30))
    ax.legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig7_training")


def lawfit(dv, y):
    """Least-squares y = y0 + slope ln(V_REF/dv): (y0, slope)."""
    A = np.column_stack([np.ones_like(dv), np.log(V_REF / dv)])
    return np.linalg.lstsq(A, y, rcond=None)[0]


def fig8():
    sl = slices()
    rows = []
    for k, (z, title, dv, td, E) in enumerate(sl):
        m = dv <= EXTRAP_EDGE * 1.0001  # 1-100 uV, 13 points
        ref = dict(t0=lawfit(dv[m], td[m])[0], tau=lawfit(dv[m], td[m])[1], E0=lawfit(dv[m], E[m])[0], s=lawfit(dv[m], E[m])[1])
        for fam in ORDER:
            p = predict(fam, dv[m], np.tile(z, (m.sum(), 1)))  # (seeds, 13, 2)
            fit = np.array([[*lawfit(dv[m], q[:, 0]), *lawfit(dv[m], q[:, 1])] for q in p])  # t0 tau E0 s
            row = dict(slice=k + 1, title=title, family=fam, **{f"spectre_{n}": ref[n] for n in ref})
            for j, n in enumerate(("t0", "tau", "E0", "s")):
                row[f"model_{n}"], row[f"model_{n}_sd"] = fit[:, j].mean(), fit[:, j].std()
            if fam.startswith("kbnn"):  # the network's own parameter outputs at this z
                with torch.no_grad():
                    q = MODELS[fam].parts(features([V_REF], z[None]))
                own = dict(t0=q["t0"] * T_UNIT, tau=q["tau"] * T_UNIT, E0=q["E0"] * E_UNIT, s=q["s"] * E_UNIT)
                for n, v in own.items():
                    row[f"kbnn_own_{n}"] = float(v.mean())
            rows.append(row)
    with open(os.path.join(FIGS, "fig8_params.csv"), "w") as fh:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(f"{r[k]:.5g}" if isinstance(r.get(k), float) else str(r.get(k, "")) for k in keys) + "\n")
    fig, axs = plt.subplots(2, 2, figsize=(7, 5.6))
    panels = (("tau", "(a) τ, regeneration time constant (ps)"), ("t0", "(b) t₀, log-law delay at 1 mV (ps)"),
              ("s", "(c) s, energy slope (fJ per neper)"), ("E0", "(d) E₀, log-law energy at 1 mV (fJ)"))
    for ax, (n, title) in zip(axs.flat, panels):
        lo = min(min(r[f"spectre_{n}"], r[f"model_{n}"]) for r in rows)
        hi = max(max(r[f"spectre_{n}"], r[f"model_{n}"]) for r in rows)
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=MUTED, lw=0.8, zorder=1)
        for fam in ORDER:
            r = [x for x in rows if x["family"] == fam]
            ax.errorbar([x[f"spectre_{n}"] for x in r], [x[f"model_{n}"] for x in r], yerr=[x[f"model_{n}_sd"] for x in r],
                        ls="none", marker=MARK[fam], ms=4.5, color=COLOR[fam], mec="white" if fam.startswith("kbnn") else COLOR[fam],
                        mfc=COLOR[fam] if fam.startswith("kbnn") else "white", mew=0.8, elinewidth=0.7, capsize=0,
                        label=label(fam) + (" (its own τ, t₀, s, E₀)" if fam.startswith("kbnn") else " (law fitted to its output)"),
                        zorder=4 if fam.startswith("kbnn") else 3)
        ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), title=title, xlabel="Spectre, fitted on 1–100 µV",
               ylabel="model")
        ax.set_aspect("equal", adjustable="box")
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=1, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "fig8_parameters")
    print("  per slice, Spectre / KBNN / ReLU MLP / tanh MLP (KBNN own parameter in brackets):")
    for n in ("tau", "t0", "s", "E0"):
        print(f"  {n}:")
        for k in range(9):
            r = {x["family"]: x for x in rows if x["slice"] == k + 1}
            print(f"    {r['kbnn_lin']['title']:44s} {r['kbnn_lin'][f'spectre_{n}']:8.3f}  "
                  f"{r['kbnn_lin'][f'model_{n}']:8.3f} [{r['kbnn_lin'][f'kbnn_own_{n}']:8.3f}]  "
                  f"{r['mlp_relu'][f'model_{n}']:8.3f}  {r['mlp_tanh'][f'model_{n}']:8.3f}")


FIGURES = dict(sizing=fig_sizing, fig4=fig4, fig5=fig5, fig6=fig6, fig7=fig7, fig8=fig8)
if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    for name in sys.argv[1:] or FIGURES:
        print(name)
        FIGURES[name]()
