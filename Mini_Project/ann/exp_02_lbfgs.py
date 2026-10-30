"""Experiment 02: training method, Adam vs L-BFGS (PROJECT_PLAN.md section 5, optional optimizer comparison;
also comparison 5, training cost). Same families, sizes and seeds as exp_01 (KBNN with the linear correction
input only); exp_01's Adam results (30k epochs) are the reference.
  lbfgs        L-BFGS from the initial weights
  adam+lbfgs   2000 Adam epochs, then L-BFGS
Usage: [--iters N]"""
import os
import sys

import numpy as np

from nets import DEV, RESULTS, MLP, KBNN, load, train, train_lbfgs, summary, save

SEEDS = range(5)
H_MLP = [5, 10, 20, 40, 80, 160]
H_KB = [3, 5, 10, 20, 40, 80]
ITERS = int(sys.argv[sys.argv.index("--iters") + 1]) if "--iters" in sys.argv else 3000

tr, va, te, ex = (load(n) for n in ("train", "val", "test", "extrap"))
y_mean = tuple(tr["y"].mean(0).tolist())
grid = lambda hs: ([h for h in hs for _ in SEEDS], [s for _ in hs for s in SEEDS])
(wm, sm), (wk, sk) = grid(H_MLP), grid(H_KB)
make = {"mlp_relu": lambda: MLP(wm, sm, y_mean=y_mean), "mlp_tanh": lambda: MLP(wm, sm, act="tanh", y_mean=y_mean),
        "kbnn_lin": lambda: KBNN(wk, wk, sk)}
ref = np.genfromtxt(os.path.join(RESULTS, "exp_01_sizes.csv"), delimiter=",", names=True, dtype=None, encoding=None)
ref_hist = np.load(os.path.join(RESULTS, "exp_01_sizes_hist.npz"))

rows, hist, models = [], {}, {}
print(f"L-BFGS {ITERS} iterations; |error| in %, mean over 5 seeds (worst seed in brackets); val = RMS relative error")
for fam in make:
    for method in ("lbfgs", "adam+lbfgs"):
        model = make[fam]().to(DEV)
        pre = train(model, tr, va, steps=2000)["seconds"] if method == "adam+lbfgs" else 0.0
        info = train_lbfgs(model, tr, va, iters=ITERS)
        key = f"{fam}_{method}"
        hist[key], models[key] = info, model
        s = summary(model, [va, te, ex])
        widths = model.config.get("widths") or model.config["widths_p"]
        print(f"\n== {key}: {len(widths)} models, {pre + info['seconds']:.0f} s, {info['evals']} loss evaluations")
        print(f"{'width':>5} | {'val RMS':>15} | {'test t_d avg':>12} {'max':>13} | {'test E avg':>10} {'max':>13} |"
              f" {'extrap t_d avg':>14} {'max':>13} | {'extrap E avg':>12} {'max':>13} | best iter")
        for h in sorted(set(widths)):
            i = np.flatnonzero(np.array(widths) == h)
            r = ref[(ref["family"] == fam) & (ref["width"] == h)]
            f = lambda k: f"{s[k][i].mean():.3f}"
            g = lambda k: f"{s[k][i].mean():.3f} ({s[k][i].max():.2f})"
            vrms = 100 * np.sqrt(info["hist_val"][:, i].min(0)).mean()
            vref = 100 * np.sqrt(ref_hist[f"{fam}_hist_val"][:, ref_hist[f"{fam}_widths"] == h].min(0)).mean()
            print(f"{h:5d} | {vrms:15.3f} | {f('test_td_avg'):>12} {g('test_td_max'):>13} | {f('test_E_avg'):>10}"
                  f" {g('test_E_max'):>13} | {f('extrap_td_avg'):>14} {g('extrap_td_max'):>13} | {f('extrap_E_avg'):>12}"
                  f" {g('extrap_E_max'):>13} | {int(np.median(info['best_step'][i]))}")
            print(f"{'Adam':>5} | {vref:15.3f} | {np.mean(r['test_td_avg']):12.3f} {np.mean(r['test_td_max']):13.3f} |"
                  f" {np.mean(r['test_E_avg']):10.3f} {np.mean(r['test_E_max']):13.3f} | {np.mean(r['extrap_td_avg']):14.3f}"
                  f" {np.mean(r['extrap_td_max']):13.3f} | {np.mean(r['extrap_E_avg']):12.3f} {np.mean(r['extrap_E_max']):13.3f} |"
                  " (exp_01, 30k epochs)")
        for m in range(len(widths)):
            rows.append(dict(family=fam, method=method, width=int(widths[m]), seed=int(model.config["seeds"][m]),
                             params=int(model.n_params[m]), best_iter=int(info["best_step"][m]),
                             train_s=float(pre + info["seconds"]), **{k: float(v[m]) for k, v in s.items()}))
save("exp_02_lbfgs", models, rows)
np.savez(os.path.join(RESULTS, "exp_02_lbfgs_hist.npz"),
         **{f"{k}_{n}": v for k, h in hist.items() for n, v in h.items() if n not in ("seconds", "evals")},
         **{f"{k}_widths": np.array(m.config.get("widths") or m.config["widths_p"]) for k, m in models.items()})
