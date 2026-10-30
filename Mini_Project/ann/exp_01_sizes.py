"""Experiment 01: accuracy and extrapolation vs network size, MLP vs KBNN (PROJECT_PLAN.md section 6,
comparisons 1 and 3). Five seeds per configuration; each family trains as one batch on the GPU.
  mlp_relu   4 -> H ReLU -> 2, H = 5 10 20 40 80 160
  mlp_tanh   the same with tanh (control: the textbook's classic MLP)
  kbnn_lin   parameter block h + correction block h (ReLU), h = 3 5 10 20 40 80, correction input |dvin|
  kbnn_sqrt  the same with sqrt|dvin| as the correction input
A KBNN with blocks of h has about the parameter count of the MLP with H = 2h.
Usage: [--steps N]"""
import os
import sys

import numpy as np

from nets import DEV, RESULTS, MLP, KBNN, load, train, summary, save

SEEDS = range(5)
H_MLP = [5, 10, 20, 40, 80, 160]
H_KB = [3, 5, 10, 20, 40, 80]
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 30000

tr, va, te, ex = (load(n) for n in ("train", "val", "test", "extrap"))
y_mean = tuple(tr["y"].mean(0).tolist())
grid = lambda hs: ([h for h in hs for _ in SEEDS], [s for _ in hs for s in SEEDS])
(wm, sm), (wk, sk) = grid(H_MLP), grid(H_KB)
families = {"mlp_relu": MLP(wm, sm, y_mean=y_mean), "mlp_tanh": MLP(wm, sm, act="tanh", y_mean=y_mean),
            "kbnn_lin": KBNN(wk, wk, sk), "kbnn_sqrt": KBNN(wk, wk, sk, corr="sqrt")}

rows, hist = [], {}
print(f"{STEPS} full-batch Adam steps per model, train {len(tr['y'])} rows; |error| in %, mean over 5 seeds"
      " (worst seed in brackets for the max columns)")
for fam, model in families.items():
    model.to(DEV)
    info = train(model, tr, va, steps=STEPS)
    hist[fam] = info
    s = summary(model, [va, te, ex])
    widths = model.config.get("widths") or model.config["widths_p"]
    print(f"\n== {fam}: {len(widths)} models in {info['seconds']:.0f} s")
    print(f"{'width':>5} {'params':>6} | {'test t_d avg':>12} {'max':>13} | {'test E avg':>10} {'max':>13} |"
          f" {'extrap t_d avg':>14} {'max':>13} | {'extrap E avg':>12} {'max':>13} | best step")
    for h in sorted(set(widths)):
        i = np.flatnonzero(np.array(widths) == h)
        f = lambda k: f"{s[k][i].mean():.3f}"
        g = lambda k: f"{s[k][i].mean():.3f} ({s[k][i].max():.2f})"
        print(f"{h:5d} {model.n_params[i[0]]:6d} | {f('test_td_avg'):>12} {g('test_td_max'):>13} | {f('test_E_avg'):>10}"
              f" {g('test_E_max'):>13} | {f('extrap_td_avg'):>14} {g('extrap_td_max'):>13} | {f('extrap_E_avg'):>12}"
              f" {g('extrap_E_max'):>13} | {int(np.median(info['best_step'][i]))}")
    for m in range(len(widths)):
        rows.append(dict(family=fam, width=int(widths[m]), seed=int(model.config["seeds"][m]),
                         params=int(model.n_params[m]), best_step=int(info["best_step"][m]),
                         train_s=float(info["seconds"]), **{k: float(v[m]) for k, v in s.items()}))
save("exp_01_sizes", families, rows)
# loss history (one full-batch step = one epoch), for show_history.py and the training-cost figure
np.savez(os.path.join(RESULTS, "exp_01_sizes_hist.npz"),
         **{f"{fam}_{k}": v for fam, h in hist.items() for k, v in h.items() if k != "seconds"},
         **{f"{fam}_widths": np.array(m.config.get("widths") or m.config["widths_p"]) for fam, m in families.items()})
