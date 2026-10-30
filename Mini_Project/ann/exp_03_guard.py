"""Experiment 03: does a guard band around the input box fix the worst-case errors at its edges? In exp_01/02 the
worst test errors sit at large dvin and near the faces of the box, whatever the optimizer. Same families, sizes
and seeds as exp_01, Adam 30k epochs, these training sets:
  base    train                (2800 rows)
  more    train + train_ext    (3300 rows, all inside the box): the control, same amount of data as guard
  guard   train + guard        (3300 rows; 500 in a band 10 % beyond the faces, never below 100 uV)
  guard_z   train + the guard rows with |dvin| <= 0.3 V (3204 rows: guard band in vcm, vdd, temp only)
  guard_04  train + the guard rows with |dvin| <= 0.4 V
Validation, test and extrap sets are unchanged. Prints the test errors by region (distance from the box edge,
dvin decade) as well. Usage: [--steps N] [--sets base,more,...] (results file gets the set names)"""
import os
import sys

import numpy as np

from nets import DEV, RESULTS, MLP, KBNN, load, concat, subset, train, summary, regions, save

SEEDS = range(5)
H_MLP = [5, 10, 20, 40, 80, 160]
H_KB = [3, 5, 10, 20, 40, 80]
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 30000

base, va, te, ex = (load(n) for n in ("train", "val", "test", "extrap"))
guard = load("guard")
gdv = np.abs(guard["raw"][:, 0])
sets = {"base": base, "more": concat(base, load("train_ext")), "guard": concat(base, guard),
        "guard_z": concat(base, subset(guard, gdv <= 0.3, "guard_z")),
        "guard_04": concat(base, subset(guard, gdv <= 0.4, "guard_04"))}
if "--sets" in sys.argv:
    sets = {k: sets[k] for k in sys.argv[sys.argv.index("--sets") + 1].split(",")}
NAME = "exp_03_guard" + ("" if "--sets" not in sys.argv else "_" + "_".join(sets))
y_mean = tuple(base["y"].mean(0).tolist())
grid = lambda hs: ([h for h in hs for _ in SEEDS], [s for _ in hs for s in SEEDS])
(wm, sm), (wk, sk) = grid(H_MLP), grid(H_KB)
make = {"mlp_relu": lambda: MLP(wm, sm, y_mean=y_mean), "mlp_tanh": lambda: MLP(wm, sm, act="tanh", y_mean=y_mean),
        "kbnn_lin": lambda: KBNN(wk, wk, sk)}

rows, hist, models = [], {}, {}
print(f"Adam {STEPS} epochs; test |error| in %, mean over 5 seeds. 'edge max': worst t_d error within 5 % of a face"
      " of the box; '>100 mV max': worst t_d error at dvin 100-300 mV")
for fam in make:
    for tset, tr in sets.items():
        model = make[fam]().to(DEV)
        info = train(model, tr, va, steps=STEPS)
        key = f"{fam}_{tset}"
        hist[key], models[key] = info, model
        s = summary(model, [va, te, ex]) | {f"test_{k}": v for k, v in regions(model, te).items()}
        widths = np.array(model.config.get("widths") or model.config["widths_p"])
        print(f"\n== {key}: {len(tr['y'])} training rows, {info['seconds']:.0f} s")
        print(f"{'width':>5} | {'t_d avg':>7} {'max':>6} {'edge max':>8} {'>100 mV max':>11} | {'E avg':>6} {'max':>6} |"
              f" {'extrap t_d avg':>14} {'max':>6} | {'extrap E avg':>12} {'max':>6}")
        for h in sorted(set(widths)):
            i = widths == h
            f = lambda k: s[k][i].mean()
            print(f"{h:5d} | {f('test_td_avg'):7.3f} {f('test_td_max'):6.2f} {f('test_edge_0_0.05_td_max'):8.2f}"
                  f" {f('test_dv_0.1_0.31_td_max'):11.2f} | {f('test_E_avg'):6.3f} {f('test_E_max'):6.2f} |"
                  f" {f('extrap_td_avg'):14.3f} {f('extrap_td_max'):6.2f} | {f('extrap_E_avg'):12.3f} {f('extrap_E_max'):6.2f}")
        for m in range(len(widths)):
            rows.append(dict(family=fam, trainset=tset, width=int(widths[m]), seed=int(model.config["seeds"][m]),
                             params=int(model.n_params[m]), best_step=int(info["best_step"][m]),
                             train_s=float(info["seconds"]), **{k: float(v[m]) for k, v in s.items()}))
save(NAME, models, rows)
np.savez(os.path.join(RESULTS, f"{NAME}_hist.npz"),
         **{f"{k}_{n}": v for k, h in hist.items() for n, v in h.items() if n != "seconds"},
         **{f"{k}_widths": np.array(m.config.get("widths") or m.config["widths_p"]) for k, m in models.items()})
