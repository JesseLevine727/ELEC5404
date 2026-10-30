"""Experiment 05: depth check, the last tuning step; the models are frozen after it (PROJECT_PLAN.md section 5).
Two hidden layers (the 4-layer MLP; for the KBNN both blocks get two layers), trained as in exp_04
(base training set, Adam 60k epochs, 5 seeds):
  mlp_relu, mlp_tanh  H = 20 40 80 per layer        kbnn_lin  h = 10 20 40 per layer and block
Final choice per family, over exp_04 (one hidden layer) and these, by validation error only: the lowest
validation RMS error (mean over seeds), unless a model with fewer weights is within 5 % of it, then the one
with the fewest weights. Writes results/final.pt (replacing exp_04's). Usage: [--steps N]"""
import csv
import os
import sys

import numpy as np
import torch

from nets import DEV, RESULTS, MLP, KBNN, load, pick, rebuild, train, summary, regions, save

SEEDS = range(5)
SIZES = {"mlp_relu": [20, 40, 80], "mlp_tanh": [20, 40, 80], "kbnn_lin": [10, 20, 40]}
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 60000
TOL = 0.05

tr, va, te, ex = (load(n) for n in ("train", "val", "test", "extrap"))
y_mean = tuple(tr["y"].mean(0).tolist())
one = list(csv.DictReader(open(os.path.join(RESULTS, "exp_04_select.csv"))))
one_models = torch.load(os.path.join(RESULTS, "exp_04_select.pt"), weights_only=False)
rows, hist, models, final = [], {}, {}, {}
for fam, hs in SIZES.items():
    w, s = [h for h in hs for _ in SEEDS], [k for _ in hs for k in SEEDS]
    model = (KBNN(w, w, s, layers=2) if fam.startswith("kbnn") else
             MLP(w, s, layers=2, act=fam.split("_")[1], y_mean=y_mean)).to(DEV)
    info = train(model, tr, va, steps=STEPS)
    hist[fam], models[fam] = info, model
    vrms = 100 * np.sqrt(info["hist_val"].min(0))
    m = summary(model, [va, te, ex]) | {f"test_{k}": v for k, v in regions(model, te).items()}
    for k in range(len(w)):
        rows.append(dict(family=fam, layers=2, width=int(w[k]), seed=int(s[k]), params=int(model.n_params[k]),
                         best_step=int(info["best_step"][k]), val_rms=float(vrms[k]), train_s=float(info["seconds"]),
                         **{key: float(v[k]) for key, v in m.items()}))
    # candidates: (layers, width) -> network indices in their batch, validation RMS per network, weights
    cands = {}
    for h in hs:
        i = np.flatnonzero(np.array(w) == h)
        cands[(2, h)] = (i, vrms[i], model.n_params[i[0]])
    for h in sorted({int(r["width"]) for r in one if r["family"] == fam}):
        rr = [(k, r) for k, r in enumerate([r for r in one if r["family"] == fam]) if int(r["width"]) == h]
        cands[(1, h)] = (np.array([k for k, _ in rr]), np.array([float(r["val_rms"]) for _, r in rr]), int(rr[0][1]["params"]))
    best = min(v[1].mean() for v in cands.values())
    choice = min((c for c in cands if cands[c][1].mean() <= (1 + TOL) * best), key=lambda c: cands[c][2])
    print(f"\n== {fam}: {info['seconds']:.0f} s for the two-layer networks, {STEPS} epochs. Validation RMS error, mean over seeds:")
    for c in sorted(cands, key=lambda c: cands[c][2]):
        print(f"  {c[0]} hidden layer{'s' if c[0] > 1 else ' '} x {c[1]:4d} ({cands[c][2]:5d} weights): {cands[c][1].mean():.3f} %"
              f" (seeds {cands[c][1].min():.3f}-{cands[c][1].max():.3f})" + ("   <- chosen" if c == choice else ""))
    idx, v, _ = cands[choice]
    src = model if choice[0] == 2 else rebuild(one_models[fam])
    chosen = pick(src, [int(k) for k in idx])
    final[fam] = dict(config=chosen.config, state=chosen.state_dict(), rep=int(np.argmin(v)), size=choice[1], layers=choice[0])

save("exp_05_depth", models, rows)
np.savez(os.path.join(RESULTS, "exp_05_depth_hist.npz"),
         **{f"{k}_{n}": v for k, h in hist.items() for n, v in h.items() if n != "seconds"},
         **{f"{k}_widths": np.array(m.config.get("widths") or m.config["widths_p"]) for k, m in models.items()})
torch.save(final, os.path.join(RESULTS, "final.pt"))
print("\nFROZEN final models (results/final.pt): " + ", ".join(
    f"{fam} {f['layers']} hidden layer(s) x {f['size']}" for fam, f in final.items()))
