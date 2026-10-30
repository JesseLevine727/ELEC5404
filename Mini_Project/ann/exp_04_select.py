"""Experiment 04: choose the final model of each family by validation error only (test and extrap untouched).
In exp_01 the largest sizes were still improving at 30k epochs, so the sweep is extended upward and trained
for 60k epochs; 5 seeds each, base training set, one hidden layer (the 3-layer MLP).
  mlp_relu  H = 80 160 320 640      mlp_tanh  H = 80 160 320      kbnn_lin  h = 40 80 160 320
Rule: the size with the lowest validation RMS error (mean over seeds), unless a smaller size is within 5 % of
it, then the smallest such. The 5 seeds of the chosen size are kept (results/final.pt); the representative
network is the seed with the lowest validation error. Usage: [--steps N]"""
import os
import sys

import numpy as np
import torch

from nets import DEV, RESULTS, MLP, KBNN, load, pick, train, summary, regions, save

SEEDS = range(5)
SIZES = {"mlp_relu": [80, 160, 320, 640], "mlp_tanh": [80, 160, 320], "kbnn_lin": [40, 80, 160, 320]}
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 60000
TOL = 0.05

tr, va, te, ex = (load(n) for n in ("train", "val", "test", "extrap"))
y_mean = tuple(tr["y"].mean(0).tolist())
rows, hist, models, final = [], {}, {}, {}
for fam, hs in SIZES.items():
    w, s = [h for h in hs for _ in SEEDS], [k for _ in hs for k in SEEDS]
    model = (KBNN(w, w, s) if fam.startswith("kbnn") else
             MLP(w, s, act=fam.split("_")[1], y_mean=y_mean)).to(DEV)
    info = train(model, tr, va, steps=STEPS)
    hist[fam], models[fam] = info, model
    vrms = 100 * np.sqrt(info["hist_val"].min(0))  # best validation RMS error of each network, %
    m = summary(model, [va, te, ex]) | {f"test_{k}": v for k, v in regions(model, te).items()}
    widths = np.array(w)
    mean = {h: vrms[widths == h].mean() for h in hs}
    best = min(mean.values())
    size = min(h for h in hs if mean[h] <= (1 + TOL) * best)
    idx = np.flatnonzero(widths == size)
    rep = int(idx[np.argmin(vrms[idx])])
    print(f"\n== {fam}: {info['seconds']:.0f} s for {len(w)} networks, {STEPS} epochs")
    for h in hs:
        i = widths == h
        print(f"  size {h:4d} ({model.n_params[np.flatnonzero(i)[0]]:5d} weights): val RMS {mean[h]:.3f} % "
              f"(seeds {vrms[i].min():.3f}-{vrms[i].max():.3f}), median best epoch {int(np.median(info['best_step'][i]))}"
              + ("   <- chosen" if h == size else ""))
    final[fam] = dict(size=size, seeds=[int(model.config["seeds"][k]) for k in idx], rep_seed=int(model.config["seeds"][rep]),
                      index=idx.tolist(), rep_index=rep)
    for k in range(len(w)):
        rows.append(dict(family=fam, width=int(w[k]), seed=int(s[k]), params=int(model.n_params[k]),
                         best_step=int(info["best_step"][k]), val_rms=float(vrms[k]), train_s=float(info["seconds"]),
                         chosen=int(k in idx), **{key: float(v[k]) for key, v in m.items()}))
save("exp_04_select", models, rows)
np.savez(os.path.join(RESULTS, "exp_04_select_hist.npz"),
         **{f"{k}_{n}": v for k, h in hist.items() for n, v in h.items() if n != "seconds"},
         **{f"{k}_widths": np.array(m.config.get("widths") or m.config["widths_p"]) for k, m in models.items()})
# the chosen networks, copied out of the batch (rebuild with nets.rebuild; "rep" = representative network)
out = {}
for fam, f in final.items():
    chosen = pick(models[fam], f["index"])
    with torch.no_grad():  # the copy must reproduce the batch's predictions (up to float32 summation order)
        assert torch.allclose(chosen(te), models[fam](te)[f["index"]], rtol=1e-5, atol=0)
    out[fam] = dict(config=chosen.config, state=chosen.state_dict(), rep=f["index"].index(f["rep_index"]), size=f["size"])
torch.save(out, os.path.join(RESULTS, "final.pt"))
print("\nfinal models (results/final.pt): " + ", ".join(f"{fam} size {f['size']} (representative seed {f['rep_seed']})"
                                                     for fam, f in final.items()))
