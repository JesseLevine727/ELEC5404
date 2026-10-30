"""Experiment 06: final size of each family by the course's training procedure (Zhang et al.), replacing the
exp_04/exp_05 rule, which kept growing the ReLU networks into overlearning (validation error 3x the training error).
Target accuracy E_D = 0.5 % RMS relative error over (t_d, E), about 0.5 ps on a 100 ps delay, half the 1 ps budget
(PROJECT_PLAN.md section 5). The final size is the smallest network with good learning, i.e. training and
validation errors both <= E_D (mean over the 5 seeds); an undersized network underlearns (training error > E_D).
One hidden layer (the 3-layer MLP, the 4-layer KBNN); a second layer only if one layer cannot reach E_D.
Same recipe as exp_04 (base training set, Adam 60k epochs, best-validation weights). Test and extrap are not
evaluated here; final_eval.py does that once. Writes results/final.pt. Usage: [--steps N]"""
import os
import sys

import numpy as np
import torch

from nets import DEV, RESULTS, MLP, KBNN, load, pick, rel_mse, train, save

E_D = 0.5  # %
SEEDS = range(5)
SIZES = {"mlp_relu": [10, 15, 20, 25, 30, 35, 40, 50, 60, 80],
         "mlp_tanh": [5, 8, 10, 12, 15, 20, 25, 30, 40],
         "kbnn_lin": [5, 8, 10, 12, 15, 20, 25, 30, 40]}
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 60000

tr, va = load("train"), load("val")
y_mean = tuple(tr["y"].mean(0).tolist())
rows, hist, models, final = [], {}, {}, {}
for fam, hs in SIZES.items():
    w, s = [h for h in hs for _ in SEEDS], [k for _ in hs for k in SEEDS]
    model = (KBNN(w, w, s) if fam.startswith("kbnn") else
             MLP(w, s, act=fam.split("_")[1], y_mean=y_mean)).to(DEV)
    info = train(model, tr, va, steps=STEPS)
    hist[fam], models[fam] = info, model
    with torch.no_grad():  # RMS % errors of the best-validation weights
        e_tr, e_va = (100 * rel_mse(model(d), d["y"]).sqrt().cpu().numpy() for d in (tr, va))
    widths = np.array(w)
    mean = {h: (e_tr[widths == h].mean(), e_va[widths == h].mean()) for h in hs}
    good = [h for h in hs if max(mean[h]) <= E_D]
    size = min(good) if good else None
    print(f"\n== {fam}: {info['seconds']:.0f} s for {len(w)} networks, {STEPS} epochs. RMS % error, mean over seeds:")
    for h in hs:
        i = np.flatnonzero(widths == h)
        state = "good learning" if h in good else ("underlearning" if mean[h][0] > E_D else "overlearning")
        print(f"  size {h:3d} ({model.n_params[i[0]]:4d} weights): train {mean[h][0]:.3f}  val {mean[h][1]:.3f}"
              f" (seeds {e_va[i].min():.3f}-{e_va[i].max():.3f})  val/train {mean[h][1] / mean[h][0]:.2f}  {state}"
              + ("   <- chosen" if h == size else ""))
    for k in range(len(w)):
        rows.append(dict(family=fam, layers=1, width=int(w[k]), seed=int(s[k]), params=int(model.n_params[k]),
                         best_step=int(info["best_step"][k]), train_rms=float(e_tr[k]), val_rms=float(e_va[k]),
                         train_s=float(info["seconds"]), chosen=int(w[k] == size)))
    if size is None:
        print(f"  no size reaches E_D = {E_D} %: try a second hidden layer")
        continue
    idx = np.flatnonzero(widths == size).tolist()
    chosen = pick(model, idx)
    with torch.no_grad():  # the copy must reproduce the batch's predictions (up to float32 summation order)
        assert torch.allclose(chosen(va), model(va)[idx], rtol=1e-5, atol=0)
    final[fam] = dict(config=chosen.config, state=chosen.state_dict(), rep=int(np.argmin(e_va[idx])), size=size, layers=1)

save("exp_06_target", models, rows)
np.savez(os.path.join(RESULTS, "exp_06_target_hist.npz"),
         **{f"{k}_{n}": v for k, h in hist.items() for n, v in h.items() if n != "seconds"},
         **{f"{k}_widths": np.array(m.config.get("widths") or m.config["widths_p"]) for k, m in models.items()})
torch.save(final, os.path.join(RESULTS, "final.pt"))
print("\nFROZEN final models (results/final.pt): " + ", ".join(
    f"{fam} 1 hidden layer x {f['size']}" for fam, f in final.items()))
