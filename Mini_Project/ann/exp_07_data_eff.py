"""Experiment 07: data efficiency (comparison 4, Fig 6). The frozen architectures (results/final.pt) are retrained
from scratch on the first N rows of the training set, N = 50 ... 2800. The training set is scrambled Sobol in order,
so every prefix is an evenly spread design. Same recipe as exp_06 (Adam 60k epochs, 5 seeds, best-validation
weights); early stopping uses the full validation set at every N. Test and extrap errors are measured for each
network; nothing is chosen from them. Writes results/exp_07_data_eff.csv. Usage: [--steps N]"""
import os
import sys

import numpy as np
import torch

from nets import DEV, RESULTS, T_UNIT, MLP, KBNN, load, rel_mse, train, summary, save

NS = [50, 100, 200, 400, 800, 1600, 2800]
SEEDS = range(5)
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 60000

tr, va, te, ex = (load(n) for n in ("train", "val", "test", "extrap"))
final = torch.load(os.path.join(RESULTS, "final.pt"), weights_only=False)
rows, models = [], {}
for fam, f in final.items():
    n, s = [n for n in NS for _ in SEEDS], [k for _ in NS for k in SEEDS]
    w = [f["size"]] * len(n)
    c = f["config"]
    model = (KBNN(w, w, s, layers=c["layers"], act=c["act"], corr=c["corr"]) if c["cls"] == "KBNN" else
             MLP(w, s, layers=c["layers"], act=c["act"], y_mean=c["y_mean"])).to(DEV)
    mask = (torch.arange(len(tr["y"]), device=DEV)[None, :] < torch.tensor(n, device=DEV)[:, None]).float()
    if c["cls"] == "MLP":  # output biases start at each network's own training mean
        with torch.no_grad():
            for k, nk in enumerate(n):
                model.net.b[-1][k] = tr["y"][:nk].mean(0)
    info = train(model, tr, va, steps=STEPS, w=mask)
    models[fam] = model
    with torch.no_grad():
        e_tr = 100 * rel_mse(model(tr), tr["y"], mask).sqrt().cpu().numpy()  # on each network's own N rows
        e_va = 100 * rel_mse(model(va), va["y"]).sqrt().cpu().numpy()
        ps = {d["name"]: np.abs((model(d) - d["y"])[:, :, 0].cpu().numpy()) * T_UNIT for d in (te, ex)}
    m = summary(model, [te, ex])
    print(f"\n== {fam} ({f['size']} neurons, {model.n_params[0]} weights): {info['seconds']:.0f} s. Mean over seeds:")
    print("      N  train RMS  val RMS   test t_d avg/max %   extrap t_d avg/max %   extrap t_d avg ps   extrap E avg/max %")
    for nk in NS:
        i = np.array(n) == nk
        print(f"  {nk:5d}  {e_tr[i].mean():9.3f}  {e_va[i].mean():7.3f}   {m['test_td_avg'][i].mean():8.3f} / {m['test_td_max'][i].mean():5.2f}"
              f"      {m['extrap_td_avg'][i].mean():8.3f} / {m['extrap_td_max'][i].mean():5.2f}"
              f"      {ps['extrap'][i].mean():8.2f}      {m['extrap_E_avg'][i].mean():8.3f} / {m['extrap_E_max'][i].mean():5.2f}")
    for k in range(len(n)):
        rows.append(dict(family=fam, n_train=n[k], seed=s[k], params=int(model.n_params[k]), best_step=int(info["best_step"][k]),
                         train_rms=float(e_tr[k]), val_rms=float(e_va[k]), train_s=float(info["seconds"]),
                         test_td_avg_ps=float(ps["test"][k].mean()), test_td_max_ps=float(ps["test"][k].max()),
                         extrap_td_avg_ps=float(ps["extrap"][k].mean()), extrap_td_max_ps=float(ps["extrap"][k].max()),
                         **{key: float(v[k]) for key, v in m.items()}))
save("exp_07_data_eff", models, rows)
