"""Experiment 08: training and evaluation cost of the frozen architectures (comparison 5, Fig 7, Table 4).
Training: each architecture is retrained from scratch on the base training set (Adam 60k epochs, as in exp_06),
once as a single network (seed 0) and once as the usual batch of 5 seeds, on the GPU. Recorded: wall time, and the
epochs at which the validation RMS error first reaches 1 % and E_d = 0.5 %. Evaluation: one representative
network of results/final.pt, timed for a single prediction (CPU) and for large batches (CPU and GPU).
Writes results/exp_08_cost.csv and results/exp_08_cost_hist.npz. Usage: [--steps N]"""
import os
import sys
import time

import numpy as np
import torch

from nets import DEV, RESULTS, MLP, KBNN, load, pick, rebuild, train

SEEDS = list(range(5))
TARGETS = (1.0, 0.5)  # validation RMS %, the second is E_d
STEPS = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 60000
N_BATCH = 100_000

tr, va, te = load("train"), load("val"), load("test")
final = torch.load(os.path.join(RESULTS, "final.pt"), weights_only=False)


def build(c, seeds):
    w = [c["widths"][0] if c["cls"] == "MLP" else c["widths_p"][0]] * len(seeds)
    if c["cls"] == "KBNN":
        return KBNN(w, w, seeds, layers=c["layers"], act=c["act"], corr=c["corr"]).to(DEV)
    return MLP(w, seeds, layers=c["layers"], act=c["act"], y_mean=c["y_mean"]).to(DEV)


def to(d, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in d.items()}


def tile(d, n):
    """n rows made by repeating a split, for throughput timing."""
    reps = -(-n // len(d["y"]))
    return {k: (v.repeat(reps, *[1] * (v.dim() - 1))[:n] if torch.is_tensor(v) else v) for k, v in d.items() if k != "raw"}


def timeit(fn, sync, n_min=20, t_min=1.0):
    """Median seconds per call of fn, over at least n_min calls and t_min seconds."""
    fn(); sync()
    times, start = [], time.perf_counter()
    while len(times) < n_min or time.perf_counter() - start < t_min:
        t0 = time.perf_counter(); fn(); sync(); times.append(time.perf_counter() - t0)
    return float(np.median(times))


rows, hist = [], {}
for fam, f in final.items():
    row = dict(family=fam, params=int(rebuild(f).n_params[0]))
    for tag, seeds in (("single", [0]), ("batch5", SEEDS)):
        model = build(f["config"], seeds)
        info = train(model, tr, va, steps=STEPS)
        vrms = 100 * np.sqrt(info["hist_val"])  # (checks, networks)
        hist[f"{fam}_{tag}_step"], hist[f"{fam}_{tag}_val_rms"] = info["hist_step"], vrms
        hist[f"{fam}_{tag}_train_rms"] = 100 * np.sqrt(info["hist_train"])
        row[f"{tag}_train_s"] = info["seconds"]
        for t in TARGETS:  # first check at which each network reaches the target (NaN if never)
            first = [info["hist_step"][np.argmax(v <= t)] if (v <= t).any() else np.nan for v in vrms.T]
            row[f"{tag}_epochs_to_{t:g}pct"] = float(np.nanmedian(first)) if not np.isnan(first).all() else np.nan
            row[f"{tag}_reached_{t:g}pct"] = int(np.sum(~np.isnan(first)))
        row[f"{tag}_best_val_rms"] = float(vrms.min(0).mean())
        print(f"{fam:8s} {tag:6s}: {info['seconds']:5.1f} s for {STEPS} epochs; epochs to val RMS 1 % / 0.5 %: "
              f"{row[f'{tag}_epochs_to_1pct']:.0f} / {row[f'{tag}_epochs_to_0.5pct']:.0f} (median over networks, "
              f"{row[f'{tag}_reached_0.5pct']}/{len(seeds)} reach 0.5 %); best val RMS {row[f'{tag}_best_val_rms']:.3f} %")
    # evaluation time of the representative network
    net = pick(rebuild(f), [f["rep"]])
    with torch.no_grad():
        one = to({k: (v[:1] if torch.is_tensor(v) else v) for k, v in te.items() if k != "raw"}, "cpu")
        big = tile(te, N_BATCH)
        big_cpu = to(big, "cpu")
        net_cpu = pick(rebuild(f), [f["rep"]]).cpu()
        nosync = lambda: None
        row["cpu_single_s"] = timeit(lambda: net_cpu(one), nosync, n_min=2000)
        row["cpu_batch_s_per_point"] = timeit(lambda: net_cpu(big_cpu), nosync) / N_BATCH
        row["gpu_batch_s_per_point"] = timeit(lambda: net(big), torch.cuda.synchronize) / N_BATCH
    print(f"{fam:8s} eval: CPU single {1e6 * row['cpu_single_s']:.1f} us; per point in batches of {N_BATCH}: "
          f"CPU {1e9 * row['cpu_batch_s_per_point']:.1f} ns, GPU {1e9 * row['gpu_batch_s_per_point']:.2f} ns")
    rows.append(row)

with open(os.path.join(RESULTS, "exp_08_cost.csv"), "w") as fh:
    fh.write(",".join(rows[0]) + "\n")
    for r in rows:
        fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
np.savez(os.path.join(RESULTS, "exp_08_cost_hist.npz"), **hist)
