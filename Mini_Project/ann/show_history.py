"""Training history of a saved experiment: relative RMS error (%) on the training and validation sets vs epoch
(one full-batch step = one epoch), mean over the seeds of each width.  Usage: show_history.py [experiment]"""
import os
import sys

import numpy as np

from nets import RESULTS

name = sys.argv[1] if len(sys.argv) > 1 else "exp_01_sizes"
h = np.load(os.path.join(RESULTS, f"{name}_hist.npz"))
fams = sorted({k.rsplit("_", 2)[0] for k in h.files if k.endswith("_hist_step")})
EPOCHS = [0, 100, 300, 1000, 3000, 10000, 20000, 29999]
for fam in fams:
    step, w = h[f"{fam}_hist_step"], h[f"{fam}_widths"]
    rows = [np.searchsorted(step, e) for e in EPOCHS]
    print(f"\n== {fam}: RMS relative error (%) train / val, at epoch")
    print(f"{'width':>5} | " + " | ".join(f"{e:>13d}" for e in EPOCHS) + " | best epoch (median)")
    for wd in sorted(set(w)):
        i = np.flatnonzero(w == wd)
        rms = lambda k: 100 * np.sqrt(h[f"{fam}_hist_{k}"][:, i]).mean(1)
        tr, va = rms("train"), rms("val")
        print(f"{wd:5d} | " + " | ".join(f"{tr[r]:6.3f}/{va[r]:6.3f}" for r in rows)
              + f" | {int(np.median(h[f'{fam}_best_step'][i]))}")
