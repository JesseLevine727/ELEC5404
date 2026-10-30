"""Final evaluation of the frozen models (results/final.pt from exp_05): run once, for the report's Table 3
(comparisons 1 and 3). |% error| on each split, mean +- standard deviation over the 5 seeds, and the
representative network (lowest validation error). Also the test errors by region. Writes results/final_eval.csv."""
import os

import numpy as np
import torch

from nets import RESULTS, load, rebuild, summary, regions

final = torch.load(os.path.join(RESULTS, "final.pt"), weights_only=False)
splits = [load(n) for n in ("train", "val", "test", "extrap")]
rows = []
for fam, f in final.items():
    model = rebuild(f)
    s = summary(model, splits) | {f"test_{k}": v for k, v in regions(model, splits[2]).items()}
    print(f"\n== {fam}: {f['layers']} hidden layer(s) x {f['size']}, {model.n_params[0]} weights; |% error| mean +- sd over"
          f" {len(model.config['seeds'])} seeds [representative network]")
    for split in ("train", "val", "test", "extrap"):
        print(f"  {split:6s} " + "   ".join(
            f"{o} avg {s[f'{split}_{o}_avg'].mean():.3f} +- {s[f'{split}_{o}_avg'].std():.3f} [{s[f'{split}_{o}_avg'][f['rep']]:.3f}]"
            f"  max {s[f'{split}_{o}_max'].mean():.2f} +- {s[f'{split}_{o}_max'].std():.2f} [{s[f'{split}_{o}_max'][f['rep']]:.2f}]"
            for o in ("td", "E")))
    for k in range(len(model.config["seeds"])):
        rows.append(dict(family=fam, layers=f["layers"], width=f["size"], seed=model.config["seeds"][k],
                         params=model.n_params[k], representative=int(k == f["rep"]), **{key: float(v[k]) for key, v in s.items()}))
with open(os.path.join(RESULTS, "final_eval.csv"), "w") as fh:
    fh.write(",".join(rows[0]) + "\n")
    for r in rows:
        fh.write(",".join(f"{v:.6g}" if isinstance(v, float) else str(v) for v in r.values()) + "\n")
