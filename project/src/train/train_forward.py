"""Train a forward model (geometry, frequency) -> S-parameters.

Usage:
    python -m src.train.train_forward --model mlp --epochs 200
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.config import ROOT, load_config
from src.data.dataset import band_indices, load_dataset
from src.models.mlp import MLP
from src.utils import metrics
from src.utils.features import build_xy, pred_to_complex
from src.utils.plotting import plot_prediction


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, in_dim: int, hidden: tuple[int, ...]) -> nn.Module:
    if name == "mlp":
        return MLP(in_dim, 6, hidden)
    raise ValueError(f"unknown model {name!r}")


def evaluate(model, X, Y, n, f, s_true, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X).to(device)).cpu().numpy()
    s_pred = pred_to_complex(pred, n, f)
    return s_pred, metrics.summary(s_pred, s_true)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="mlp")
    ap.add_argument("--band", default="interp", choices=["interp", "full"])
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"{args.model}_{args.band}_seed{args.seed}"
    out_dir = ROOT / "results" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_dataset(cfg=cfg)
    Xtr, Ytr, _ = build_xy(data, cfg, "train", args.band)
    Xva, Yva, _ = build_xy(data, cfg, "val", args.band)
    n_te = len(data["geom_test"])

    print(f"device={device}  tag={tag}")
    print(f"train {Xtr.shape}  val {Xva.shape}")

    model = build_model(args.model, Xtr.shape[1], tuple(args.hidden)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model={args.model}  params={n_params:,}")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)
    loss_fn = nn.MSELoss()

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Ytr_t = torch.from_numpy(Ytr).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    Yva_t = torch.from_numpy(Yva).to(device)

    n = len(Xtr_t)
    best_val = float("inf")
    best_state = None
    wait = 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        total = 0.0
        for i in range(0, n, args.batch):
            idx = perm[i : i + args.batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), Ytr_t[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        train_loss = total / n

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xva_t), Yva_t).item()
        sched.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d}  train {train_loss:.3e}  val {val_loss:.3e}  lr {opt.param_groups[0]['lr']:.1e}")
        if wait >= args.patience:
            print(f"  early stop at epoch {epoch} (best val {best_val:.3e})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    dt = time.time() - t0

    # --- evaluation -------------------------------------------------------
    results = {"model": args.model, "band": args.band, "seed": args.seed,
               "params": n_params, "train_time_s": round(dt, 1),
               "best_val_mse": best_val, "hidden": list(args.hidden)}

    Xte_i, Yte_i, f_i = build_xy(data, cfg, "test", "interp")
    s_i = data["s_test"][:, :, :, band_indices(data, "interp")]
    s_pred_i, m_i = evaluate(model, Xte_i, Yte_i, n_te, len(f_i), s_i, device)
    results["test_interp"] = m_i

    Xte_e, Yte_e, f_e = build_xy(data, cfg, "test", "extrap")
    s_e = data["s_test"][:, :, :, band_indices(data, "extrap")]
    s_pred_e, m_e = evaluate(model, Xte_e, Yte_e, n_te, len(f_e), s_e, device)
    results["test_extrap"] = m_e

    print("test interpolation:", {k: round(v, 5) for k, v in m_i.items()})
    print("test extrapolation:", {k: round(v, 5) for k, v in m_e.items()})

    torch.save(model.state_dict(), out_dir / "model.pt")
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2)

    # prediction figure on a few test samples (full band)
    Xfull, _, f_full = build_xy(data, cfg, "test", "full")
    with torch.no_grad():
        pred_full = model(torch.from_numpy(Xfull).to(device)).cpu().numpy()
    s_full_true = data["s_test"]
    s_full_pred = pred_to_complex(pred_full, n_te, len(f_full))
    plot_prediction(f_full, s_full_true[0], s_full_pred[0],
                    out=out_dir / "prediction_sample0.png",
                    title=f"{args.model} (interp train) sample 0")
    print(f"saved {out_dir}/model.pt, metrics.json, prediction_sample0.png")


if __name__ == "__main__":
    main()
