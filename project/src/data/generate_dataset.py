"""Generate the geometry -> S-parameter dataset from the analytic filter model.

Usage:
    python -m src.data.generate_dataset [--config config.yaml] [--n-train 4000]

Saves ``data/dataset.npz`` with complex S-parameters for train/val/test splits.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.config import ROOT, load_config
from src.data.filter_model import CoupledResonatorFilter, FilterSpec


def build_filter(cfg: dict) -> CoupledResonatorFilter:
    fc = cfg["filter"]
    spec = FilterSpec(
        order=int(fc["order"]),
        f0_hz=float(fc["f0_ghz"]) * 1e9,
        fbw=float(fc["fbw"]),
        ripple_db=float(fc["ripple_db"]),
        z0=float(fc["z0"]),
        cr0=float(fc["cr0_pf"]) * 1e-12,
        l0_m=float(fc["l0_mm"]) * 1e-3,
        w0_m=float(fc["w0_mm"]) * 1e-3,
        k_max=float(fc["k_max"]),
        s_decay=float(fc["s_decay_mm"]) * 1e-3,
        qe0=float(fc["qe0"]),
        s_ref=float(fc["s_ref_mm"]) * 1e-3,
        s_q=float(fc["s_q_mm"]) * 1e-3,
    )
    return CoupledResonatorFilter(spec)


def make_freqs(cfg: dict) -> np.ndarray:
    ff = cfg["frequency"]
    return np.linspace(float(ff["f_start_ghz"]) * 1e9, float(ff["f_stop_ghz"]) * 1e9, int(ff["n_points"]))


def generate(cfg: dict, n_train: int | None = None, n_val: int | None = None,
             n_test: int | None = None, seed: int | None = None) -> dict:
    ds = cfg["dataset"]
    seed = cfg["seed"] if seed is None else seed
    n_train = ds["n_train"] if n_train is None else n_train
    n_val = ds["n_val"] if n_val is None else n_val
    n_test = ds["n_test"] if n_test is None else n_test

    filt = build_filter(cfg)
    freqs = make_freqs(cfg)
    split_ghz = float(cfg["frequency"].get("extrap_split_ghz", freqs[-1] / 1e9))
    extrap_split_idx = int(np.searchsorted(freqs, split_ghz * 1e9, side="right"))
    ranges = dict(cfg["geometry"])
    ranges["sampling"] = ds.get("sampling", "uniform")

    rng = np.random.default_rng(seed)

    def split(n):
        if n <= 0:
            return np.empty((0, filt.n_geom)), np.empty((0, 2, 2, len(freqs)), dtype=complex)
        geom = filt.sample_geometry(n, rng, ranges)
        s = filt.batch_s_params(geom, freqs)
        return geom, s

    geom_tr, s_tr = split(n_train)
    geom_va, s_va = split(n_val)
    geom_te, s_te = split(n_test)

    return {
        "freq": freqs,
        "extrap_split_idx": np.array([extrap_split_idx]),
        "geom_train": geom_tr, "s_train": s_tr,
        "geom_val": geom_va, "s_val": s_va,
        "geom_test": geom_te, "s_test": s_te,
        "order": np.array([filt.spec.order]),
        "n_geom": np.array([filt.n_geom]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--n-train", type=int, default=None)
    ap.add_argument("--n-val", type=int, default=None)
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = generate(cfg, args.n_train, args.n_val, args.n_test, args.seed)

    out = Path(args.out) if args.out else ROOT / cfg["dataset"]["out_dir"] / "dataset.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **data)

    print(f"saved {out}")
    print(f"  order      : {int(data['order'][0])}  (geometry dim {int(data['n_geom'][0])})")
    print(f"  freq points: {len(data['freq'])}  ({data['freq'][0]/1e9:.2f}-{data['freq'][-1]/1e9:.2f} GHz)")
    k = int(data["extrap_split_idx"][0])
    print(f"  interpolation band: {data['freq'][0]/1e9:.2f}-{data['freq'][k-1]/1e9:.2f} GHz ({k} pts)")
    print(f"  extrapolation band: {data['freq'][k]/1e9:.2f}-{data['freq'][-1]/1e9:.2f} GHz ({len(data['freq'])-k} pts)")
    print(f"  train/val/test: {len(data['geom_train'])}/{len(data['geom_val'])}/{len(data['geom_test'])}")


if __name__ == "__main__":
    main()
