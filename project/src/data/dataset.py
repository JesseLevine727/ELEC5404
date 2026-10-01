"""Dataset loading, frequency-band splits, and geometry normalisation.

The dataset stores the full-band complex S-parameters.  The interpolation /
extrapolation split is applied *here* (by indexing frequency), so a model can
be trained on the low band and evaluated on the high band it never saw.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.config import ROOT, load_config


def load_dataset(path: str | Path | None = None, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    path = Path(path) if path else ROOT / cfg["dataset"]["out_dir"] / "dataset.npz"
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def band_indices(data: dict, band: str = "interp") -> np.ndarray:
    """Frequency indices for the 'interp', 'extrap', or 'full' band."""
    k = int(data["extrap_split_idx"][0])
    n = len(data["freq"])
    if band == "interp":
        return np.arange(0, k)
    if band == "extrap":
        return np.arange(k, n)
    if band == "full":
        return np.arange(0, n)
    raise ValueError(f"unknown band {band!r}")


def get_arrays(data: dict, split: str = "train", band: str = "interp"):
    """Return (geom, s, freq_idx) for a split and frequency band.

    geom : (N, n_geom)
    s    : (N, 2, 2, F_band) complex
    """
    geom = data[f"geom_{split}"]
    s = data[f"s_{split}"]
    idx = band_indices(data, band)
    return geom, s[:, :, :, idx], idx


def geometry_bounds(cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Min/max geometry vectors (metres) in the model's geometry ordering."""
    g = cfg["geometry"]
    n = int(cfg["filter"]["order"])
    lo = np.array(
        [g["length_mm"][0]] * n + [g["gap_mm"][0]] * (n - 1)
        + [g["feed_mm"][0]] * 2 + [g["width_mm"][0]]
    ) * 1e-3
    hi = np.array(
        [g["length_mm"][1]] * n + [g["gap_mm"][1]] * (n - 1)
        + [g["feed_mm"][1]] * 2 + [g["width_mm"][1]]
    ) * 1e-3
    return lo, hi


def normalize_geom(geom: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return (geom - lo) / (hi - lo)


def denormalize_geom(geom: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return geom * (hi - lo) + lo


def main() -> None:
    cfg = load_config()
    data = load_dataset(cfg=cfg)
    freq = data["freq"]
    k = int(data["extrap_split_idx"][0])
    print("dataset keys:", sorted(data.keys()))
    print(f"freq: {freq[0]/1e9:.2f}-{freq[-1]/1e9:.2f} GHz, {len(freq)} pts")
    print(f"interp band: {freq[0]/1e9:.2f}-{freq[k-1]/1e9:.2f} GHz ({k} pts)")
    print(f"extrap band: {freq[k]/1e9:.2f}-{freq[-1]/1e9:.2f} GHz ({len(freq)-k} pts)")
    for split in ("train", "val", "test"):
        g, s, _ = get_arrays(data, split, "full")
        print(f"  {split:5s}: geom {g.shape}, s {s.shape}")
    lo, hi = geometry_bounds(cfg)
    print("geometry bounds (mm):")
    print("  lo", np.round(lo * 1e3, 2))
    print("  hi", np.round(hi * 1e3, 2))


if __name__ == "__main__":
    main()
