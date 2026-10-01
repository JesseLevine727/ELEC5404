"""Feature/target construction shared by all forward models.

Representation
--------------
A pointwise model takes ``x = [geom_norm (d), f_norm (1)]`` and predicts the six
independent real numbers of the reciprocal 2-port at that frequency:

    y = [Re S11, Im S11, Re S21, Im S21, Re S22, Im S22]

Frequency is normalised by the *full* band (f_start..f_stop) so the interpolation
band occupies [0, 0.75] and the extrapolation band (0.75, 1.0].
"""

from __future__ import annotations

import numpy as np

from src.data.dataset import band_indices, geometry_bounds, normalize_geom

S_COMPONENTS = ((0, 0), (1, 0), (1, 1))  # S11, S21, S22


def _targets(s: np.ndarray) -> np.ndarray:
    """(N,2,2,F) complex -> (N,F,6) real [Re/Im of S11,S21,S22]."""
    parts = []
    for i, j in S_COMPONENTS:
        parts.append(s[:, i, j].real)
        parts.append(s[:, i, j].imag)
    return np.stack(parts, axis=-1)


def build_xy(data: dict, cfg: dict, split: str, band: str = "interp"):
    """Return pointwise (X, Y, freq) for a split/band.

    X    : (N*F, d+1)
    Y    : (N*F, 6)
    freq : (F,) actual frequencies (Hz) for this band
    """
    lo, hi = geometry_bounds(cfg)
    f_all = data["freq"]
    fmin, fmax = f_all[0], f_all[-1]
    idx = band_indices(data, band)
    freq = f_all[idx]

    geom = data[f"geom_{split}"]
    s = data[f"s_{split}"][:, :, :, idx]

    n, d = geom.shape
    f = len(freq)
    g = normalize_geom(geom, lo, hi)                      # (N, d)
    fn = (freq - fmin) / (fmax - fmin)                    # (F,)

    Xg = np.repeat(g, f, axis=0)                          # (N*F, d)
    Xf = np.tile(fn, n).reshape(-1, 1)                    # (N*F, 1)
    X = np.concatenate([Xg, Xf], axis=1)

    Y = _targets(s).reshape(n * f, 6)
    return X.astype(np.float32), Y.astype(np.float32), freq


def pred_to_complex(pred: np.ndarray, n: int, f: int) -> np.ndarray:
    """(N*F, 6) prediction -> (N, 2, 2, F) complex, with S12 = S21.

    Samples are assumed geometry-major then frequency, matching build_xy.
    """
    p = pred.reshape(n, f, 6)
    s = np.zeros((n, 2, 2, f), dtype=complex)
    s[:, 0, 0] = p[:, :, 0] + 1j * p[:, :, 1]
    s[:, 1, 0] = p[:, :, 2] + 1j * p[:, :, 3]
    s[:, 0, 1] = s[:, 1, 0]
    s[:, 1, 1] = p[:, :, 4] + 1j * p[:, :, 5]
    return s

