"""Error metrics for complex S-parameter predictions."""

from __future__ import annotations

import numpy as np


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.clip(np.abs(x), 1e-12, None))


def s_mse(s_pred: np.ndarray, s_true: np.ndarray) -> float:
    """Mean squared error over the six real/imag components."""
    diff = np.stack(
        [
            (s_pred[:, 0, 0] - s_true[:, 0, 0]).real,
            (s_pred[:, 0, 0] - s_true[:, 0, 0]).imag,
            (s_pred[:, 1, 0] - s_true[:, 1, 0]).real,
            (s_pred[:, 1, 0] - s_true[:, 1, 0]).imag,
            (s_pred[:, 1, 1] - s_true[:, 1, 1]).real,
            (s_pred[:, 1, 1] - s_true[:, 1, 1]).imag,
        ],
        axis=-1,
    )
    return float(np.mean(diff**2))


def mag_db_mae(s_pred: np.ndarray, s_true: np.ndarray, component=(1, 0)) -> float:
    """Mean absolute error in dB of |S_ij|."""
    i, j = component
    return float(np.mean(np.abs(db(s_pred[:, i, j]) - db(s_true[:, i, j]))))


def mag_mae(s_pred: np.ndarray, s_true: np.ndarray, component=(1, 0)) -> float:
    """Mean absolute error in linear magnitude of |S_ij|."""
    i, j = component
    return float(np.mean(np.abs(np.abs(s_pred[:, i, j]) - np.abs(s_true[:, i, j]))))


def passband_db_mae(s_pred: np.ndarray, s_true: np.ndarray, component=(1, 0),
                    threshold_db: float = -20.0) -> float:
    """dB MAE of |S_ij| restricted to points above ``threshold_db``.

    The deep stopband has |S21| near zero, so small absolute errors map to huge
    dB differences; restricting to the passband/transition gives a useful number.
    """
    i, j = component
    true_db = db(s_true[:, i, j])
    mask = true_db > threshold_db
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(db(s_pred[:, i, j])[mask] - true_db[mask])))


def summary(s_pred: np.ndarray, s_true: np.ndarray) -> dict:
    return {
        "s_mse": s_mse(s_pred, s_true),
        "s21_db_mae": mag_db_mae(s_pred, s_true, (1, 0)),
        "s21_db_mae_pb": passband_db_mae(s_pred, s_true, (1, 0)),
        "s21_mag_mae": mag_mae(s_pred, s_true, (1, 0)),
        "s11_db_mae": mag_db_mae(s_pred, s_true, (0, 0)),
    }
