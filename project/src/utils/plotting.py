"""Plotting helpers for S-parameter datasets and model predictions."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def db(x: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.clip(np.abs(x), 1e-12, None))


def plot_sample_responses(freq: np.ndarray, s: np.ndarray, n: int = 6, seed: int = 0,
                          out: str | Path | None = None):
    """Plot |S21| and |S11| for a few random samples (s shape: (N,2,2,F))."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(s), size=min(n, len(s)), replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    fg = freq / 1e9
    for i in idx:
        axes[0].plot(fg, db(s[i, 1, 0]), lw=1.2)
        axes[1].plot(fg, db(s[i, 0, 0]), lw=1.2)
    for ax, t in zip(axes, ("$|S_{21}|$ (dB)", "$|S_{11}|$ (dB)")):
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
        ax.set_ylim(-60, 5)
    fig.suptitle("Sample geometry -> response pairs")
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig


def plot_prediction(freq: np.ndarray, s_true: np.ndarray, s_pred: np.ndarray,
                    out: str | Path | None = None, title: str = "Prediction"):
    """Compare a single true vs predicted response (s shape: (2,2,F))."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharex=True)
    fg = freq / 1e9
    axes[0].plot(fg, db(s_true[1, 0]), "k", label="true")
    axes[0].plot(fg, db(s_pred[1, 0]), "r--", label="pred")
    axes[1].plot(fg, db(s_true[0, 0]), "k", label="true")
    axes[1].plot(fg, db(s_pred[0, 0]), "r--", label="pred")
    for ax, t in zip(axes, ("$|S_{21}|$ (dB)", "$|S_{11}|$ (dB)")):
        ax.set_xlabel("Frequency (GHz)")
        ax.set_ylabel(t)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150)
    return fig
