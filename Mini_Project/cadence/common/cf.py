"""Shared helpers for the ELEC3509 GPDK180 series: psfascii reader, measurements, plot style."""
import os
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SERIES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VDD = 1.8


def project(name):
    """Return (raw, data, plots) directories of a project, creating data/plots."""
    root = os.path.join(SERIES, name)
    dirs = [os.path.join(root, d) for d in ("raw", "data", "plots")]
    for d in dirs[1:]:
        os.makedirs(d, exist_ok=True)
    return dirs


# Reference palette (dataviz skill): slot 1 blue = NMOS, slot 2 orange = PMOS, slot 3 aqua
NMOS, PMOS, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def read_psf(path):
    """Read a psfascii analysis file (dc/tran/ac magnitude-free) -> dict of numpy arrays."""
    with open(path) as f:
        lines = f.read().splitlines()
    start = lines.index("VALUE") + 1
    data = {}
    # sweeps: "name" value ; single op points: "name" "units" value [PROP(...)]
    pat = re.compile(r'^"([^"]+)"\s+(?:"[^"]*"\s+)?([-+0-9.eE]+|nan|inf)')
    for ln in lines[start:]:
        if ln == "END":
            break
        m = pat.match(ln)
        if m:
            data.setdefault(m.group(1), []).append(float(m.group(2)))
    return {k: np.asarray(v) for k, v in data.items()}


def crossing(t, y, level, rising):
    """Linear-interpolated x of the first crossing of `level`."""
    s = np.sign(y - level)
    idx = np.where((s[:-1] < 0) & (s[1:] >= 0))[0] if rising else \
        np.where((s[:-1] > 0) & (s[1:] <= 0))[0]
    if len(idx) == 0:
        return np.nan
    i = idx[0]
    return t[i] + (level - y[i]) * (t[i + 1] - t[i]) / (y[i + 1] - y[i])


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "axes.edgecolor": INK2,
        "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "lines.linewidth": 2,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "legend.frameon": False, "figure.dpi": 130,
    })


def label_point(ax, x, y, text, color, dx=6, dy=0, ha="left"):
    """Marker plus a direct label in ink (not series color)."""
    ax.plot([x], [y], "o", ms=5, color=color)
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points",
                va="center", ha=ha, fontsize=8, color=INK2)


def read_ocnprint(path):
    """Read an OCEAN ocnPrint table -> (column names, 2-D array)."""
    with open(path) as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    header = re.findall(r'(\S+\([^)]*\)|\S+)', lines[0])
    rows = [[float(v) for v in ln.split()] for ln in lines[1:]]
    return header, np.asarray(rows)
