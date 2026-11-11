"""
ELEC5404 Assignment 2 - Part 2(a): batch-mode backprop (steepest descent, zero momentum)
with exact line minimization, on

    E(w) = w1^2 - 2 w1 + 1 + 2 (w2^2 - 4 w2 + 4)  =  (w1 - 1)^2 + 2 (w2 - 2)^2

starting from w = [0.5, 1.0]^T, for two epochs.

All arithmetic is done with exact fractions so you can compare against hand working,
then each line-minimisation step size is cross-checked numerically, and the contour
plot (q2a_contour.png) is drawn so you can compare it with your hand-drawn version.
"""

from fractions import Fraction as F

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Problem
# ----------------------------------------------------------------------------
H = [[F(2), F(0)],      # Hessian of E (constant, since E is quadratic)
     [F(0), F(4)]]
W_INIT = [F(1, 2), F(1)]
W_STAR = [F(1), F(2)]   # minimum, E = 0
N_EPOCHS = 2


def E(w):
    return (w[0] - 1) ** 2 + 2 * (w[1] - 2) ** 2


def grad(w):
    return [2 * w[0] - 2, 4 * w[1] - 8]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def Hv(v):
    return [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]]


def fmt(v):
    return "[" + ", ".join(f"{x} (={float(x):.6f})" if x.denominator != 1 else f"{x}"
                           for x in v) + "]"


def fx(x):
    return f"{x}" if x.denominator == 1 else f"{x} = {float(x):.6f}"


# ----------------------------------------------------------------------------
# Training, step by step
# ----------------------------------------------------------------------------
def steepest_descent(w0, n_epochs):
    w = list(w0)
    path, grads, dirs = [list(w)], [], []
    print("E(w) = (w1-1)^2 + 2(w2-2)^2     grad E = [2w1 - 2, 4w2 - 8]     H = diag(2, 4)")
    print(f"Minimum at w* = {fmt(W_STAR)}, E(w*) = 0\n")
    print(f"Initial point w(0) = {fmt(w)},  E(w(0)) = {fx(E(w))}")

    for k in range(1, n_epochs + 1):
        g = grad(w)
        h = [-g[0], -g[1]]                       # BP with zero momentum: h = -grad E
        # line minimisation along h:  E(w + eta h) = c + b*eta + a*eta^2
        a = F(1, 2) * dot(h, Hv(h))
        b = dot(g, h)
        c = E(w)
        eta = -b / (2 * a)
        w_new = [w[0] + eta * h[0], w[1] + eta * h[1]]

        print(f"\n================ Epoch {k} ================")
        print(f"  grad E(w({k-1})) = [2*{w[0]} - 2, 4*{w[1]} - 8] = {fmt(g)}")
        print(f"  h = -grad E       = {fmt(h)}")
        print(f"  Line: w(eta) = w({k-1}) + eta*h = [{w[0]} + ({h[0]})eta,  {w[1]} + ({h[1]})eta]")
        print(f"  Substitute into E:  E(eta) = ({a}) eta^2 + ({b}) eta + {c}")
        print(f"  dE/deta = 2({a}) eta + ({b}) = 0   ->   eta* = {fx(eta)}")
        print(f"      (same as  g.g / g'Hg = {dot(g, g)} / {dot(g, Hv(g))} = {dot(g, g) / dot(g, Hv(g))})")
        print(f"  w({k}) = w({k-1}) + eta* h = {fmt(w_new)}")
        print(f"  E(w({k})) = {fx(E(w_new))}")
        g_new = grad(w_new)
        print(f"  Check: h . grad E(w({k})) = {dot(h, g_new)}  "
              f"(0 => new gradient is perpendicular to the line just searched,\n"
              f"         i.e. the line is tangent to the contour through w({k}))")

        # numeric cross-check of eta*: brute-force minimise E along the line
        etas = np.linspace(0, 2 * float(eta), 200001)
        wf, hf = np.array(w, float), np.array(h, float)
        pts = wf[None, :] + etas[:, None] * hf[None, :]
        Evals = (pts[:, 0] - 1) ** 2 + 2 * (pts[:, 1] - 2) ** 2
        print(f"  Numeric check: argmin over eta grid = {etas[Evals.argmin()]:.6f}  "
              f"(exact {float(eta):.6f})")

        grads.append(g)
        dirs.append(h)
        w = w_new
        path.append(list(w))
    return path, grads, dirs


# ----------------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def plot(path, grads, dirs, fname="q2a_contour.png"):
    P = np.array(path, float)
    Ek = [float(E(p)) for p in path]
    G = [np.array(g, float) / np.linalg.norm(np.array(g, float)) for g in grads]  # unit grads

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6.4), facecolor="#fcfcfb")

    def base(ax, xl, yl, bg, title):
        ax.set_facecolor("#fcfcfb")
        w1, w2 = np.meshgrid(np.linspace(*xl, 600), np.linspace(*yl, 600))
        Z = (w1 - 1) ** 2 + 2 * (w2 - 2) ** 2
        cs = ax.contour(w1, w2, Z, levels=bg, colors=GRID, linewidths=1)
        ax.clabel(cs, fmt=lambda v: f"{v:g}", fontsize=7, colors=INK2)
        # contours through each w(k): each line search ends tangent to one of these
        ax.contour(w1, w2, Z, levels=sorted(Ek), colors=INK2, linewidths=1.2, linestyles="--")
        ax.plot(P[:, 0], P[:, 1], color=BLUE, lw=2, zorder=3)
        ax.plot(1, 2, marker="*", ms=14, color=AQUA, mec="#fcfcfb", mew=1.5, zorder=6)
        ax.set_xlim(*xl); ax.set_ylim(*yl)
        ax.set_aspect("equal")          # essential: otherwise right angles look skewed
        ax.set_xlabel("w1", color=INK2); ax.set_ylabel("w2", color=INK2)
        ax.set_title(title, color=INK, fontsize=11, loc="left")
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)

    def arrow(ax, p, d, color):
        ax.annotate("", xy=p + d, xytext=p, zorder=5,
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=16))

    def point(ax, k, xytext, ha):
        ax.plot(*P[k], "o", ms=8, color=BLUE, mec="#fcfcfb", mew=2, zorder=6)
        ax.annotate(f"w({k}) = ({P[k,0]:.4f}, {P[k,1]:.4f})\nE = {Ek[k]:.4g}", P[k],
                    xytext=xytext, textcoords="offset points", fontsize=9, color=INK,
                    ha=ha, va="center", zorder=7)

    # ---- panel 1: epoch 1 -------------------------------------------------------
    base(ax1, (-0.7, 2.7), (0.2, 3.25), [0.25, 0.5, 1.0, 1.5, 3.0, 4.0],
         "Epoch 1 (full view)")
    L = 0.55
    arrow(ax1, P[0], L * G[0], ORANGE)
    arrow(ax1, P[0], -L * G[0], BLUE)
    ax1.text(*(P[0] + L * G[0] + [0.05, -0.04]), "∇E(w(0)) ∝ [−1, −4]", color=INK,
             fontsize=9, ha="left", va="top")
    ax1.text(*(P[0] - 0.6 * L * G[0] + [0.06, 0]), "h0 = −∇E ∝ [1, 4]", color=INK,
             fontsize=9, ha="left", va="center")
    point(ax1, 0, (12, -6), "left")
    point(ax1, 1, (-10, 14), "right")
    ax1.annotate("w*  (epoch 2 → zoom)", (1, 2), xytext=(1.45, 2.45), fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    # ---- panel 2: epoch 2 -------------------------------------------------------
    base(ax2, (0.66, 1.08), (1.86, 2.13), [0.0005, 0.005, 0.02], "Epoch 2 (zoom near w*)")
    L = 0.07
    arrow(ax2, P[1], L * G[1], ORANGE)
    arrow(ax2, P[1], -L * G[1], BLUE)
    ax2.text(*(P[1] + L * G[1] + [0, 0.008]), "∇E(w(1)) ∝ [−4, 1]", color=INK,
             fontsize=9, ha="center", va="bottom")
    ax2.text(*(P[1] - L * G[1] + [0, 0.012]), "h1 = −∇E ∝ [4, −1]", color=INK,
             fontsize=9, ha="left", va="bottom")
    ax2.text(0.715, 1.90, "h0 (arriving)", color=INK2, fontsize=8, ha="right", rotation=76)
    point(ax2, 1, (14, 16), "left")
    point(ax2, 2, (10, -22), "left")
    ax2.annotate("w* = (1, 2), E = 0", (1, 2), xytext=(1.03, 2.075), fontsize=9, color=INK,
                 ha="center", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8))

    # legend (identity is carried by label text too, not colour alone)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=ORANGE, lw=2.2, label="∇E (uphill, normal to contour)"),
               Line2D([], [], color=BLUE, lw=2.2, label="h = −∇E and line-search path"),
               Line2D([], [], color=INK2, lw=1.2, ls="--", label="contour through w(k)"),
               Line2D([], [], color=AQUA, marker="*", ls="", ms=12, label="minimum w*")]
    ax1.legend(handles=handles, loc="lower right", fontsize=8, frameon=True,
               facecolor="#fcfcfb", edgecolor=GRID)

    fig.suptitle("Part 2(a): steepest descent (BP, zero momentum) with line minimisation"
                 "   (arrows show direction only, not length)",
                 color=INK, fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(fname, dpi=160)
    print(f"\nSaved contour plot -> {fname}")


if __name__ == "__main__":
    path, grads, dirs = steepest_descent(W_INIT, N_EPOCHS)
    print("\nAngle check: h0 . h1 =", dot(dirs[0], dirs[1]),
          "(0 => consecutive steepest-descent steps are perpendicular: zig-zag)")
    plot(path, grads, dirs)
