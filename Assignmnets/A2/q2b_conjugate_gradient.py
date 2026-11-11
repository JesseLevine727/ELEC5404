"""
ELEC5404 Assignment 2 - Part 2(b): conjugate gradient with exact line minimization on

    E(w) = (w1 - 1)^2 + 2 (w2 - 2)^2,    w(0) = [0.5, 1.0]^T

Epoch 1 is identical to steepest descent (h0 = -g0). From epoch 2 on,
    h_k = -g_k + gamma_k * h_{k-1}
with Fletcher-Reeves gamma = (g_k . g_k) / (g_{k-1} . g_{k-1}).
Polak-Ribiere, gamma = g_k . (g_k - g_{k-1}) / (g_{k-1} . g_{k-1}), is also printed;
the two agree here because exact line search makes g_k perpendicular to g_{k-1}.

Exact fractions throughout so the numbers can be compared with hand working.
"""

from fractions import Fraction as F

H = [[F(2), F(0)], [F(0), F(4)]]
W_INIT = [F(1, 2), F(1)]


def E(w):
    return (w[0] - 1) ** 2 + 2 * (w[1] - 2) ** 2


def grad(w):
    return [2 * w[0] - 2, 4 * w[1] - 8]


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]


def Hv(v):
    return [H[0][0] * v[0] + H[0][1] * v[1], H[1][0] * v[0] + H[1][1] * v[1]]


def f4(x):
    return f"{float(x):.4f}"


def vec(v):
    return "[" + ", ".join(f4(x) for x in v) + "]"


def line_min(w, h):
    a = F(1, 2) * dot(h, Hv(h))     # E(w + eta h) = a eta^2 + b eta + c
    b = dot(grad(w), h)
    c = E(w)
    return a, b, c, -b / (2 * a)


w = list(W_INIT)
g_prev = h_prev = None
print(f"w(0) = {vec(w)}   E = {f4(E(w))}")

for k in range(1, 3):
    g = grad(w)
    print(f"\n================ Epoch {k} ================")
    print(f"  g = grad E(w({k-1})) = {vec(g)}   (exact {g})")
    if h_prev is None:
        h = [-g[0], -g[1]]
        print(f"  h = -g = {vec(h)}")
    else:
        gamma_fr = dot(g, g) / dot(g_prev, g_prev)
        gamma_pr = dot(g, [g[0] - g_prev[0], g[1] - g_prev[1]]) / dot(g_prev, g_prev)
        h = [-g[0] + gamma_fr * h_prev[0], -g[1] + gamma_fr * h_prev[1]]
        print(f"  -g = {vec([-g[0], -g[1]])}")
        print(f"  g.g = {f4(dot(g, g))},  g_prev.g_prev = {f4(dot(g_prev, g_prev))}")
        print(f"  gamma (Fletcher-Reeves) = {gamma_fr} = {float(gamma_fr):.6f}")
        print(f"  gamma (Polak-Ribiere)   = {gamma_pr} = {float(gamma_pr):.6f}"
              f"   (g.g_prev = {dot(g, g_prev)})")
        print(f"  gamma*h_prev = {vec([gamma_fr * h_prev[0], gamma_fr * h_prev[1]])}")
        print(f"  h = -g + gamma*h_prev = {vec(h)}   (direction ∝ [{h[0] / -h[1]:g}, -1])")
        print(f"  conjugacy check  h_prev' H h = {dot(h_prev, Hv(h))}   "
              f"(perpendicular? h_prev . h = {f4(dot(h_prev, h))})")
        to_min = [1 - w[0], 2 - w[1]]
        print(f"  direction from w({k-1}) to w* = {vec(to_min)}  "
              f"∝ [{to_min[0] / -to_min[1]:g}, -1]")
    a, b, c, eta = line_min(w, h)
    print(f"  E(eta) = {float(a):.4f} eta^2 + ({float(b):.4f}) eta + {float(c):.4f}")
    print(f"  eta* = {eta} = {float(eta):.4f}")
    for t in [F(0), F(1, 5), F(2, 5), F(1, 2), F(3, 5), F(4, 5)]:
        p = [w[0] + t * h[0], w[1] + t * h[1]]
        print(f"     eta = {float(t):.1f}:  w = {vec(p)}  E = {f4(E(p))}")
    w = [w[0] + eta * h[0], w[1] + eta * h[1]]
    print(f"  w({k}) = {vec(w)}   (exact {w[0]}, {w[1]})   E = {E(w)}")
    g_prev, h_prev = g, h

print(f"\nE reaches {E(w)} after 2 epochs -> CG converges in n = 2 epochs for this 2-weight quadratic.")
