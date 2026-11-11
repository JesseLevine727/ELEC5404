"""
ELEC5404 Assignment 2 - Part 1: one epoch of batch-mode backpropagation, by hand.

Network (Figure A1):  1 input -> 2 sigmoid hidden -> 1 linear output
    gamma_k = w_k0 + w_k1 * x            (hidden pre-activation, k = 1, 2)
    z_k     = sigmoid(gamma_k)
    y       = v0 + v1*z1 + v2*z2         (linear output)

Error:  E(w) = 1/2 * sum_i (y(x_i, w) - d_i)^2
Update: w_new = w_old - eta * dE/dw      (batch mode: gradient summed over all samples
                                          first, ONE update per epoch, no momentum)

The script prints every intermediate quantity in the same order you would compute it
by hand, then cross-checks the analytic gradient two independent ways:
  1. central finite differences on E(w)
  2. PyTorch autograd (skipped if torch isn't installed)
"""

import math

# ----------------------------------------------------------------------------
# Problem data
# ----------------------------------------------------------------------------
ETA = 0.1
DATA = [(0.0, 2.0),   # (x_i, d_i), i = 1
        (1.0, 4.0)]   #             i = 2

W0 = {  # initial weights from Figure A1
    "v0": 0.0, "v1": 4.0, "v2": -1.0,
    "w10": 0.0, "w11": 2.0,
    "w20": 1.0, "w21": 0.1,
}
ORDER = ["v0", "v1", "v2", "w10", "w11", "w20", "w21"]


def sigmoid(g):
    return 1.0 / (1.0 + math.exp(-g))


def forward(w, x):
    g1 = w["w10"] + w["w11"] * x
    g2 = w["w20"] + w["w21"] * x
    z1, z2 = sigmoid(g1), sigmoid(g2)
    y = w["v0"] + w["v1"] * z1 + w["v2"] * z2
    return g1, g2, z1, z2, y


def error(w):
    return 0.5 * sum((forward(w, x)[-1] - d) ** 2 for x, d in DATA)


# ----------------------------------------------------------------------------
# Step-by-step epoch
# ----------------------------------------------------------------------------
def one_epoch(w, verbose=True):
    p = print if verbose else (lambda *a, **k: None)
    grad = {k: 0.0 for k in ORDER}
    E = 0.0

    for i, (x, d) in enumerate(DATA, start=1):
        g1, g2, z1, z2, y = forward(w, x)
        e = y - d                      # dE_i/dy
        E += 0.5 * e ** 2

        # output-layer derivatives (y is linear, so dy/dv0 = 1, dy/dvk = zk)
        dv0, dv1, dv2 = e * 1.0, e * z1, e * z2

        # hidden-layer local gradients: delta_k = e * v_k * z_k (1 - z_k)
        s1, s2 = z1 * (1 - z1), z2 * (1 - z2)       # sigmoid'(gamma) = z(1-z)
        delta1 = e * w["v1"] * s1
        delta2 = e * w["v2"] * s2
        dw10, dw11 = delta1 * 1.0, delta1 * x + 0.0   # +0.0 avoids printing -0
        dw20, dw21 = delta2 * 1.0, delta2 * x + 0.0

        p(f"\n--- Sample {i}: x = {x:g}, d = {d:g} ---")
        p("  Forward pass")
        p(f"    gamma1 = w10 + w11*x = {w['w10']:g} + {w['w11']:g}*{x:g} = {g1:.6f}")
        p(f"    gamma2 = w20 + w21*x = {w['w20']:g} + {w['w21']:g}*{x:g} = {g2:.6f}")
        p(f"    z1 = sigmoid({g1:.4f}) = {z1:.6f}")
        p(f"    z2 = sigmoid({g2:.4f}) = {z2:.6f}")
        p(f"    y  = v0 + v1*z1 + v2*z2 = {w['v0']:g} + {w['v1']:g}*{z1:.6f} + ({w['v2']:g})*{z2:.6f} = {y:.6f}")
        p(f"    e  = y - d = {y:.6f} - {d:g} = {e:.6f}")
        p(f"    E_{i} = 1/2 e^2 = {0.5 * e**2:.6f}")
        p("  Backward pass (this sample's contribution)")
        p(f"    dE/dv0  = e          = {dv0:+.6f}")
        p(f"    dE/dv1  = e*z1       = {dv1:+.6f}")
        p(f"    dE/dv2  = e*z2       = {dv2:+.6f}")
        p(f"    z1(1-z1) = {s1:.6f},  z2(1-z2) = {s2:.6f}")
        p(f"    delta1 = e*v1*z1(1-z1) = {e:.6f}*{w['v1']:g}*{s1:.6f} = {delta1:+.6f}")
        p(f"    delta2 = e*v2*z2(1-z2) = {e:.6f}*({w['v2']:g})*{s2:.6f} = {delta2:+.6f}")
        p(f"    dE/dw10 = delta1     = {dw10:+.6f}")
        p(f"    dE/dw11 = delta1*x   = {dw11:+.6f}")
        p(f"    dE/dw20 = delta2     = {dw20:+.6f}")
        p(f"    dE/dw21 = delta2*x   = {dw21:+.6f}")

        for k, val in zip(ORDER, [dv0, dv1, dv2, dw10, dw11, dw20, dw21]):
            grad[k] += val

    w_new = {k: w[k] - ETA * grad[k] for k in ORDER}

    p(f"\n=== Total training error  E(w_old) = {E:.6f} ===")
    p("\n=== Batch gradient (sum over samples) and weight update, eta = %g ===" % ETA)
    p(f"  {'weight':>6} | {'w_old':>9} | {'dE/dw':>10} | {'-eta*dE/dw':>10} | {'w_new':>9}")
    p("  " + "-" * 56)
    for k in ORDER:
        p(f"  {k:>6} | {w[k]:9.5f} | {grad[k]:+10.6f} | {-ETA * grad[k]:+10.6f} | {w_new[k]:9.5f}")
    return E, grad, w_new


# ----------------------------------------------------------------------------
# Independent checks
# ----------------------------------------------------------------------------
def finite_diff_grad(w, h=1e-6):
    g = {}
    for k in ORDER:
        wp, wm = dict(w), dict(w)
        wp[k] += h
        wm[k] -= h
        g[k] = (error(wp) - error(wm)) / (2 * h)
    return g


def torch_grad(w):
    try:
        import torch
    except ImportError:
        return None
    t = {k: torch.tensor(v, dtype=torch.float64, requires_grad=True) for k, v in w.items()}
    x = torch.tensor([d[0] for d in DATA], dtype=torch.float64)
    d = torch.tensor([d[1] for d in DATA], dtype=torch.float64)
    z1 = torch.sigmoid(t["w10"] + t["w11"] * x)
    z2 = torch.sigmoid(t["w20"] + t["w21"] * x)
    y = t["v0"] + t["v1"] * z1 + t["v2"] * z2
    E = 0.5 * ((y - d) ** 2).sum()
    E.backward()
    return {k: t[k].grad.item() for k in ORDER}


if __name__ == "__main__":
    E, grad, w_new = one_epoch(W0)

    print("\n=== Verification of dE/dw ===")
    fd = finite_diff_grad(W0)
    tg = torch_grad(W0)
    hdr = f"  {'weight':>6} | {'analytic':>11} | {'finite diff':>11}"
    hdr += f" | {'autograd':>11}" if tg else ""
    print(hdr)
    for k in ORDER:
        row = f"  {k:>6} | {grad[k]:+11.7f} | {fd[k]:+11.7f}"
        row += f" | {tg[k]:+11.7f}" if tg else ""
        print(row)
    max_err = max(abs(grad[k] - fd[k]) for k in ORDER)
    print(f"  max |analytic - finite diff| = {max_err:.2e}  ->  {'PASS' if max_err < 1e-6 else 'FAIL'}")
    if tg is None:
        print("  (torch not installed - autograd check skipped)")

    print(f"\nSanity check: E(w_old) = {E:.6f}  ->  E(w_new) = {error(w_new):.6f}"
          f"  ({'decreased' if error(w_new) < E else 'did NOT decrease'})")
