"""Shared code for the MLP / KBNN models of the StrongARM latch (PROJECT_PLAN.md section 5).

Data: cadence/latch/data/dataset/<set>.csv (column definitions in the README there).
Every model object is a batch of M independent networks trained together on the GPU, one batched matmul per
layer: seeds, sizes and training subsets are all just positions in the batch. Narrower networks are padded to
the widest one and their unused neurons masked off, so each trains exactly as if it were alone.

Units inside the models: t_d in 100 ps, E in 30 fJ (targets of order 1). Loss: relative MSE over both outputs.
"""
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.abspath(__file__))
DSET = os.path.join(ROOT, "..", "cadence", "latch", "data", "dataset")
RESULTS = os.path.join(ROOT, "results")
DV_LO, DV_HI = 1e-4, 0.3  # training range of |dvin| (V); fixes the log-input scaling
Z_LO, Z_HI = np.array([0.4, 0.9, 0.0]), np.array([0.6, 1.1, 85.0])  # vcm (V), vdd (V), temp (C)
T_UNIT, E_UNIT = 100.0, 30.0  # ps, fJ
V_REF = 1e-3  # log-law reference: L = ln(V_REF / |dvin|)
SOFTPLUS_1 = math.log(math.e - 1)  # softplus(SOFTPLUS_1) = 1

assert torch.cuda.is_available(), "training is meant to run on the GPU"
DEV = torch.device("cuda")


def load(name, n=None):
    """One dataset split as GPU tensors (first n rows if n is given):
    x  = (log10|dvin|, vcm, vdd, temp) scaled to [-1, 1] by the training ranges (the MLP input)
    zs = (vcm, vdd, temp) scaled, L = ln(V_REF / |dvin|), v = |dvin| / DV_HI, y = (t_d, E) in model units."""
    a = np.loadtxt(os.path.join(DSET, f"{name}.csv"), delimiter=",", skiprows=1, ndmin=2)[:n]
    assert (a[:, 8] == 1).all(), f"{name}: rows that failed the Cadence checks"
    dv, z = np.abs(a[:, 0]), a[:, 1:4]
    u = 2 * (np.log10(dv) - math.log10(DV_LO)) / (math.log10(DV_HI) - math.log10(DV_LO)) - 1
    zs = 2 * (z - Z_LO) / (Z_HI - Z_LO) - 1
    t = lambda c: torch.tensor(c, dtype=torch.float32, device=DEV)
    return dict(name=name, raw=a, x=t(np.column_stack([u, zs])), zs=t(zs), L=t(np.log(V_REF / dv)),
                v=t(dv / DV_HI), y=t(np.column_stack([a[:, 4] / T_UNIT, a[:, 5] / E_UNIT])))


def concat(*splits):
    """Several splits as one (e.g. train + guard)."""
    return dict(name="+".join(d["name"] for d in splits), raw=np.vstack([d["raw"] for d in splits]),
                **{k: torch.cat([d[k] for d in splits]) for k in ("x", "zs", "L", "v", "y")})


class BatchMLP(torch.nn.Module):
    """M independent fully connected networks n_in -> (width -> act) x layers -> n_out, evaluated together.
    widths[m] is model m's hidden width and seeds[m] its initialization seed."""

    def __init__(self, n_in, n_out, widths, seeds, layers=1, act="relu", out_scale=0.1):
        super().__init__()
        M, H = len(widths), max(widths)
        self.act = dict(relu=torch.relu, tanh=torch.tanh)[act]
        self.register_buffer("mask", (torch.arange(H)[None, :] < torch.tensor(widths)[:, None]).float()[:, None, :])
        gens = [torch.Generator().manual_seed(int(s)) for s in seeds]
        dims, fans = [n_in] + [H] * layers + [n_out], [[n_in] * M] + [list(widths)] * layers
        self.W, self.b = torch.nn.ParameterList(), torch.nn.ParameterList()
        for i in range(layers + 1):
            # He initialization of the hidden layers with each model's true fan-in; a small output layer, so
            # every model starts close to its output biases
            gain = (2.0 if act == "relu" else 1.0) if i < layers else out_scale ** 2
            W = torch.stack([torch.randn(dims[i], dims[i + 1], generator=g) * math.sqrt(gain / fans[i][m])
                             for m, g in enumerate(gens)])
            self.W.append(torch.nn.Parameter(W))
            self.b.append(torch.nn.Parameter(torch.zeros(M, 1, dims[i + 1])))
        self.n_params = [n_in * w + w + (layers - 1) * (w * w + w) + w * n_out + n_out for w in widths]

    def forward(self, x):
        h = x.expand(len(self.mask), *x.shape) if x.dim() == 2 else x
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            h = torch.baddbmm(b, h, W)
            if i < len(self.W) - 1:
                h = self.act(h) * self.mask
        return h


class MLP(torch.nn.Module):
    """Black box: (log10|dvin|, vcm, vdd, temp) -> (t_d, E). Output biases start at the training means."""

    def __init__(self, widths, seeds, layers=1, act="relu", y_mean=(1.2, 1.1)):
        super().__init__()
        self.config = dict(cls="MLP", widths=list(widths), seeds=list(seeds), layers=layers, act=act, y_mean=list(y_mean))
        self.net = BatchMLP(4, 2, widths, seeds, layers, act)
        with torch.no_grad():
            self.net.b[-1][:] = torch.tensor(y_mean)
        self.n_params = self.net.n_params

    def forward(self, d):
        return self.net(d["x"])


class KBNN(torch.nn.Module):
    """Knowledge-based model (PROJECT_PLAN.md section 5):
        t_d = t0(z) + tau(z) L + [g(v, z) - g(0, z)],    E = E0(z) + s(z) L + [q(v, z) - q(0, z)]
    z = (vcm, vdd, temp), L = ln(1 mV / |dvin|), v = |dvin| (linear, or its square root with corr="sqrt").
    The parameter block sees z only, so the slopes of t_d and E against L are exactly tau(z) and s(z) at every
    dvin; the correction block is zero at dvin = 0. Starts from the nominal law: t0 106 ps, tau 10 ps,
    E0 32 fJ, s 0.7 fJ per neper."""
    TAU0, S0 = 10 / T_UNIT, 0.7 / E_UNIT  # tau and s = these x softplus(network output)

    def __init__(self, widths_p, widths_c, seeds, layers=1, act="relu", corr="lin"):
        super().__init__()
        self.config = dict(cls="KBNN", widths_p=list(widths_p), widths_c=list(widths_c), seeds=list(seeds),
                           layers=layers, act=act, corr=corr)
        self.par = BatchMLP(3, 4, widths_p, seeds, layers, act)
        self.cor = BatchMLP(4, 2, widths_c, [s + 10_000 for s in seeds], layers, act)
        self.corr = corr
        with torch.no_grad():
            self.par.b[-1][:] = torch.tensor([106 / T_UNIT, SOFTPLUS_1, 32 / E_UNIT, SOFTPLUS_1])
        self.n_params = [a + b for a, b in zip(self.par.n_params, self.cor.n_params)]

    def parts(self, d):
        """The learned quantities, in model units: t0, tau, E0, s (functions of z) and the corrections g, q."""
        p = self.par(d["zs"])
        v = d["v"] if self.corr == "lin" else d["v"].sqrt()
        c = (self.cor(torch.cat([v[:, None], d["zs"]], 1))
             - self.cor(torch.cat([torch.zeros_like(v)[:, None], d["zs"]], 1)))
        return dict(t0=p[..., 0], tau=self.TAU0 * F.softplus(p[..., 1]), E0=p[..., 2],
                    s=self.S0 * F.softplus(p[..., 3]), g=c[..., 0], q=c[..., 1])

    def forward(self, d):
        k = self.parts(d)
        return torch.stack([k["t0"] + k["tau"] * d["L"] + k["g"], k["E0"] + k["s"] * d["L"] + k["q"]], -1)


def rel_mse(pred, y, w=None):
    """Relative MSE over both outputs, one value per model. w (M, N) weights the rows per model, e.g. to give
    each model its own training subset."""
    e = ((pred - y) / y).pow(2).mean(-1)
    return e.mean(-1) if w is None else (e * w).sum(-1) / w.sum(-1)


def train(model, tr, va, steps=30000, lr=3e-3, lr_end=3e-5, w=None, check=20):
    """Full-batch Adam with a cosine learning-rate decay. Early stopping: each model ends with the parameters
    of its best validation loss (checked every `check` steps). Returns timing, best steps and loss history."""
    params = list(model.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=lr_end)
    best = [p.detach().clone() for p in params]
    M = params[0].shape[0]
    best_val = torch.full((M,), float("inf"), device=DEV)
    best_step = torch.zeros(M, dtype=torch.long, device=DEV)
    hist_step, hist_tr, hist_va = [], [], []
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(steps):
        loss = rel_mse(model(tr), tr["y"], w)
        opt.zero_grad(set_to_none=True)
        loss.sum().backward()  # the models share no parameters, so each gets only its own gradient
        opt.step()
        sched.step()
        if step % check == 0 or step == steps - 1:
            with torch.no_grad():
                val = rel_mse(model(va), va["y"])
                better = val < best_val
                best_val = torch.where(better, val, best_val)
                best_step[better] = step
                for p, b in zip(params, best):
                    b[better] = p.detach()[better]
                hist_step.append(step)
                hist_tr.append(loss.detach())
                hist_va.append(val)
    with torch.no_grad():
        for p, b in zip(params, best):
            p.copy_(b)
    torch.cuda.synchronize()
    return dict(seconds=time.perf_counter() - t0, best_step=best_step.cpu().numpy(), hist_step=np.array(hist_step),
                hist_train=torch.stack(hist_tr).cpu().numpy(), hist_val=torch.stack(hist_va).cpu().numpy())


def _flat(tensors):
    """Per-model flat vectors (M, P) from a list of tensors with leading dimension M."""
    return torch.cat([t.reshape(t.shape[0], -1) for t in tensors], 1)


def _assign(params, flat):
    with torch.no_grad():
        i = 0
        for p in params:
            n = p[0].numel()
            p.copy_(flat[:, i:i + n].reshape(p.shape))
            i += n


def train_lbfgs(model, tr, va, iters=3000, mem=20, w=None, check=10, c1=1e-4, max_halvings=30):
    """Full-batch L-BFGS run separately for every model in the batch: each keeps its own curvature memory
    (the last `mem` steps), scaling and backtracking (Armijo) line search, so the models stay independent.
    Steps that show no positive curvature are left out of the memory; a model whose line search fails
    restarts from a short steepest-descent step. Early stopping and the returned record are as in train();
    one iteration is one gradient evaluation plus the line-search evaluations, counted in `evals`."""
    params = list(model.parameters())
    M = params[0].shape[0]

    def loss_grad():
        loss = rel_mse(model(tr), tr["y"], w)
        return loss.detach(), _flat(torch.autograd.grad(loss.sum(), params)).detach()

    x = _flat([p.detach() for p in params])
    f, g = loss_grad()
    S, Y, R = [], [], []
    gamma = torch.zeros(M, device=DEV)  # initial inverse-Hessian scale; 0 = no curvature information yet
    best, best_val = x.clone(), torch.full((M,), float("inf"), device=DEV)
    best_step = torch.zeros(M, dtype=torch.long, device=DEV)
    hist_step, hist_tr, hist_va, evals = [], [], [], 1
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for it in range(iters):
        # two-loop recursion with per-model dot products
        q, alphas = g.clone(), []
        for s, y, rho in zip(reversed(S), reversed(Y), reversed(R)):
            a = rho * (s * q).sum(1)
            q -= a[:, None] * y
            alphas.append(a)
        r = torch.where(gamma > 0, gamma, 1e-2 / g.norm(dim=1).clamp_min(1e-30))[:, None] * q
        for s, y, rho, a in zip(S, Y, R, reversed(alphas)):
            r += (a - rho * (y * r).sum(1))[:, None] * s
        d = -r
        gd = (g * d).sum(1)
        uphill = gd >= 0
        d[uphill] = -g[uphill]
        gd = (g * d).sum(1)
        # backtracking line search, per model
        step, searching = torch.ones(M, device=DEV), torch.ones(M, dtype=torch.bool, device=DEV)
        for _ in range(max_halvings):
            _assign(params, x + step[:, None] * d)
            with torch.no_grad():
                fn = rel_mse(model(tr), tr["y"], w)
            evals += 1
            searching &= ~(fn <= f + c1 * step * gd)
            if not searching.any():
                break
            step = torch.where(searching, step / 2, step)
        step = torch.where(searching, torch.zeros_like(step), step)
        x_new = x + step[:, None] * d
        _assign(params, x_new)
        f, g_new = loss_grad()
        evals += 1
        s, y = x_new - x, g_new - g
        sy = (s * y).sum(1)
        ok = sy > 1e-10 * s.norm(dim=1) * y.norm(dim=1)
        S.append(s), Y.append(y), R.append(torch.where(ok, 1 / sy.clamp_min(1e-30), torch.zeros_like(sy)))
        gamma = torch.where(ok, sy / (y * y).sum(1).clamp_min(1e-30), gamma)
        if searching.any():  # failed line search: forget this model's curvature memory
            for rho in R:
                rho[searching] = 0
            gamma[searching] = 0
        if len(S) > mem:
            S.pop(0), Y.pop(0), R.pop(0)
        x, g = x_new, g_new
        if it % check == 0 or it == iters - 1:
            with torch.no_grad():
                val = rel_mse(model(va), va["y"])
            better = val < best_val
            best_val = torch.where(better, val, best_val)
            best_step[better] = it
            best[better] = x[better]
            hist_step.append(it), hist_tr.append(f), hist_va.append(val)
    _assign(params, best)
    torch.cuda.synchronize()
    return dict(seconds=time.perf_counter() - t0, best_step=best_step.cpu().numpy(), evals=evals,
                hist_step=np.array(hist_step), hist_train=torch.stack(hist_tr).cpu().numpy(),
                hist_val=torch.stack(hist_va).cpu().numpy())


@torch.no_grad()
def pct_errors(model, d):
    """Percent errors (M, N, 2) of t_d and E on one split."""
    return (100 * (model(d) - d["y"]) / d["y"]).cpu().numpy()


def summary(model, splits):
    """Per model (arrays of length M): average and worst |% error| of t_d and E on each split."""
    out = {}
    for d in splits:
        e = np.abs(pct_errors(model, d))
        for k, o in enumerate(("td", "E")):
            out[f"{d['name']}_{o}_avg"] = e[:, :, k].mean(1)
            out[f"{d['name']}_{o}_max"] = e[:, :, k].max(1)
    return out


DV_BANDS = [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1), (1e-1, 0.31)]
EDGE_BANDS = [(0.0, 0.05), (0.05, 0.2), (0.2, 1.01)]


def regions(model, d):
    """Per model: average and worst |% error| of t_d and E by dvin decade (keys dv_<lo>_<hi>_...) and by distance
    from the edge of the input box (keys edge_<lo>_<hi>_..., scaled units: 0 = on a face, 1 = the centre)."""
    e = np.abs(pct_errors(model, d))
    dv, edge = np.abs(d["raw"][:, 0]), 1 - np.abs(d["x"].cpu().numpy()).max(1)
    out = {}
    for tag, key, bands in (("dv", dv, DV_BANDS), ("edge", edge, EDGE_BANDS)):
        for lo, hi in bands:
            sel = (key >= lo) & (key < hi)
            for k, o in enumerate(("td", "E")):
                out[f"{tag}_{lo:g}_{hi:g}_{o}_avg"] = e[:, sel, k].mean(1)
                out[f"{tag}_{lo:g}_{hi:g}_{o}_max"] = e[:, sel, k].max(1)
    return out


def save(name, models, rows):
    """results/<name>.csv (one row per model) and results/<name>.pt (configs and weights, see rebuild)."""
    os.makedirs(RESULTS, exist_ok=True)
    keys = list(rows[0])
    with open(os.path.join(RESULTS, f"{name}.csv"), "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k]) for k in keys) + "\n")
    torch.save({k: dict(config=m.config, state=m.state_dict()) for k, m in models.items()},
               os.path.join(RESULTS, f"{name}.pt"))


def rebuild(saved):
    """A trained model from one entry of a saved results/<name>.pt."""
    c = dict(saved["config"])
    model = dict(MLP=MLP, KBNN=KBNN)[c.pop("cls")](**c).to(DEV)
    model.load_state_dict(saved["state"])
    return model
