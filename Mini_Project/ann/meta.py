"""SAR ADC metastability error rate from a comparator delay model (PROJECT_PLAN.md section 7).

ADC (decided 2026-10-30): 10-bit asynchronous SAR. After sampling (SAMPLE of the period) the N comparisons run back
to back, each followed by a fixed overhead T_OH (latch reset, SAR logic, DAC settling). A conversion fails if they
do not fit in the window W = (1 - SAMPLE) / fs. Full scale 2*VDD differential (references = supply), so the
largest residue is VDD; VCM = VDD/2; the input is uniform over the full scale. Noise and offset do not change the
rate to first order: they move where the input sits, not how often it lands within dV_meta of a threshold.

Rate by the threshold sum (exact for a deterministic delay model when one comparison is stuck):
  For each of the 2^N - 1 comparison thresholds j (comparison k_j), the conversion fails when the input is within
  dV_meta_j of it, where t_d(dV_meta_j) = W - N T_OH - (sum of the other N - 1 delays at that threshold).
  rate = sum_j 2 dV_meta_j / V_FS.
  With the KBNN near metastability, t_d = t0 + tau ln(V_REF / dV), so dV_meta_j = V_REF exp(-(t_left_j - t0) / tau).
Cross-check: brute-force Monte Carlo of whole conversions on the GPU (mc_rate).
Delays above 0.3 V (the training range) are clamped for the models: t_d(0.3 V) slightly overestimates them."""
import math

import numpy as np
import torch

from nets import DEV, DV_LO, DV_HI, Z_LO, Z_HI, V_REF, T_UNIT

N_BITS, SAMPLE, T_OH = 10, 0.25, 200.0  # bits, sampling fraction of the period, overhead per comparison (ps)
DV_TRAIN_MAX = 0.3
GRID = np.logspace(-40, math.log10(1.2), 6001)  # dvin grid of the delay tables (V)


def condition(vdd, temp):
    """(vcm, vdd, temp) and the full scale for a supply: VCM = VDD/2, V_FS = 2 VDD."""
    return np.array([vdd / 2, vdd, temp]), 2 * vdd


def window(fs):
    """Time left for the N decisions (ps) at sampling rate fs (Hz)."""
    return (1 - SAMPLE) / np.asarray(fs) * 1e12 - N_BITS * T_OH


def thresholds(n=N_BITS):
    """The 2^n - 1 comparison thresholds: comparison index k (1..n) and the residue magnitudes of all n comparisons
    (units of the full scale) when the input sits exactly on the threshold of comparison k (its residue is 0).
    Comparisons before k see |D_k - D_m|; those after k see 2^-m (the input sits on a code boundary)."""
    ks, rs = [], []

    def walk(k, D, prev):
        rs.append([abs(D - p) for p in prev] + [0.0] + [2.0 ** -m for m in range(k + 1, n + 1)])
        ks.append(k)
        if k < n:
            for s in (1, -1):
                walk(k + 1, D + s * 2.0 ** -(k + 1), prev + [D])

    walk(1, 0.0, [])
    return np.array(ks), np.array(rs)


KS, RS = thresholds()


class Table:
    """A delay curve t_d(dvin) in ps on GRID, from a model or from Spectre points; evaluates and inverts it."""

    def __init__(self, td):
        self.td = np.asarray(td, float)
        # monotone envelope (largest delay at any input >= dvin), so the failure set {t_d > t} is dvin < dV_meta
        env = np.maximum.accumulate(self.td[::-1])[::-1]
        self.x = env[::-1] + 1e-9 * np.arange(len(env))  # increasing as dvin decreases
        self.y = np.log(GRID[::-1])

    def __call__(self, dv):
        return np.interp(np.log(np.maximum(dv, GRID[0])), np.log(GRID), self.td)

    def dv_meta(self, t):
        """Largest dvin whose delay exceeds t (GRID[0] if none down to 1e-40 V)."""
        return np.exp(np.interp(t, self.x, self.y, left=self.y[0], right=self.y[-1]))


def model_tables(model, z):
    """One Table per network of a batched model at condition z (inputs above 0.3 V clamped)."""
    from nets import features
    dv = np.minimum(GRID, DV_TRAIN_MAX)
    with torch.no_grad():
        t = model(features(dv, np.tile(z, (len(dv), 1)))).cpu().numpy()[:, :, 0] * T_UNIT
    return [Table(row) for row in t]


def spectre_table(dv, td):
    """Table from Spectre points (dvin ascending): monotone cubic in ln dvin; below the smallest point the local
    log-law slope of the lowest decade continues the curve (marked by the caller)."""
    from scipy.interpolate import PchipInterpolator
    ln = np.log(dv)
    f = PchipInterpolator(ln, td, extrapolate=False)
    g = np.log(GRID)
    out = f(np.clip(g, ln[0], ln[-1]))
    low = g < ln[0]
    lo = ln < ln[0] + math.log(10) + 1e-9
    slope = np.polyfit(ln[lo], td[lo], 1)[0]  # ps per neper (negative)
    out[low] = td[0] + slope * (g[low] - ln[0])
    return Table(out)


def rate(table, fs, vfs, rest_table=None):
    """Metastability error rate per conversion by the threshold sum, for each fs (Hz). With rest_table, the other
    N - 1 delays come from it and only the stuck comparison from table (isolates the extrapolation)."""
    r = RS * vfs
    t = (rest_table or table)(np.where(r > 0, r, 1.0))
    rest = np.where(RS > 0, t, 0.0).sum(1)  # the other N - 1 delays at each threshold (ps)
    left = window(fs)[:, None] - rest[None, :]  # time the stuck comparison may take
    dvm = np.minimum(table.dv_meta(left), vfs * 2.0 ** -KS[None, :])
    return np.minimum(2 * dvm.sum(1) / vfs, 1.0)


def rate_kbnn_closed(t0, tau, rest, fs, vfs):
    """The KBNN's closed form: dV_meta_j = V_REF exp(-(t_left_j - t0) / tau); rest = the other delays per threshold."""
    left = window(fs)[:, None] - rest[None, :]
    return np.minimum(2 * V_REF * np.exp(-(left - t0) / tau).sum(1) / vfs, 1.0)


def fs_at(fs, r, target):
    """Sampling rate at which a rate curve (increasing in fs) crosses target (NaN if it never does)."""
    lr = np.log10(np.maximum(r, 1e-300))
    i = np.flatnonzero(lr >= math.log10(target))
    if not len(i) or i[0] == 0:
        return np.nan
    a, b = i[0] - 1, i[0]
    return float(np.exp(np.interp(math.log10(target), [lr[a], lr[b]], [math.log(fs[a]), math.log(fs[b])])))


def _features(dv, z):
    """nets.features on the GPU for float64 residues (a float32 input would lose residues below ~0.1 uV)."""
    n = len(dv)
    u = 2 * (torch.log10(dv) - math.log10(DV_LO)) / (math.log10(DV_HI) - math.log10(DV_LO)) - 1
    zs = torch.tensor(2 * (z - Z_LO) / (Z_HI - Z_LO) - 1, dtype=torch.float32, device=DEV).expand(n, 3)
    return dict(x=torch.cat([u.float()[:, None], zs], 1), zs=zs, L=torch.log(V_REF / dv).float(),
                v=(dv / DV_HI).float())


def mc_rate(net, z, vfs, fs, conversions, chunk=500_000, seed=0):
    """Brute force: the fraction of `conversions` uniform random inputs whose N delays (one network, GPU) do not fit
    in the window, for each fs. Returns (rates, failures, seconds)."""
    import time
    gen = torch.Generator(device=DEV).manual_seed(seed)
    need = torch.tensor(np.sort(window(fs)), device=DEV)  # ascending
    count = torch.zeros(len(need) + 1, dtype=torch.long, device=DEV)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    done = 0
    with torch.no_grad():
        while done < conversions:
            m = min(chunk, conversions - done)
            v = (torch.rand(m, generator=gen, device=DEV, dtype=torch.float64) - 0.5) * vfs
            D = torch.zeros_like(v)
            res = []
            for k in range(1, N_BITS + 1):
                r = v - D
                res.append(r.abs())
                D = D + torch.where(r > 0, 1.0, -1.0) * vfs * 2.0 ** -(k + 1)
            dv = torch.stack(res, 1).reshape(-1).clamp(min=1e-300, max=DV_TRAIN_MAX)
            total = (net(_features(dv, z))[0, :, 0].double() * T_UNIT).reshape(m, N_BITS).sum(1)
            count += torch.bincount(torch.searchsorted(need, total), minlength=len(need) + 1)
            done += m
    torch.cuda.synchronize()
    seconds = time.perf_counter() - t0
    # a conversion fails at window w if its total exceeds w: failures(w_i) = number of totals above need[i]
    above = torch.flip(torch.cumsum(torch.flip(count, [0]), 0), [0])[1:].cpu().numpy()
    order = np.argsort(window(fs))
    fails = np.empty(len(fs), dtype=np.int64)
    fails[order] = above
    return fails / conversions, fails, seconds
