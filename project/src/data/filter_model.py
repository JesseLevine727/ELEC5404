"""Analytic coupled-resonator bandpass filter model.

This is the *coarse* model used to generate training data for the neural
surrogates.  A capacitively-coupled parallel-LC resonator ladder is built from
a Chebyshev low-pass prototype, and a physically-motivated geometry -> circuit
mapping turns microstrip-like dimensions (resonator lengths, coupling gaps,
feed gaps, line width) into component values.

Topology (N resonators):

    port1 -- C_in --[ L||C res1 ]-- C12 --[ res2 ]-- ... -- C_out -- port2

Each series branch is a coupling capacitor (a J-inverter); each shunt branch is
a parallel LC resonator.  S-parameters are obtained by cascading ABCD matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import skrf as rf

C0 = 299_792_458.0


def chebyshev_g(n: int, ripple_db: float = 0.5) -> np.ndarray:
    """Return Chebyshev low-pass prototype g-values g0 .. g_{n+1}."""
    beta = np.log(1.0 / np.tanh(ripple_db / 17.37))
    gamma = np.sinh(beta / (2.0 * n))
    a = np.array([np.sin((2 * k - 1) * np.pi / (2 * n)) for k in range(1, n + 1)])
    b = np.array([gamma**2 + np.sin(k * np.pi / n) ** 2 for k in range(1, n + 1)])

    g = np.zeros(n + 2)
    g[0] = 1.0
    g[1] = 2.0 * a[0] / gamma
    for k in range(2, n + 1):
        g[k] = 4.0 * a[k - 1] * a[k - 2] / (b[k - 2] * g[k - 1])
    if n % 2 == 1:
        g[n + 1] = 1.0
    else:
        g[n + 1] = 1.0 / np.tanh(beta / 4.0) ** 2
    return g


@dataclass
class FilterSpec:
    order: int = 3
    f0_hz: float = 2.0e9
    fbw: float = 0.10
    ripple_db: float = 0.5
    z0: float = 50.0
    cr0: float = 1.0e-12
    l0_m: float = 10.0e-3
    w0_m: float = 1.0e-3
    k_max: float = 0.15
    s_decay: float = 0.5e-3
    qe0: float = 16.0
    s_ref: float = 0.3e-3
    s_q: float = 0.5e-3


class CoupledResonatorFilter:
    """Geometry-parameterised capacitively-coupled bandpass filter."""

    def __init__(self, spec: FilterSpec):
        self.spec = spec
        self.g = chebyshev_g(spec.order, spec.ripple_db)
        self.omega0 = 2.0 * np.pi * spec.f0_hz
        self._build_nominal()

    # -- design helpers ----------------------------------------------------
    def _build_nominal(self) -> None:
        """Nominal coupling coefficients and external Qs from the prototype."""
        n = self.spec.order
        g = self.g
        self.k_nom = self.spec.fbw / np.sqrt(g[1 : n + 1] * g[2 : n + 2])
        self.qe_nom_in = g[0] * g[1] / self.spec.fbw
        self.qe_nom_out = g[n] * g[n + 1] / self.spec.fbw

    @property
    def n_geom(self) -> int:
        """Geometry dimension: N lengths + (N-1) gaps + 2 feeds + 1 width."""
        return 2 * self.spec.order + 2

    def sample_geometry(self, n: int, rng: np.random.Generator, ranges: dict) -> np.ndarray:
        """Uniform / Latin-hypercube samples of geometry vectors (metres)."""
        n = int(n)
        dims = self.n_geom
        lo = np.array(
            [ranges["length_mm"][0]] * self.spec.order
            + [ranges["gap_mm"][0]] * (self.spec.order - 1)
            + [ranges["feed_mm"][0]] * 2
            + [ranges["width_mm"][0]],
            dtype=float,
        ) * 1e-3
        hi = np.array(
            [ranges["length_mm"][1]] * self.spec.order
            + [ranges["gap_mm"][1]] * (self.spec.order - 1)
            + [ranges["feed_mm"][1]] * 2
            + [ranges["width_mm"][1]],
            dtype=float,
        ) * 1e-3

        if ranges.get("sampling", "uniform") == "lhs":
            # simple Latin hypercube: random permutation per dimension
            u = np.empty((n, dims))
            for d in range(dims):
                u[:, d] = (rng.permutation(n) + rng.random(n)) / n
        else:
            u = rng.random((n, dims))
        return lo + u * (hi - lo)

    # -- geometry -> circuit ----------------------------------------------
    def geometry_to_params(self, geom: np.ndarray) -> dict:
        s = self.spec
        n = s.order
        lengths = geom[:n]
        gaps = geom[n : n + (n - 1)]
        feeds = geom[n + (n - 1) : n + (n - 1) + 2]
        width = geom[-1]

        cr_base = s.cr0 * (width / s.w0_m)

        # resonator detuning: length up -> resonance down
        f_i = s.f0_hz * (s.l0_m / lengths)
        omega_i = 2.0 * np.pi * f_i
        cr = np.full(n, cr_base)
        lr = 1.0 / (omega_i**2 * cr)

        # inter-resonator coupling capacitors
        k = s.k_max * np.exp(-gaps / s.s_decay)
        cc = k * cr_base

        # external coupling capacitors from target Qe
        qe_in = s.qe0 * np.exp((feeds[0] - s.s_ref) / s.s_q)
        qe_out = s.qe0 * np.exp((feeds[1] - s.s_ref) / s.s_q)
        c_in = np.sqrt(cr_base / (s.z0 * qe_in * self.omega0))
        c_out = np.sqrt(cr_base / (s.z0 * qe_out * self.omega0))

        return {
            "lr": lr, "cr": cr, "cc": cc,
            "c_in": c_in, "c_out": c_out,
            "k": k, "qe_in": qe_in, "qe_out": qe_out,
            "f_i": f_i, "cr_base": cr_base,
        }

    # -- circuit -> S-parameters ------------------------------------------
    def s_params(self, geom: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        p = self.geometry_to_params(geom)
        return self._ladder_s(p, freqs)

    def _ladder_s(self, p: dict, freqs: np.ndarray) -> np.ndarray:
        w = 2.0 * np.pi * np.asarray(freqs, dtype=float)
        z0 = self.spec.z0
        n = self.spec.order

        A = np.ones_like(w)
        B = np.zeros_like(w)
        Cm = np.zeros_like(w)
        D = np.ones_like(w)

        def cascade(A1, B1, C1, D1):
            nonlocal A, B, Cm, D
            A2, B2, C2, D2 = A, B, Cm, D
            A = A1 * A2 + B1 * C2
            B = A1 * B2 + B1 * D2
            Cm = C1 * A2 + D1 * C2
            D = C1 * B2 + D1 * D2

        def series_cap(c):
            z = 1.0 / (1j * w * c)
            return np.ones_like(w), z, np.zeros_like(w), np.ones_like(w)

        def shunt_res(lr, cr):
            y = 1j * w * cr + 1.0 / (1j * w * lr)
            return np.ones_like(w), np.zeros_like(w), y, np.ones_like(w)

        cascade(*series_cap(p["c_in"]))
        for i in range(n):
            cascade(*shunt_res(p["lr"][i], p["cr"][i]))
            if i < n - 1:
                cascade(*series_cap(p["cc"][i]))
        cascade(*series_cap(p["c_out"]))

        denom = A + B / z0 + Cm * z0 + D
        s11 = (A + B / z0 - Cm * z0 - D) / denom
        s21 = 2.0 / denom
        s12 = 2.0 * (A * D - B * Cm) / denom
        s22 = (-A + B / z0 - Cm * z0 + D) / denom
        return np.stack([np.stack([s11, s12]), np.stack([s21, s22])], axis=0)

    # -- dataset helper ----------------------------------------------------
    def batch_s_params(self, geoms: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """Return S with shape (n_samples, 2, 2, n_freq)."""
        return np.stack([self.s_params(g, freqs) for g in geoms], axis=0)


def s_to_touchstone_like(freqs: np.ndarray, s: np.ndarray) -> rf.Network:
    """Wrap a (2,2,F) S array as a scikit-rf Network (convenience)."""
    f = rf.Frequency.from_f(freqs, unit="hz")
    return rf.Network(frequency=f, s=s.transpose(2, 0, 1), z0=50.0)
