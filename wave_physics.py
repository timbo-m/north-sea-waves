"""Wave physics for sea states over shallow banks.

The surf-finder treated one swell train breaking at H = 0.78 d. Ships meet a
random sea, so here a sea state (Hs, Tp) is spread over a Rayleigh range of
wave heights and the Battjes-Janssen (1978) model gives the fraction of those
waves that are breaking because of the depth:

    Hm   = 0.88 / k * tanh(gamma * k * d / 0.88)    largest wave the depth allows
    Hrms = Hs / sqrt(2)
    (1 - Qb) / ln(Qb) = -(Hrms / Hm)^2               Qb = fraction breaking

Before it reaches the bank the sea also shoals: Hs_bank = Hs_offshore * Ks with
Ks = sqrt(cg(d_offshore) / cg(d_bank)). Inverting gives the offshore Hs (at a
given peak period) that makes a chosen fraction of waves break on the bank,
which is the forecast trigger we report.
"""
import numpy as np

G = 9.81
GAMMA = 0.73          # Battjes & Stive (1985) mean breaker parameter for random seas

# Hazard levels: fraction of waves breaking because of the depth.
LEVELS = [
    ("steep", 0.005, "Steep, heaping seas with occasional breakers (1 in 200 waves)"),
    ("breaking", 0.05, "Breaking seas: 1 in 20 waves breaks, dangerous for small and medium vessels"),
    ("heavy", 0.20, "Heavy broken water: 1 in 5 waves breaks, dangerous for any vessel"),
]


def wavenumber(T, d):
    """Linear dispersion, Fenton & McKee explicit approximation (good to ~1.5%)."""
    T = np.asarray(T, float); d = np.maximum(np.asarray(d, float), 0.1)
    L0 = G * T ** 2 / (2 * np.pi)
    L = L0 * np.tanh((2 * np.pi * d / L0) ** 0.75) ** (2 / 3)
    return 2 * np.pi / L


def group_speed(T, d):
    k = wavenumber(T, d); d = np.maximum(np.asarray(d, float), 0.1)
    kd = np.minimum(k * d, 30)
    c = 2 * np.pi / (np.asarray(T, float) * k)
    return c * 0.5 * (1 + 2 * kd / np.sinh(2 * kd))


def shoaling(T, d, d0):
    """Ks taking a sea from depth d0 (where the wave model value is) to depth d."""
    return np.sqrt(group_speed(T, d0) / group_speed(T, d))


def h_max(T, d, gamma=GAMMA):
    k = wavenumber(T, d)
    return 0.88 / k * np.tanh(gamma * k * np.maximum(d, 0.1) / 0.88)


def ratio_for_qb(q):
    """Hrms/Hm at which the Battjes-Janssen breaking fraction equals q."""
    return np.sqrt(-(1 - q) / np.log(q))


def qb_from_ratio(b):
    """Breaking fraction for b = Hrms/Hm (solve (1-Q)/lnQ = -b^2)."""
    b = np.atleast_1d(np.asarray(b, float))
    lo, hi = np.full_like(b, 1e-12), np.ones_like(b) - 1e-12
    for _ in range(60):
        m = 0.5 * (lo + hi)
        f = (1 - m) / np.log(m) + b ** 2           # decreasing in m
        lo = np.where(f > 0, m, lo); hi = np.where(f > 0, hi, m)
    q = 0.5 * (lo + hi)
    q[b >= 1] = 1.0
    q[b < 0.2] = 0.0
    return q if q.size > 1 else float(q[0])


def qb(Hs0, Tp, d, d0, gamma=GAMMA):
    """Fraction of waves breaking on depth d for offshore sea state Hs0/Tp at depth d0."""
    Hs = Hs0 * shoaling(Tp, d, d0)
    return qb_from_ratio(Hs / np.sqrt(2) / h_max(Tp, d, gamma))


def hs_trigger(q, Tp, d, d0, gamma=GAMMA):
    """Offshore Hs (at depth d0) that makes fraction q of waves break on depth d.

    NaN where that Hs would be steeper than real seas get (Hs/L0 > 1/15),
    i.e. the depth alone can't trip that period."""
    h = np.sqrt(2) * ratio_for_qb(q) * h_max(Tp, d, gamma) / shoaling(Tp, d, d0)
    L0 = G * np.asarray(Tp, float) ** 2 / (2 * np.pi)
    return np.where(h > L0 / 15, np.nan, h)
