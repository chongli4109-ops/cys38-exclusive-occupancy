"""
tessera_bifurcation.py
----------------------
Commitment analysis.

The execution layer is driven by the composite input u = damage x release of
Bcl-xL inhibition.  Continuation of the caspase steady states in u locates
the pair of saddle-node points that bound the bistable region, and the
dynamic commitment threshold is obtained by bisecting on the deprivation
duration at which a single cell ceases to recover.
"""

import numpy as np
from scipy.optimize import brentq

import tessera_params as P
import tessera_model as M


def caspase_rhs(C, u, p):
    return (p['v_c0'] + p['v_cf'] * C * C / (p['K_cf'] ** 2 + C * C)) * u \
           - p['d_casp'] * C


def branches(u, p, cmax=40.0, n=40001):
    C = np.linspace(0.0, cmax, n)
    v = caspase_rhs(C, u, p)
    idx = np.where(np.diff(np.sign(v)) != 0)[0]
    return [brentq(caspase_rhs, C[i], C[i + 1], args=(u, p)) for i in idx]


def continuation(p, u_lo=1e-3, u_hi=1.2, n=600):
    us = np.linspace(u_lo, u_hi, n)
    rec = [(u, branches(u, p)) for u in us]
    counts = np.array([len(r[1]) for r in rec])
    folds = []
    for i in range(1, len(counts)):
        if counts[i] != counts[i - 1]:
            folds.append(0.5 * (us[i] + us[i - 1]))
    return us, rec, folds


def saddle_node_points(p):
    """Locate the two fold points bounding bistability to high precision."""
    def n_roots(u):
        return len(branches(u, p))

    us, _rec, folds = continuation(p)
    out = []
    for f in folds:
        lo, hi = f - (us[1] - us[0]), f + (us[1] - us[0])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if n_roots(mid) == n_roots(lo):
                lo = mid
            else:
                hi = mid
        out.append(0.5 * (lo + hi))
    return out


def survives(t_ogd, p, barriers, gstate, y0, donor=None, t_end=720.0):
    kw = dict(t_ogd=float(t_ogd), t_end=t_end)
    if donor:
        kw.update(donor)
    out = M.simulate(p=p, barriers=barriers, gstate=gstate,
                     prot=M.Protocol(**kw), t_eval=np.array([0.0, t_end]),
                     y0=y0, rtol=1e-6, atol=1e-9)
    if out is None:
        return False, 1.0
    _t, Y = out
    d = float(Y[M.IDX['D']][-1])
    return d < 0.5, d


def commitment_threshold(p, barriers, gstate, y0, donor=None,
                         lo=5.0, hi=400.0, tol=0.5):
    """Longest deprivation a cell survives, by bisection."""
    ok_lo, _ = survives(lo, p, barriers, gstate, y0, donor)
    ok_hi, _ = survives(hi, p, barriers, gstate, y0, donor)
    if not ok_lo:
        return float('nan')
    if ok_hi:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        ok, _ = survives(mid, p, barriers, gstate, y0, donor)
        if ok:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def rescue_boundary(p, barriers, gstate, y0, doses, kind="bolus",
                    t_ogd=90.0, t_end=720.0, times=None):
    """Latest administration time that still rescues, for each dose."""
    times = np.arange(0.0, 301.0, 10.0) if times is None else times
    out = {}
    for d in doses:
        latest = np.nan
        for tt in times:
            donor = dict(donor_dose=float(d), donor_time=float(tt),
                         donor_kind=kind)
            ok, _ = survives(t_ogd, p, barriers, gstate, y0, donor, t_end)
            if ok:
                latest = tt
        out[float(d)] = latest
    return out
