"""
tessera_sensitivity.py
----------------------
Variance-based global sensitivity analysis of the residue layer and its
upstream drivers, using Saltelli sampling and first-order plus total-effect
Sobol indices.  Morris elementary effects provide a cheap screening pass.
"""

import numpy as np
from SALib.sample import sobol as sobol_sample
from SALib.sample import morris as morris_sample
from SALib.analyze import sobol as sobol_analyze
from SALib.analyze import morris as morris_analyze

import tessera_params as P
import tessera_model as M

# Parameters explored, with the span applied to each.  Barriers and state
# free energies are shifted additively in units of RT; kinetic parameters are
# varied by half a decade either side of nominal.
SENS_SPEC = [
    ("b_ox", "rt", 1.2), ("b_psulf", "rt", 1.2), ("b_overox", "rt", 1.2),
    ("b_psox", "rt", 1.2), ("b_depsulf", "rt", 1.2), ("b_glut", "rt", 1.2),
    ("g_SOH", "rt", 1.2), ("g_SSH", "rt", 1.2), ("g_SO2H", "rt", 1.2),
    ("k_turn", "log", 0.5), ("v_cbs", "log", 0.5), ("k_sqr", "log", 0.5),
    ("K_sqr_inh", "log", 0.4), ("q_nox", "log", 0.4), ("k_px", "log", 0.4),
    ("K_rps3", "log", 0.5),
]
NAMES = [s[0] for s in SENS_SPEC]


def problem():
    b = P.edge_barriers()
    eng = [e[P.E_NAME] for e in P.EDGES]
    bounds = []
    for nm, kind, span in SENS_SPEC:
        if kind == "rt":
            if nm.startswith("b_"):
                c = b[eng.index(nm[2:])]
            else:
                c = P.G_STATE[P.STATES.index(nm[2:])]
            bounds.append([c - span, c + span])
        else:
            c = np.log10(P.NOMINAL[nm])
            bounds.append([c - span, c + span])
    return dict(num_vars=len(SENS_SPEC), names=NAMES, bounds=bounds)


def apply_sample(row):
    p = dict(P.NOMINAL)
    b = P.edge_barriers().copy()
    g = P.G_STATE.copy()
    eng = [e[P.E_NAME] for e in P.EDGES]
    for val, (nm, kind, _s) in zip(row, SENS_SPEC):
        if kind == "rt":
            if nm.startswith("b_"):
                b[eng.index(nm[2:])] = val
            else:
                g[P.STATES.index(nm[2:])] = val
        else:
            p[nm] = 10.0 ** val
    return p, b, g


def outputs(row, t_ogd=90.0, t_end=720.0):
    """Three scalar responses: irreversible loading of the site, the loss of
    the persulfide reservoir, and the fate of the cell."""
    p, b, g = apply_sample(row)
    try:
        y0 = M.normoxic_steady_state(p, b, g, t_relax=2000.0)
        out = M.simulate(p=p, barriers=b, gstate=g,
                         prot=M.Protocol(t_ogd=t_ogd, t_end=t_end),
                         t_eval=np.linspace(0.0, t_end, 121), y0=y0,
                         rtol=1e-5, atol=1e-8)
    except Exception:
        return np.array([np.nan, np.nan, np.nan])
    if out is None:
        return np.array([np.nan, np.nan, np.nan])
    t, Y = out
    peak_sulfinate = float(Y[M.IDX['th3']].max())
    auc_persulfide = float(np.trapezoid(Y[M.IDX['th2']], t) / (t[-1] - t[0]))
    death = float(Y[M.IDX['D']][-1])
    return np.array([peak_sulfinate, auc_persulfide, death])


def run_sobol(N=256, seed=7):
    prob = problem()
    X = sobol_sample.sample(prob, N, calc_second_order=False, seed=seed)
    Y = np.array([outputs(r) for r in X])
    res = {}
    labels = ["peak_sulfinate", "mean_persulfide", "death_fraction"]
    for k, lab in enumerate(labels):
        y = Y[:, k]
        good = np.isfinite(y)
        yy = np.where(good, y, np.nanmean(y[good]) if good.any() else 0.0)
        if np.std(yy) < 1e-12:
            res[lab] = dict(S1=np.zeros(prob['num_vars']),
                            ST=np.zeros(prob['num_vars']),
                            S1_conf=np.zeros(prob['num_vars']),
                            ST_conf=np.zeros(prob['num_vars']))
            continue
        a = sobol_analyze.analyze(prob, yy, calc_second_order=False,
                                  print_to_console=False, seed=seed)
        res[lab] = {k2: np.asarray(a[k2]) for k2 in ['S1', 'ST', 'S1_conf', 'ST_conf']}
    return prob, X, Y, res


def run_morris(N=60, seed=11):
    prob = problem()
    X = morris_sample.sample(prob, N, num_levels=4, seed=seed)
    Y = np.array([outputs(r) for r in X])
    out = {}
    labels = ["peak_sulfinate", "mean_persulfide", "death_fraction"]
    for k, lab in enumerate(labels):
        y = Y[:, k]
        good = np.isfinite(y)
        yy = np.where(good, y, np.nanmean(y[good]) if good.any() else 0.0)
        a = morris_analyze.analyze(prob, X, yy, num_levels=4,
                                   print_to_console=False, seed=seed)
        out[lab] = {k2: np.asarray(a[k2]) for k2 in ['mu_star', 'sigma']}
    return prob, out
