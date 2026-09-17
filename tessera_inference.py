"""
tessera_inference.py
--------------------
Parameter estimation, identifiability analysis and model comparison.

Identifiability is assessed in two complementary ways.  The sensitivity
matrix of the standardised residuals gives the Fisher information, whose
eigenvalue spectrum, condition number and pairwise collinearity indices
quantify how well the parameters are jointly constrained by a given
measurement panel.  A profile likelihood is then computed exactly for a
subset of parameters as an independent confirmation.
"""

import numpy as np
from scipy.optimize import least_squares

import tessera_params as P
import tessera_model as M
import tessera_data as D


# ----------------------------------------------------------------------
# parameter packing
# ----------------------------------------------------------------------
def pack(variant):
    """Names and nominal values of the free residue-layer parameters."""
    names, x0, kind = [], [], []
    b = P.edge_barriers()
    eng = [e[P.E_NAME] for e in P.EDGES]
    if variant == "tessera":
        for i, n in enumerate(eng):
            names.append("b_" + n); x0.append(b[i]); kind.append("lin")
        for i in range(1, P.NSTATE):
            names.append("g_" + P.STATES[i]); x0.append(P.G_STATE[i]); kind.append("lin")
    elif variant == "unconstrained":
        v = M.default_vpar("unconstrained")
        for i, n in enumerate(eng):
            names += ["k_%s_f" % n, "k_%s_r" % n]
            x0 += [np.log(max(v[2 * i], 1e-14)), np.log(max(v[2 * i + 1], 1e-14))]
            kind += ["log", "log"]
    elif variant == "lumped":
        names = ["b_ox", "b_red", "gamma_s", "K_s"]
        x0 = [b[0], b[1], 1.8, 25.0]
        kind = ["lin", "lin", "lin", "lin"]
    elif variant == "independent":
        for i in [2, 0, 4, 6, 3, 1, 5, 7]:
            names.append("b_" + eng[i]); x0.append(b[i]); kind.append("lin")
    elif variant == "hill":
        names = ["gamma_s", "K_s", "K_ox"]
        x0 = [1.8, 25.0, 3.0]
        kind = ["lin", "lin", "lin"]
    else:
        raise KeyError(variant)
    return names, np.asarray(x0, dtype=float), kind


def unpack(variant, x):
    """Map a free-parameter vector onto model arguments."""
    b = P.edge_barriers().copy()
    g = P.G_STATE.copy()
    vpar = M.default_vpar(variant)
    if variant == "tessera":
        b[:] = x[:P.NEDGE]
        g[1:] = x[P.NEDGE:P.NEDGE + P.NSTATE - 1]
    elif variant == "unconstrained":
        vpar = np.exp(np.asarray(x, dtype=float))
    elif variant == "lumped":
        b[0], b[1] = x[0], x[1]
        vpar = np.array([x[2], x[3], 3.0])
    elif variant == "independent":
        for slot, i in enumerate([2, 0, 4, 6, 3, 1, 5, 7]):
            b[i] = x[slot]
    elif variant == "hill":
        vpar = np.array([x[0], x[1], x[2]])
    return b, g, vpar


# ----------------------------------------------------------------------
# objective
# ----------------------------------------------------------------------
def residual_vector(x, variant, dataset, panel=None, conds=None):
    panel = D.FIT_PANEL if panel is None else panel
    p = dict(P.NOMINAL)
    b, g, vpar = unpack(variant, x)
    try:
        y0 = M.normoxic_steady_state(p, b, g, t_relax=2500.0,
                                     variant=variant, vpar=vpar)
    except Exception:
        return np.full(2000, 1e3)
    res = []
    for name, kw in D.conditions():
        if conds is not None and name not in conds:
            continue
        pred = D.forward(p, b, g, kw, variant=variant, vpar=vpar, y0=y0)
        if pred is None:
            return np.full(2000, 1e3)
        res.append(D.residuals(pred, dataset['data'][name], panel=panel))
    r = np.concatenate(res)
    return np.where(np.isfinite(r), r, 1e3)


def cost(x, variant, dataset, panel=None, conds=None):
    r = residual_vector(x, variant, dataset, panel, conds)
    return 0.5 * float(r @ r)


def fit(variant, dataset, panel=None, x0=None, max_nfev=260, seed=0,
        perturb=0.0, verbose=0):
    names, xn, kind = pack(variant)
    if x0 is None:
        x0 = xn.copy()
        if perturb > 0:
            rng = np.random.default_rng(seed)
            x0 = x0 + rng.normal(0.0, perturb, size=x0.size)
    lo = np.where(np.array(kind) == "log", x0 - 4.0, x0 - P.PRIOR_RT_SPAN * 2)
    hi = np.where(np.array(kind) == "log", x0 + 4.0, x0 + P.PRIOR_RT_SPAN * 2)
    sol = least_squares(residual_vector, x0, bounds=(lo, hi),
                        args=(variant, dataset, panel), method='trf',
                        max_nfev=max_nfev, verbose=verbose, x_scale='jac')
    return dict(variant=variant, names=names, x=sol.x, cost=float(sol.cost),
                nfev=int(sol.nfev), success=bool(sol.success))


# ----------------------------------------------------------------------
# identifiability
# ----------------------------------------------------------------------
def sensitivity_matrix(x, variant, dataset, panel=None, rel=1e-4):
    r0 = residual_vector(x, variant, dataset, panel)
    S = np.zeros((r0.size, x.size))
    for i in range(x.size):
        h = rel * max(abs(x[i]), 1.0)
        xp = x.copy(); xp[i] += h
        S[:, i] = (residual_vector(xp, variant, dataset, panel) - r0) / h
    return S


def identifiability(S, names, tau=1.0, se_max=0.5):
    """
    Absolute identifiability of a parameter set given a measurement panel.

    Parameters are carried in natural-logarithmic rate units or in units of
    RT, which are equivalent for this model because a rate is generated as
    the exponential of an energy.  A direction is therefore counted as
    identifiable when displacing it by one unit changes the standardised
    residual vector by at least one standard deviation, that is when the
    corresponding eigenvalue of the Fisher information exceeds tau.  A
    relative criterion is avoided deliberately: rescaling the whole
    information matrix, as happens when observables are removed, leaves a
    relative spectrum unchanged while destroying real information.
    """
    F = S.T @ S
    w = np.clip(np.linalg.eigvalsh(F), 0.0, None)
    n_ident = int(np.sum(w > tau))
    pos = w[w > tau]
    cond = float(pos.max() / pos.min()) if pos.size else float('inf')
    norms = np.linalg.norm(S, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    Sn = S / norms
    ev = np.linalg.eigvalsh(Sn.T @ Sn)
    gamma = float(1.0 / np.sqrt(max(ev.min(), 1e-300)))
    try:
        cov = np.linalg.pinv(F, rcond=1e-12)
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except Exception:
        se = np.full(len(names), np.nan)
    n_well = int(np.sum(np.isfinite(se) & (se < se_max)))
    return dict(eigenvalues=w[::-1], n_identifiable=n_ident, condition=cond,
                collinearity=gamma, stderr=se, n_param=len(names),
                lambda_min=float(w.min()), lambda_max=float(w.max()),
                n_well_determined=n_well,
                median_se=float(np.nanmedian(se)) if se.size else float('nan'))


def profile_likelihood(variant, dataset, idx, xhat, panel=None,
                       span=1.2, n=9, max_nfev=45):
    """Exact profile likelihood along one parameter direction."""
    names, _, kind = pack(variant)
    base = cost(xhat, variant, dataset, panel)
    grid = xhat[idx] + np.linspace(-span, span, n)
    prof = []
    for v in grid:
        free = [j for j in range(xhat.size) if j != idx]
        x0 = xhat.copy()

        def r(z):
            xx = x0.copy()
            xx[idx] = v
            xx[free] = z
            return residual_vector(xx, variant, dataset, panel)

        sol = least_squares(r, xhat[free], method='lm', max_nfev=max_nfev)
        prof.append(float(sol.cost))
    return grid, np.asarray(prof), base


def information_criteria(cost_value, n_data, n_param):
    """Gaussian log-likelihood on standardised residuals."""
    chi2 = 2.0 * cost_value
    ll = -0.5 * chi2
    aic = 2 * n_param - 2 * ll
    bic = n_param * np.log(n_data) - 2 * ll
    aicc = aic + (2 * n_param * (n_param + 1)) / max(n_data - n_param - 1, 1)
    return dict(chi2=chi2, logL=ll, AIC=aic, AICc=aicc, BIC=bic)
