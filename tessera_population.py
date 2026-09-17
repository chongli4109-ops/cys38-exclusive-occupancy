"""
tessera_population.py
---------------------
Population-level response.  The single-cell decision layer is an all-or-none
switch, so a graded viability curve of the kind a colorimetric assay reports
can only arise from cell-to-cell variability.  An ensemble of cells is drawn
with log-normal multiplicative dispersion on the parameters known to vary
between cells, and the surviving fraction is the population observable.
"""

import numpy as np

import tessera_params as P
import tessera_model as M

# Parameters allowed to vary between cells, with the coefficient of
# variation applied to each.
HETERO = {
    'v_c0': 0.35, 'v_cf': 0.30, 'K_cf': 0.25, 'd_casp': 0.20,
    'k_dam': 0.30, 'k_rep': 0.25, 'rho_syn': 0.18, 'v_cbs': 0.30,
    'k_ikk': 0.25, 'K_bcl': 0.25, 'k_txb': 0.25, 'k_exec': 0.20,
}
HETERO_BARRIER = {'ox': 0.20, 'overox': 0.20, 'psulf': 0.20}


def draw_population(n, seed=0, scale=1.0):
    """Draw n parameter sets; scale rescales every dispersion."""
    rng = np.random.default_rng(seed)
    base = dict(P.NOMINAL)
    b0 = P.edge_barriers()
    names = [e[P.E_NAME] for e in P.EDGES]
    cells = []
    for _ in range(n):
        p = dict(base)
        for k, cv in HETERO.items():
            s = np.sqrt(np.log1p((cv * scale) ** 2))
            p[k] = base[k] * np.exp(rng.normal(-0.5 * s * s, s))
        b = b0.copy()
        for k, cv in HETERO_BARRIER.items():
            b[names.index(k)] = b0[names.index(k)] + rng.normal(0.0, cv * scale)
        cells.append((p, b))
    return cells


def population_response(prot, cells, t_end=None, y0_cache=None):
    """Surviving fraction and mean trajectories across the ensemble."""
    surv, traj = [], []
    for p, b in cells:
        y0 = (y0_cache.get(id(p)) if y0_cache else None)
        if y0 is None:
            y0 = M.normoxic_steady_state(p, b, P.G_STATE)
            if y0_cache is not None:
                y0_cache[id(p)] = y0
        out = M.simulate(p=p, barriers=b, prot=prot,
                         t_eval=np.linspace(0.0, prot.t_end, 121),
                         y0=y0, rtol=1e-6, atol=1e-9)
        if out is None:
            continue
        t, Y = out
        surv.append(1.0 - Y[M.IDX['D']][-1])
        traj.append(Y)
    return float(np.mean(surv)), np.array(surv), np.mean(traj, axis=0), t
