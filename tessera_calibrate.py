"""
tessera_calibrate.py
--------------------
Solve the occupancy-layer barrier heights so that the resting distribution
of p65 Cys38 matches a specified target occupancy under normoxic
cosubstrate activities.  The state free energies and the cosubstrate
free-energy changes are held fixed, so the Wegscheider cycle conditions
continue to hold exactly for every calibrated parameter set.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.linalg import null_space

import tessera_params as P
import tessera_model as M

# The six independent cycles of the eleven-edge occupancy graph on six
# states (cyclomatic number 11 - 6 + 1 = 6).
CYCLES = [[('ox', 1), ('trxSOH', 1)],
          [('ox', 1), ('psulf', 1), ('depsulf', 1)],
          [('ox', 1), ('glut', 1), ('deglut', 1)],
          [('ox', 1), ('overox', 1), ('srx', 1)],
          [('nitros', 1), ('denitr', 1)],
          [('psulf', 1), ('psox', 1)]]

# Resting occupancy targeted at the normoxic reference state.
TARGET = np.array([0.795, 0.012, 0.102, 0.022, 0.047, 0.022])


def stationary(Q):
    ns = null_space(Q.T)
    v = np.abs(ns[:, 0])
    return v / v.sum()


def resting_occupancy(barriers, u, k_turn=None):
    k_turn = P.NOMINAL['k_turn'] if k_turn is None else k_turn
    Q = M.generator(u['P'], u['S'], u['G'], u['X'], u['A'], u['NO'],
                    barriers, P.G_STATE, P.NOMINAL)
    return stationary(Q + M.turnover_matrix(k_turn))


def calibrate(u, target=TARGET, lam=0.08):
    b0 = P.edge_barriers()

    def resid(b):
        th = resting_occupancy(b, u)
        return np.concatenate([(th - target) / 0.01, lam * (b - b0)])

    sol = least_squares(resid, b0, method='trf', xtol=1e-12, ftol=1e-12)
    return sol.x, resting_occupancy(sol.x, u)


def cycle_report(barriers, u, gstate=None):
    """
    Verify thermodynamic consistency.  For every independent cycle the
    product of forward-to-reverse rate ratios must equal the exponential of
    the cycle affinity built from the cosubstrate terms alone, i.e. it must
    be independent of the state free energies.
    """
    gstate = P.G_STATE if gstate is None else gstate
    cycles = CYCLES
    names = [e[0] for e in P.EDGES]
    out = []
    for cyc in cycles:
        lr = 0.0
        for nm, _ in cyc:
            e = names.index(nm)
            _n, i, j, _b, d, af, ar = P.EDGES[e]
            b = barriers[e]
            kf = np.exp(gstate[i] - b) * M.cosubstrate(af, u['P'], u['S'],
                        u['G'], u['X'], u['A'], u['NO'], P.NOMINAL)
            kr = np.exp(gstate[j] - b - d) * M.cosubstrate(ar, u['P'], u['S'],
                        u['G'], u['X'], u['A'], u['NO'], P.NOMINAL)
            lr += np.log(kf / kr)
        out.append((tuple(n for n, _ in cyc), lr))
    return out


def _activities_from_x(x):
    lnp, lns, lng, xi, lnv, la = x
    return dict(P=np.exp(lnp) * P.P_REF, S=np.exp(lns) * P.S_REF,
                G=np.exp(lng) * P.G_REF, X=1.0 / (1.0 + np.exp(-xi)),
                A=1.0 / (1.0 + np.exp(-la)), NO=np.exp(lnv))


def equilibrium_activities(barriers=None):
    """
    Cosubstrate activities at which every independent cycle affinity
    vanishes simultaneously, solved numerically so that the construction
    remains valid if the edge set is extended.
    """
    barriers = P.edge_barriers() if barriers is None else barriers

    def resid(x):
        u = _activities_from_x(x)
        return np.array([lr for _n, lr in cycle_report(barriers, u)])

    x0 = np.array([-11.0, -4.0, -6.0, -6.0, 1.0, -8.0])
    sol = least_squares(resid, x0, xtol=1e-15, ftol=1e-15, gtol=1e-15)
    return _activities_from_x(sol.x), float(np.max(np.abs(sol.fun)))


def equilibrium_check(barriers=None):
    """
    Relax the occupancy chain at the annulling activities and confirm that it
    reaches the Boltzmann distribution of the state free energies with zero
    net cycle flux, the numerical signature of thermodynamic consistency.
    """
    barriers = P.edge_barriers() if barriers is None else barriers
    g = P.G_STATE
    boltz = np.exp(-g) / np.exp(-g).sum()
    u, _res = equilibrium_activities(barriers)
    Q = M.generator(u['P'], u['S'], u['G'], u['X'], u['A'], u['NO'],
                    barriers, g, P.NOMINAL)
    th = stationary(Q)
    flux = np.array([[Q[i, j] * th[i] - Q[j, i] * th[j]
                      for j in range(P.NSTATE)] for i in range(P.NSTATE)])
    return th, boltz, float(np.max(np.abs(flux))), u


def effective_potential(barriers, u, gstate=None):
    """
    Build the effective state potential implied by the rate constants at the
    given cosubstrate activities.  Existence of a single-valued potential is
    equivalent to vanishing cycle affinity; path independence is therefore a
    direct numerical test of the Wegscheider conditions.
    """
    gstate = P.G_STATE if gstate is None else gstate
    Q = M.generator(u['P'], u['S'], u['G'], u['X'], u['A'], u['NO'],
                    barriers, gstate, P.NOMINAL)
    paths = {0: [[0]], 1: [[0, 1]],
             2: [[0, 1, 2], [0, 2]], 3: [[0, 1, 3], [0, 3]],
             4: [[0, 1, 4], [0, 4]], 5: [[0, 5], [0, 1, 2, 0, 5]]}
    pot, spread = {}, {}
    for node, plist in paths.items():
        vals = []
        for path in plist:
            v, ok = 0.0, True
            for a, b in zip(path[:-1], path[1:]):
                if Q[a, b] <= 0 or Q[b, a] <= 0:
                    ok = False
                    break
                v += -np.log(Q[a, b] / Q[b, a])
            if ok:
                vals.append(v)
        pot[node] = vals[0] if vals else np.nan
        spread[node] = (max(vals) - min(vals)) if len(vals) > 1 else 0.0
    ge = np.array([pot[i] for i in range(P.NSTATE)])
    w = np.exp(-ge)
    return ge, w / w.sum(), max(spread.values())
