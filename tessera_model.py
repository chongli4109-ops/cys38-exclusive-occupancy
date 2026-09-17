"""
tessera_model.py
----------------
Core dynamical system of the TESSERA framework.

Four coupled layers:
  L1  bioenergetic and oxidant state driven by the oxygen-glucose
      deprivation protocol
  L2  mutually exclusive occupancy of p65 Cys38 on the probability simplex,
      with rate constants generated from a state-energy / barrier
      parameterisation that satisfies the Wegscheider cycle conditions
  L3  IKK / IkB / NF-kB nucleocytoplasmic module whose transcriptional
      output is gated by the Cys38 occupancy vector
  L4  Bcl-xL / effector-caspase survival-death decision with positive
      feedback, giving a saddle-node commitment point
"""

import numpy as np
from scipy.integrate import solve_ivp

import tessera_params as P

# ----------------------------------------------------------------------
# state indexing
# ----------------------------------------------------------------------
IDX = dict(
    A=0, M=1, Px=2, G=3, S=4, X=5,
    th0=6, th1=7, th2=8, th3=9, th4=10, th5=11,
    K=12, Nc=13, Nn=14, NIc=15, NIn=16, Ic=17, In=18,
    mI=19, a20=20, mB=21, B=22, C=23, D=24,
)
NVAR = 25
TH = slice(6, 12)


# ----------------------------------------------------------------------
# OGD protocol
# ----------------------------------------------------------------------
class Protocol:
    """Oxygen-glucose availability and exogenous sulfide donor input."""

    def __init__(self, t_ogd=90.0, t_end=720.0, ramp=2.0,
                 donor_dose=0.0, donor_time=0.0, donor_kind="bolus",
                 donor_tau=6.0):
        self.t_ogd = t_ogd
        self.t_end = t_end
        self.ramp = ramp
        self.donor_dose = donor_dose
        self.donor_time = donor_time
        self.donor_kind = donor_kind
        self.donor_tau = donor_tau

    def oxygen(self, t):
        """Availability: 0 through the deprivation window, 1 after restoration."""
        z = np.clip((t - self.t_ogd) / self.ramp, -50.0, 50.0)
        return 1.0 / (1.0 + np.exp(-z))

    def donor(self, t):
        """Exogenous sulfide delivery rate (uM/min)."""
        if self.donor_dose <= 0.0:
            return 0.0
        dt = t - self.donor_time
        if dt < 0.0:
            return 0.0
        if self.donor_kind == "bolus":
            # rapid-release donor, e.g. a sulfide salt
            return self.donor_dose / self.donor_tau * np.exp(-dt / self.donor_tau)
        # slow-release donor, e.g. a morpholine phosphinodithioate
        tau = self.donor_tau * 25.0
        return self.donor_dose / tau * np.exp(-dt / tau)


# ----------------------------------------------------------------------
# occupancy generator
# ----------------------------------------------------------------------
def cosubstrate(name, Px, S, G, X, A, NOl, p):
    """Activity factor of the cosubstrate on one directed edge."""
    if name == "one":
        return 1.0
    if name == "P":
        return Px / P.P_REF
    if name == "S":
        return S / P.S_REF
    if name == "G":
        return G / P.G_REF
    if name == "X":
        return X
    if name == "Xox":
        return 1.0 - X
    if name == "NO":
        return NOl
    if name == "srx":
        return X * A / (P.K_SRX + A)
    if name == "srxrev":
        return (1.0 - X) * (1.0 - A)
    raise KeyError(name)


def turnover_matrix(k_turn):
    """Unidirectional replacement of every modified state by reduced thiol."""
    T = np.zeros((P.NSTATE, P.NSTATE))
    for i in range(P.NSTATE):
        if i != P.S_SH:
            T[i, P.S_SH] = k_turn
    np.fill_diagonal(T, -T.sum(axis=1))
    return T


def generator(Px, S, G, X, A, NOl, barriers, gstate, p, dG=None):
    """
    Infinitesimal generator Q of the occupancy chain.
    Q[i, j] is the rate of i -> j for i != j; rows sum to zero.
    """
    if dG is None:
        dG = P.edge_dG()
    Q = np.zeros((P.NSTATE, P.NSTATE))
    for e, (name, i, j, _b, _d, af, ar) in enumerate(P.EDGES):
        b = barriers[e]
        d = dG[e]
        kf = np.exp(gstate[i] - b) * cosubstrate(af, Px, S, G, X, A, NOl, p)
        kr = np.exp(gstate[j] - b - d) * cosubstrate(ar, Px, S, G, X, A, NOl, p)
        Q[i, j] += kf
        Q[j, i] += kr
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def oxidant_source(t, prot, A, Sl, p):
    """Three-phase oxidant production plus sulfide-dependent respiratory leak."""
    O = prot.oxygen(t)
    # early mitochondrial burst at the onset of deprivation
    j_mito = p['q_mito'] * np.exp(-max(t, 0.0) / p['tau_mito'])
    # delayed, energy-linked xanthine oxidase phase during deprivation
    ramp = 1.0 / (1.0 + np.exp(-(t - p['t_xo']) / 6.0))
    j_xo = p['q_xo'] * ramp * (1.0 - O) * (1.0 - A) ** 2
    # reoxygenation burst and reverse electron transport
    dt = t - prot.t_ogd
    if dt > 0.0:
        rx = np.exp(-dt / p['tau_nox'])
        j_nox = p['q_nox'] * O * rx
        j_ret = p['q_ret'] * O * rx * (1.0 - A)
    else:
        j_nox = j_ret = 0.0
    # respiratory inhibition by excess sulfide raises electron leak
    leak = (Sl / p['K_sqr_inh']) ** p['n_sqr']
    j_leak = p['q_leak'] * leak / (1.0 + leak)
    return p['q_base'] + j_mito + j_xo + j_nox + j_ret + j_leak


def occupancy_layer(variant, th, Px, S, G, X, A, p, barriers, gstate, dG, vpar):
    """
    Residue layer.  Returns the derivative of the six carried variables, the
    retained DNA-binding competence, and the RPS3 recruitment gate.  The
    variants differ only in how chemistry at Cys38 is represented; every
    other layer of the model is identical across them.
    """
    if variant == "tessera":
        Q = generator(Px, S, G, X, A, p['NO_level'], barriers, gstate, p, dG=dG)
        dth = (Q + turnover_matrix(p['k_turn'])).T @ th
        comp = float(P.W_COMP @ th)
        occ = th[P.S_SSH]

    elif variant == "unconstrained":
        # identical topology, but every directed rate is a free parameter,
        # so the Wegscheider cycle conditions are not imposed
        Q = free_generator(Px, S, G, X, A, p['NO_level'], vpar, p)
        dth = (Q + turnover_matrix(p['k_turn'])).T @ th
        comp = float(P.W_COMP @ th)
        occ = th[P.S_SSH]

    elif variant == "lumped":
        # single reduced/oxidised dichotomy with sulfide as a multiplier,
        # the representation used by conventional redox signalling models
        r = np.clip(th[0], 0.0, 1.0)
        kox = np.exp(-barriers[0]) * Px / P.P_REF
        kred = np.exp(-barriers[1]) * X
        dr = -kox * r + kred * (1.0 - r) + p['k_turn'] * (1.0 - r)
        dth = np.zeros(P.NSTATE)
        dth[0] = dr
        boost = vpar[0] * S / (vpar[1] + S)
        comp = (P.W_COMP[P.S_SH] * r + P.W_COMP[P.S_SO2H] * (1.0 - r)) * (1.0 + boost)
        occ = r * S / (vpar[1] + S)

    elif variant == "independent":
        # four modification indicators evolving independently, the product
        # form inherited from multisite phosphorylation theory
        mm = np.clip(th[:4], 0.0, 1.0)
        kon = np.array([np.exp(-barriers[2]) * S / P.S_REF,
                        np.exp(-barriers[0]) * Px / P.P_REF,
                        np.exp(-barriers[4]) * Px / P.P_REF,
                        np.exp(-barriers[6]) * G / P.G_REF])
        koff = np.array([np.exp(-barriers[3]) * X, np.exp(-barriers[1]) * X,
                         np.exp(-barriers[5]), np.exp(-barriers[7]) * X])
        dm = kon * (1.0 - mm) - (koff + p['k_turn']) * mm
        dth = np.zeros(P.NSTATE)
        dth[:4] = dm
        wk = np.array([P.W_COMP[P.S_SSH], P.W_COMP[P.S_SOH],
                       P.W_COMP[P.S_SO2H], P.W_COMP[P.S_SSG]])
        comp = float(np.prod(1.0 - (1.0 - wk) * mm))
        occ = mm[0]

    elif variant == "hill":
        # no residue chemistry: a phenomenological modifier on the output
        dth = np.zeros(P.NSTATE)
        comp = 1.0 / (1.0 + (Px / vpar[2]) ** 2)
        occ = S / (vpar[1] + S)

    else:
        raise KeyError(variant)

    o = occ ** p['h_rps3']
    rps3 = o / (p['K_rps3'] ** p['h_rps3'] + o)
    return dth, comp, rps3


def free_generator(Px, S, G, X, A, NOl, vpar, p):
    """Generator built from 2*NEDGE unconstrained directed rate constants."""
    Q = np.zeros((P.NSTATE, P.NSTATE))
    for e, ed in enumerate(P.EDGES):
        i, j = ed[P.E_I], ed[P.E_J]
        af, ar = ed[P.E_AF], ed[P.E_AR]
        Q[i, j] += vpar[2 * e] * cosubstrate(af, Px, S, G, X, A, NOl, p)
        Q[j, i] += vpar[2 * e + 1] * cosubstrate(ar, Px, S, G, X, A, NOl, p)
    np.fill_diagonal(Q, 0.0)
    np.fill_diagonal(Q, -Q.sum(axis=1))
    return Q


def default_vpar(variant):
    if variant == "unconstrained":
        b, g, d = P.edge_barriers(), P.G_STATE, P.edge_dG()
        v = np.zeros(2 * P.NEDGE)
        for e, ed in enumerate(P.EDGES):
            i, j = ed[P.E_I], ed[P.E_J]
            v[2 * e] = np.exp(g[i] - b[e])
            v[2 * e + 1] = np.exp(g[j] - b[e] - d[e])
        return v
    if variant in ("lumped", "independent", "hill"):
        return np.array([1.8, 25.0, 3.0])
    return np.zeros(1)


def rhs(t, y, prot, p, barriers, gstate, dG=None, variant="tessera", vpar=None):
    y = np.maximum(y, 0.0)
    A, M, Px, G, S, X = y[0:6]
    th = y[TH]
    K, Nc, Nn, NIc, NIn, Ic, In, mI, a20, mB, B, C, D = y[12:25]
    O = prot.oxygen(t)

    # ---- L1 bioenergetics and oxidant handling -----------------------
    inh = 1.0 / (1.0 + (S / p['K_sqr_inh']) ** p['n_sqr'])
    dA = p['rho_syn'] * O * M * inh * (1.0 - A) - p['rho_use'] * A
    dM = (-p['k_dam'] * Px * Px / (p['K_mdam'] ** 2 + Px * Px) * M
          + p['k_rep'] * A * (1.0 - M))
    # oxidant clearance follows reductant supply: substrate availability
    # (proxied by O) dominates, with a weaker dependence on energy charge
    clear = (p['k_px'] * (G / P.G_REF) * (0.30 + 0.70 * O) * (0.40 + 0.60 * A)
             + p['k_p0'])
    dPx = oxidant_source(t, prot, A, S, p) - clear * Px
    dGsh = p['v_gsh'] * A * (p['G_tot'] - G) - p['k_gox'] * Px * G
    hyp = 1.0 + (p['f_hyp'] - 1.0) * (1.0 - O)
    dS = (p['v_cbs'] * hyp * (0.35 + 0.65 * A) - p['k_sqr'] * O * S
          - p['k_sloss'] * S
          + prot.donor(t))
    dX = p['k_trxr'] * A * (1.0 - X) - p['k_trxu'] * Px * X

    # ---- L2 Cys38 occupancy on the simplex ---------------------------
    # p65 turnover, where present, is a driven unidirectional process
    # sustained by translation and is therefore excluded from the
    # Wegscheider constraint applied to the chemical interconversion edges.
    dth, comp, rps3 = occupancy_layer(variant, th, Px, S, G, X, A, p,
                                      barriers, gstate, dG, vpar)

    # ---- L3 NF-kB module ---------------------------------------------
    phi_i = Nn * comp                                 # inflammatory drive
    phi_s = Nn * comp * rps3                          # survival-gene drive

    # transcription and translation are arrested by energy failure
    fE = A / (p['K_E'] + A)

    dK = (p['k_ikk'] * Px / (p['K_ikk'] + Px) * (1.0 - K)
          - p['d_ikk'] * K * (1.0 + p['k_a20'] * a20))
    rel = p['k_rel'] * K
    dNIc = p['k_ass'] * Nc * Ic - p['k_dis'] * NIc - rel * NIc
    dNc = (-p['k_ass'] * Nc * Ic + p['k_dis'] * NIc + rel * NIc
           - p['k_in'] * Nc + p['k_out'] * Nn + p['k_exp'] * NIn)
    dNn = p['k_in'] * Nc - p['k_out'] * Nn - p['k_ass'] * Nn * In + p['k_dis'] * NIn
    dNIn = p['k_ass'] * Nn * In - p['k_dis'] * NIn - p['k_exp'] * NIn
    dIc = (p['k_tli'] * fE * mI - p['k_ass'] * Nc * Ic + p['k_dis'] * NIc
           - p['d_ikb'] * Ic - p['k_iin'] * Ic + p['k_iout'] * In - rel * Ic)
    dIn = (p['k_iin'] * Ic - p['k_iout'] * In - p['k_ass'] * Nn * In
           + p['k_dis'] * NIn - p['d_ikb'] * In)
    dmI = p['k_txi'] * fE * phi_i / (p['K_txi'] + phi_i) - p['d_mi'] * mI
    da = p['k_txa'] * fE * phi_i / (p['K_txa'] + phi_i) - p['d_a20'] * a20

    # ---- L4 survival-death decision ----------------------------------
    dmB = p['k_txb'] * fE * phi_s / (p['K_txb'] + phi_s) - p['d_mb'] * mB
    dB = p['k_tlb'] * fE * mB - p['d_bcl'] * B
    # commitment is driven by persistent mitochondrial failure together
    # with the acute oxidant load, not by the transient burst alone
    dam = (1.0 - M) + p['w_ox'] * (1.0 - A) * Px / (p['K_dam'] + Px)
    fb = p['v_cf'] * C * C / (p['K_cf'] ** 2 + C * C)
    nb = p['n_bcl']
    guard = p['K_bcl'] ** nb / (p['K_bcl'] ** nb + B ** nb)
    dC = (p['v_c0'] + fb) * dam * guard - p['d_casp'] * C
    # irreversible execution: cleaved substrate does not revert
    ce = C ** p['h_exec']
    dD = p['k_exec'] * ce / (p['K_exec'] ** p['h_exec'] + ce) * (1.0 - D)

    return np.array([dA, dM, dPx, dGsh, dS, dX,
                     dth[0], dth[1], dth[2], dth[3], dth[4], dth[5],
                     dK, dNc, dNn, dNIc, dNIn, dIc, dIn, dmI, da, dmB, dB,
                     dC, dD])


# ----------------------------------------------------------------------
# steady state and simulation
# ----------------------------------------------------------------------
def normoxic_steady_state(p, barriers, gstate, t_relax=4000.0,
                          variant="tessera", vpar=None):
    """Relax the system under full oxygen and glucose to obtain y0."""
    prot = Protocol(t_ogd=-1e9, t_end=t_relax)
    y0 = np.zeros(NVAR)
    y0[IDX['A']] = 0.75
    y0[IDX['M']] = 1.0
    y0[IDX['Px']] = 0.05
    y0[IDX['G']] = 5.0
    y0[IDX['S']] = 25.0
    y0[IDX['X']] = 0.9
    y0[TH] = np.array([0.94, 0.005, 0.03, 0.005, 0.015, 0.005])
    y0[IDX['Nc']] = 0.05
    y0[IDX['NIc']] = 0.95
    y0[IDX['Ic']] = 0.35
    y0[IDX['B']] = 1.0
    if vpar is None:
        vpar = default_vpar(variant)
    if variant == "lumped":
        y0[TH] = 0.0
        y0[IDX['th0']] = 0.95
    elif variant == "independent":
        y0[TH] = 0.0
        y0[IDX['th0']] = 0.10
    elif variant == "hill":
        y0[TH] = 0.0
    sol = solve_ivp(rhs, (0.0, t_relax), y0,
                    args=(prot, p, barriers, gstate, None, variant, vpar),
                    method="LSODA", rtol=1e-7, atol=1e-9, dense_output=False)
    ss = np.maximum(sol.y[:, -1], 0.0)
    if variant in ("tessera", "unconstrained"):
        ss[TH] /= ss[TH].sum()
    ss[IDX['C']] = 0.0
    ss[IDX['D']] = 0.0
    return ss


def simulate(p=None, barriers=None, gstate=None, prot=None,
             t_eval=None, y0=None, dG=None, rtol=1e-6, atol=1e-9,
             variant="tessera", vpar=None):
    p = dict(P.NOMINAL) if p is None else p
    barriers = P.edge_barriers() if barriers is None else barriers
    gstate = P.G_STATE.copy() if gstate is None else gstate
    prot = Protocol() if prot is None else prot
    if y0 is None:
        y0 = normoxic_steady_state(p, barriers, gstate)
    if t_eval is None:
        t_eval = np.linspace(0.0, prot.t_end, 361)
    if vpar is None:
        vpar = default_vpar(variant)
    sol = solve_ivp(rhs, (0.0, prot.t_end), y0,
                    args=(prot, p, barriers, gstate, dG, variant, vpar),
                    method="LSODA", t_eval=t_eval, rtol=rtol, atol=atol)
    if not sol.success:
        return None
    Y = np.maximum(sol.y, 0.0)
    if variant in ("tessera", "unconstrained"):
        tot = Y[TH].sum(axis=0)
        Y[TH] /= np.where(tot > 0, tot, 1.0)
    return sol.t, Y


def viability(Y, p):
    """Surviving fraction: the complement of irreversibly executed cells."""
    return 1.0 - Y[IDX['D']]


def observables(t, Y, p):
    """Quantities a wet-lab protocol could actually report."""
    return dict(
        persulfide=Y[IDX['th2']],                      # tag-switch on p65
        sulfinate=Y[IDX['th3']],                       # site-specific MS
        nuclear=Y[IDX['Nn']] / (Y[IDX['Nn']] + Y[IDX['Nc']] + 1e-12),
        bclxl=Y[IDX['B']],
        atp=Y[IDX['A']] / max(Y[IDX['A']][0], 1e-12),
        viability=viability(Y, p),
    )
