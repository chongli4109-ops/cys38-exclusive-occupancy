"""
tessera_params.py
-----------------
Parameter definitions for the TESSERA model
(Thermodynamically-constrained Exclusive Single-Site Energy-coupled Redox Allocation).

Every parameter carries a provenance tag:
    'lit'   - value or range constrained by published measurement
    'est'   - not measured for p65 Cys38; estimated, prior range stated
    'set'   - structural / scaling choice fixed by construction

Time unit: minutes.  H2O2 in micromolar.  Sulfide pool in micromolar.
Glutathione in millimolar.  Fractional variables dimensionless.
Free energies and barriers are expressed in units of RT.
"""

import numpy as np

RT = 1.0

# ----------------------------------------------------------------------
# Cys38 occupancy states (mutually exclusive occupancy of one sulfur atom)
# ----------------------------------------------------------------------
STATES = ["SH", "SOH", "SSH", "SO2H", "SSG", "SNO"]
S_SH, S_SOH, S_SSH, S_SO2H, S_SSG, S_SNO = range(6)
NSTATE = 6

# Standard free energies of the six protein states (RT units).
# Only differences matter; SO2H is the deepest well, consistent with the
# thermodynamic stability of the sulfinate.
G_STATE = np.array([0.0, 2.0, 1.0, -1.0, 0.5, 0.8])

# ----------------------------------------------------------------------
# Edge table.
# Each edge e = (i, j, b_e, dG_e, act_f, act_r)
#   k_{i->j} = exp(g_i - b_e) * act_f(u)
#   k_{j->i} = exp(g_j - b_e - dG_e) * act_r(u)
# so that k_f/k_r = exp(g_i - g_j + dG_e) * act_f/act_r, and around any
# closed cycle the g-terms telescope to zero: the cycle affinity is fixed
# entirely by the cosubstrate chemistry.  This enforces the Wegscheider
# conditions by construction.
# ----------------------------------------------------------------------
EDGES = [
    # name,      i,      j,       barrier b_e, cosubstrate dG_e, fwd act, rev act
    ("ox",      S_SH,   S_SOH,   1.9549, 12.0, "P",   "one"),
    ("trxSOH",  S_SOH,  S_SH,    1.5581,  5.0, "X",   "Xox"),
    ("psulf",   S_SOH,  S_SSH,   0.8822,  6.0, "S",   "one"),
    ("depsulf", S_SSH,  S_SH,    3.5363,  3.0, "X",   "Xox"),
    ("overox",  S_SOH,  S_SO2H,  6.2664, 14.0, "P",   "one"),
    ("srx",     S_SO2H, S_SH,   11.4991, 10.0, "srx", "srxrev"),
    ("glut",    S_SOH,  S_SSG,   2.1297,  7.0, "G",   "one"),
    ("deglut",  S_SSG,  S_SH,    1.7389,  4.0, "X",   "Xox"),
    ("nitros",  S_SH,   S_SNO,   4.3838,  2.0, "NO",  "one"),
    ("denitr",  S_SNO,  S_SH,    1.4262,  3.0, "X",   "Xox"),
    ("psox",    S_SSH,  S_SOH,   3.2822,  9.0, "P",   "one"),
]
E_NAME, E_I, E_J, E_B, E_DG, E_AF, E_AR = range(7)
EDGE_NAMES = [e[0] for e in EDGES]
NEDGE = len(EDGES)

# Reference activities used to non-dimensionalise the cosubstrate terms
P_REF = 1.0      # uM H2O2
S_REF = 30.0     # uM sulfide/polysulfide pool
G_REF = 5.0      # mM reduced glutathione
K_SRX = 1.0      # ATP half-saturation of sulfiredoxin

# ----------------------------------------------------------------------
# Transcriptional competence weights per occupancy state.
# w = retained sequence-specific DNA-binding competence of the p65 dimer.
# ----------------------------------------------------------------------
W_COMP = np.array([1.00,   # SH   - reference competent thiolate
                   0.18,   # SOH  - sulfenylation lowers binding
                   0.96,   # SSH  - persulfide retains binding
                   0.03,   # SO2H - sulfinate abolishes binding
                   0.12,   # SSG  - glutathionylation suppresses binding
                   0.22])  # SNO  - nitrosylation suppresses binding

# ----------------------------------------------------------------------
# Nominal parameter vector
# ----------------------------------------------------------------------
NOMINAL = dict(
    # --- bioenergetics -------------------------------------------------
    rho_syn   = 0.090,   # ATP resynthesis rate constant            est
    rho_use   = 0.030,   # basal ATP consumption                    est
    k_dam     = 0.0600,  # oxidant damage to mitochondrial pool     est
    K_mdam    = 2.20,    # oxidant half-saturation of damage         est
    k_rep     = 0.0400,  # ATP-dependent mitochondrial repair       est
    K_sqr_inh = 220.0,    # sulfide half-inhibition of respiration   lit-informed
    n_sqr     = 3.0,     # cooperativity of respiratory inhibition  set

    # --- oxidant handling ---------------------------------------------
    q_mito    = 0.62,    # early mitochondrial burst amplitude      lit-informed
    tau_mito  = 4.5,     # early burst decay time constant (min)    lit
    q_xo      = 0.30,    # xanthine-oxidase source gain             lit-informed
    t_xo      = 26.0,    # xanthine-oxidase onset delay (min)       lit
    q_nox     = 1.55,    # reoxygenation NADPH-oxidase burst        lit-informed
    tau_nox   = 11.0,    # reoxygenation burst decay (min)          lit-informed
    q_ret     = 0.85,    # reverse-electron-transport contribution  lit-informed
    k_px      = 1.45,    # peroxiredoxin/GPx removal rate constant  lit-informed
    k_p0      = 0.055,   # oxidant-independent clearance            est
    q_base    = 0.480,   # basal oxidant tone at the protein        est
    q_leak    = 0.620,   # sulfide-dependent respiratory leak       lit-informed

    # --- thiol pools ---------------------------------------------------
    v_gsh     = 0.1200,  # ATP-dependent glutathione resynthesis    lit-informed
    G_tot     = 6.0,     # total glutathione pool (mM)              lit
    k_gox     = 0.0100,   # oxidative consumption of glutathione     est
    k_trxr    = 0.30,    # thioredoxin reductase turnover           est
    k_trxu    = 0.070,    # thioredoxin oxidative load               est

    # --- sulfide pool ---------------------------------------------------
    v_cbs     = 1.310,    # CBS/CSE sulfide production               lit-informed
    f_hyp     = 1.10,    # hypoxic modulation of sulfide synthesis  lit-informed
    k_sqr     = 0.100,   # SQR-dependent oxygen-requiring clearance lit-informed
    k_sloss   = 0.020,   # non-oxidative sulfide loss               est
    k_turn    = 0.0025,  # p65 turnover: replacement by newly made thiol  lit
    NO_level  = 1.0,     # normalised nitrosative input             est

    # --- NF-kB module ---------------------------------------------------
    k_ikk     = 0.145,   # oxidant-driven IKK activation            est
    K_ikk     = 1.10,    # oxidant half-activation of IKK           est
    d_ikk     = 0.075,   # IKK inactivation                         est
    k_a20     = 0.85,    # A20-dependent IKK inactivation gain      lit-informed
    k_ass     = 0.62,    # NF-kB / IkB association                  lit
    k_dis     = 0.0075,  # complex dissociation                     lit
    k_rel     = 0.55,    # IKK-driven release of NF-kB              lit
    k_in      = 0.092,   # NF-kB nuclear import                     lit
    k_out     = 0.014,   # NF-kB nuclear export                     lit
    k_iin     = 0.018,   # IkB nuclear import                       lit
    k_iout    = 0.011,   # IkB nuclear export                       lit
    k_exp     = 0.048,   # nuclear complex export                   lit
    k_txi     = 0.155,   # IkB transcription rate                   lit
    K_txi     = 0.42,    # IkB promoter half-saturation             est
    d_mi      = 0.028,   # IkB mRNA decay                           lit
    k_tli     = 0.26,    # IkB translation                          lit
    d_ikb     = 0.013,   # IkB protein decay                        lit
    k_txa     = 0.082,   # A20 transcription                        est
    K_txa     = 0.55,    # A20 promoter half-saturation             est
    d_a20     = 0.024,   # A20 decay                                est

    # --- survival / death ------------------------------------------------
    k_txb     = 0.115,   # Bcl-xL transcription rate                est
    K_txb     = 0.30,    # Bcl-xL promoter half-saturation          est
    K_rps3    = 0.120,   # RPS3 recruitment threshold in theta_SSH  est
    h_rps3    = 3.0,     # RPS3 recruitment cooperativity           set
    d_mb      = 0.021,   # Bcl-xL mRNA decay                        lit
    k_tlb     = 0.0253,    # Bcl-xL translation                       lit
    d_bcl     = 0.0095,  # Bcl-xL protein decay                     lit
    v_c0      = 0.0040,  # basal effector-caspase activation        est
    v_cf      = 0.9000,   # caspase positive-feedback amplitude      lit-informed
    K_cf      = 1.200,    # caspase feedback half-saturation         est
    K_bcl     = 0.600,    # Bcl-xL inhibition constant               est
    w_ox      = 0.600,   # weight of the acute oxidant term in damage   est
    K_dam     = 1.30,    # oxidant half-saturation of damage drive  est
    d_casp    = 0.0500,   # caspase turnover                         est
    n_bcl     = 3.0,     # steepness of Bcl-xL inhibition of caspase   set
    K_E       = 0.250,   # energy gating of transcription/translation est
    k_exec    = 0.0300,  # rate of irreversible execution           est
    K_exec    = 1.000,   # caspase activity at half-maximal execution est
    h_exec    = 4.0,     # steepness of the execution step          set
)

# Provenance map used to build the manuscript parameter table
PROVENANCE = {
    'lit': ['tau_mito', 't_xo', 'G_tot', 'k_ass', 'k_dis', 'k_rel', 'k_in',
            'k_out', 'k_iin', 'k_iout', 'k_exp', 'k_txi', 'd_mi', 'k_tli',
            'd_ikb', 'd_mb', 'k_tlb', 'd_bcl'],
    'lit-informed': ['K_sqr_inh', 'q_mito', 'q_xo', 'q_nox', 'tau_nox', 'q_ret',
                     'k_px', 'v_gsh', 'v_cbs', 'f_hyp', 'k_sqr', 'k_a20', 'v_cf'],
    'set': ['n_sqr', 'h_rps3', 'h_exec'],
}
PROVENANCE['lit-informed'].append('q_leak')
PROVENANCE['lit'].append('k_turn')
PROVENANCE['set'].append('n_bcl')
PROVENANCE['est'] = [k for k in NOMINAL
                     if k not in PROVENANCE['lit']
                     and k not in PROVENANCE['lit-informed']
                     and k not in PROVENANCE['set']]

# Parameters carried through estimation, identifiability and sensitivity.
# Restricted to those that are not directly measurable for p65 Cys38.
ESTIMATED = [
    'b_ox', 'b_psulf', 'b_overox', 'b_psox', 'b_glut', 'b_depsulf', 'k_turn',
    'g_SOH', 'g_SSH', 'g_SO2H',
    'k_dam', 'k_rep', 'K_sqr_inh', 'q_ret', 'k_px',
    'v_cbs', 'k_sqr', 'k_ikk', 'K_rps3', 'k_txb', 'v_cf', 'K_bcl', 'K_dam', 'k_exec', 'K_exec', 'w_ox',
]

# log10 prior bounds (multiplicative factor around nominal) for estimated
# kinetic parameters; additive bounds in RT units for barriers/free energies.
PRIOR_LOG_SPAN = 1.0      # one decade either side of nominal
PRIOR_RT_SPAN  = 2.5      # +/- 2.5 RT for barriers and state free energies


def edge_barriers():
    return np.array([e[E_B] for e in EDGES], dtype=float)


def edge_dG():
    return np.array([e[E_DG] for e in EDGES], dtype=float)


def default_theta_vector():
    """Free parameters of the occupancy layer as a flat vector."""
    return np.concatenate([edge_barriers(), G_STATE])
