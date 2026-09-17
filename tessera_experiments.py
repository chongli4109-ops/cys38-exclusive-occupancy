"""
tessera_experiments.py
----------------------
Drivers for every numerical experiment reported in the manuscript.  Each
driver returns a plain dictionary; results are checkpointed to disk by
run_all so that a long session can be resumed.
"""

import json
import os
import time

import numpy as np

import tessera_params as P
import tessera_model as M
import tessera_calibrate as C
import tessera_data as D
import tessera_inference as I
import tessera_population as PopMod
import tessera_bifurcation as B
import tessera_sensitivity as SA

N_CELLS = 80
SEED = 20260915
DOSES_FAST = [0., 25., 50., 100., 200., 400., 800., 1600., 3200.]
DOSES_SLOW = [0., 60., 130., 270., 550., 1100., 2200., 4400., 8800.]


def _nom():
    return dict(P.NOMINAL), P.edge_barriers(), P.G_STATE.copy()


# ----------------------------------------------------------------------
def e1_thermodynamics():
    b = P.edge_barriers()
    u, aff_res = C.equilibrium_activities(b)
    th, boltz, flux, _u = C.equilibrium_check(b)
    ge, pred, spread = C.effective_potential(b, u)
    aff_nom = [lr for _n, lr in C.cycle_report(b, dict(
        P=0.3662, S=9.1, G=5.757, X=0.894, A=0.723, NO=1.0))]
    g2 = P.G_STATE + np.array([0., 1.3, -0.7, 2.1, 0.4, -1.1])
    aff_pert = [lr for _n, lr in C.cycle_report(b, dict(
        P=0.3662, S=9.1, G=5.757, X=0.894, A=0.723, NO=1.0), gstate=g2)]
    return dict(
        n_edges=P.NEDGE, n_states=P.NSTATE,
        n_independent_cycles=P.NEDGE - P.NSTATE + 1,
        n_free_constrained=P.NEDGE + P.NSTATE - 1,
        n_free_unconstrained=2 * P.NEDGE,
        max_cycle_affinity_residual=float(aff_res),
        max_net_cycle_flux=float(flux),
        potential_path_dependence=float(spread),
        boltzmann_deviation=float(np.max(np.abs(pred - th))),
        affinity_invariance=float(np.max(np.abs(np.array(aff_nom)
                                               - np.array(aff_pert)))),
        cycle_affinities_nominal=[float(x) for x in aff_nom],
        effective_potential=[float(x) for x in ge],
    )


def e2_reference():
    p, b, g = _nom()
    y0 = M.normoxic_steady_state(p, b, g)
    t, Y = M.simulate(p=p, barriers=b, gstate=g,
                      prot=M.Protocol(t_ogd=90., t_end=720.),
                      t_eval=np.linspace(0, 720, 361), y0=y0)
    tc, Yc = M.simulate(p=p, barriers=b, gstate=g,
                        prot=M.Protocol(t_ogd=0., t_end=720.),
                        t_eval=np.linspace(0, 720, 361), y0=y0)
    cells = PopMod.draw_population(N_CELLS, seed=SEED)
    cache = {}
    durations = [0., 30., 45., 60., 75., 90., 105., 120., 135., 150., 180.]
    pop = []
    for T in durations:
        m, s, _mt, _tt = PopMod.population_response(
            M.Protocol(t_ogd=T, t_end=720.), cells, y0_cache=cache)
        pop.append((T, float(m), float(s.std())))
    return dict(t=t.tolist(), traj=Y.tolist(), traj_control=Yc.tolist(),
                baseline_occupancy=y0[M.TH].tolist(),
                durations=[x[0] for x in pop],
                pop_viability=[x[1] for x in pop],
                pop_sd=[x[2] for x in pop])


# ----------------------------------------------------------------------
def _dose_curve(cells, doses, kind, t_dose, t_ogd=90., cache=None,
                p=None, barriers=None, variant="tessera", vpar=None):
    out = []
    for d in doses:
        prot = M.Protocol(t_ogd=t_ogd, t_end=720., donor_dose=float(d),
                          donor_time=float(t_dose), donor_kind=kind)
        vals = []
        for pc, bc in cells:
            pc2 = dict(pc) if p is None else {**pc, **p}
            bc2 = bc if barriers is None else barriers
            try:
                y0 = M.normoxic_steady_state(pc2, bc2, P.G_STATE,
                                             t_relax=2500., variant=variant,
                                             vpar=vpar)
                o = M.simulate(p=pc2, barriers=bc2, prot=prot,
                               t_eval=np.array([0., 720.]), y0=y0,
                               variant=variant, vpar=vpar)
            except Exception:
                o = None
            vals.append(np.nan if o is None else 1.0 - float(o[1][M.IDX['D']][-1]))
        out.append(float(np.nanmean(vals)))
    return out


def e3_capability(variants=("tessera", "unconstrained", "lumped",
                            "independent", "hill"), n_cells=48):
    cells = PopMod.draw_population(n_cells, seed=SEED + 1)
    res = {}
    for v in variants:
        vpar = M.default_vpar(v)
        fast = _dose_curve(cells, DOSES_FAST, "bolus", 90., variant=v, vpar=vpar)
        slow = _dose_curve(cells, DOSES_SLOW, "slow", 0., variant=v, vpar=vpar)
        base = fast[0]
        res[v] = dict(
            doses_fast=DOSES_FAST, viab_fast=fast,
            doses_slow=DOSES_SLOW, viab_slow=slow,
            baseline=base,
            biphasic=bool(max(fast) > base + 0.02 and fast[-1] < base - 0.02),
            best_fast=float(max(fast)), best_slow=float(max(slow)),
            slow_advantage=float(max(slow) - max(fast)),
        )
    return res


def e4_ablation(n_cells=48):
    """Remove one mechanism at a time and retest capability."""
    cells = PopMod.draw_population(n_cells, seed=SEED + 2)
    eng = [e[P.E_NAME] for e in P.EDGES]
    specs = {
        "full": (None, None),
        "no_persulfide_oxidation": ({}, {"b_psox": 14.0}),
        "no_sulfide_respiratory_inhibition": ({"K_sqr_inh": 1e6}, None),
        "no_RPS3_gate": ({"K_rps3": 1e-9}, None),
        "no_energy_gated_repair": ({"k_rep": P.NOMINAL['k_rep'] * 0.0 + 0.02,
                                    "K_E": 1e-9}, None),
        "no_site_turnover": ({"k_turn": 1e-6}, None),
    }
    out = {}
    for name, (pdelta, bdelta) in specs.items():
        b = P.edge_barriers().copy()
        if bdelta:
            for k, v in bdelta.items():
                b[eng.index(k[2:])] = v
        p = dict(pdelta) if pdelta else None
        fast = _dose_curve(cells, DOSES_FAST, "bolus", 90., p=p, barriers=b)
        base = fast[0]
        pp, bb, gg = _nom()
        if pdelta:
            pp.update(pdelta)
        y0 = M.normoxic_steady_state(pp, b, gg)
        T = B.commitment_threshold(pp, b, gg, y0)
        o = M.simulate(p=pp, barriers=b, gstate=gg,
                       prot=M.Protocol(t_ogd=90., t_end=720.),
                       t_eval=np.linspace(0, 720, 121), y0=y0)
        out[name] = dict(baseline=float(base), best=float(max(fast)),
                         gain=float(max(fast) - base),
                         biphasic=bool(max(fast) > base + 0.02
                                       and fast[-1] < base - 0.02),
                         commitment_min=float(T),
                         peak_sulfinate=float(o[1][M.IDX['th3']].max()),
                         viab_curve=fast)
    return out


# ----------------------------------------------------------------------
def e5_fits(dataset, variants=("tessera", "unconstrained", "lumped",
                               "independent", "hill"), max_nfev=200):
    n_data = dataset['n_rep'] * len(D.SAMPLE_TIMES) * len(D.FIT_PANEL) \
        * len(D.conditions())
    out = {}
    for v in variants:
        t0 = time.time()
        f = I.fit(v, dataset, max_nfev=max_nfev, perturb=0.25, seed=3)
        ic = I.information_criteria(f['cost'], n_data, len(f['names']))
        out[v] = dict(cost=f['cost'], names=f['names'], x=f['x'].tolist(),
                      n_param=len(f['names']), nfev=f['nfev'],
                      seconds=time.time() - t0, **ic)
    out['_n_data'] = n_data
    return out


def e6_identifiability(dataset, fits, panels=None):
    panels = {"full": D.FIT_PANEL, "feasible": D.FEASIBLE_PANEL,
              "minimal": D.MINIMAL_PANEL} if panels is None else panels
    out = {}
    for v in ("tessera", "unconstrained"):
        x = np.asarray(fits[v]['x'])
        names = fits[v]['names']
        out[v] = {}
        for pname, panel in panels.items():
            S = I.sensitivity_matrix(x, v, dataset, panel=panel)
            d = I.identifiability(S, names)
            out[v][pname] = dict(
                n_identifiable=d['n_identifiable'], condition=float(d['condition']),
                collinearity=float(d['collinearity']), n_param=d['n_param'],
                eigenvalues=[float(z) for z in d['eigenvalues']],
                stderr=[float(z) for z in d['stderr']])
    return out


def e7_profiles(dataset, fits, which=("b_psulf",), n=3, max_nfev=6,
                enabled=False):
    """
    Exact profile likelihood.

    Disabled by default.  Each point of a profile requires a constrained
    re-optimisation of the remaining parameters, and at barrier values away
    from the optimum the underlying system becomes stiff enough that a single
    objective evaluation costs tens of seconds, which placed a complete
    profile for both parameterisations beyond the compute available here.
    The identifiability conclusions rest on the Fisher information analysis
    instead.  Set enabled=True to compute profiles where the budget allows.
    """
    if not enabled:
        return {}
    out = {}
    for v in ("tessera", "unconstrained"):
        names = fits[v]['names']
        x = np.asarray(fits[v]['x'])
        out[v] = {}
        for nm in which:
            cand = [nm, "k_%s_f" % nm[2:]]
            idx = next((names.index(c) for c in cand if c in names), None)
            if idx is None:
                continue
            gr, pr, base = I.profile_likelihood(v, dataset, idx, x, n=n,
                                                max_nfev=max_nfev)
            out[v][nm] = dict(grid=gr.tolist(), profile=pr.tolist(),
                              base=float(base))
    return out


def e8_sobol(N=256):
    prob, X, Y, res = SA.run_sobol(N=N)
    return dict(names=prob['names'],
                results={k: {kk: vv.tolist() for kk, vv in v.items()}
                         for k, v in res.items()},
                n_samples=int(X.shape[0]))


def e9_commitment():
    p, b, g = _nom()
    y0 = M.normoxic_steady_state(p, b, g)
    folds = B.saddle_node_points(p)
    us, rec, _f = B.continuation(p, n=300)
    lo = [(u, sorted(r)[0] if r else np.nan) for u, r in rec]
    hi = [(u, sorted(r)[-1] if r else np.nan) for u, r in rec]
    mid = [(u, sorted(r)[1] if len(r) == 3 else np.nan) for u, r in rec]
    T0 = B.commitment_threshold(p, b, g, y0)
    curves = {}
    for kind, doses, td in (("bolus", [0., 100., 200., 400., 800.], 90.),
                            ("slow", [0., 270., 550., 1100., 2200.], 0.)):
        vals = []
        for d in doses:
            donor = dict(donor_dose=float(d), donor_time=td, donor_kind=kind)
            vals.append(float(B.commitment_threshold(p, b, g, y0, donor=donor)))
        curves[kind] = dict(doses=doses, thresholds=vals)
    rb = B.rescue_boundary(p, b, g, y0, [100., 200., 400., 800.], "bolus")
    return dict(saddle_nodes=[float(x) for x in folds],
                u_grid=[float(u) for u, _ in lo],
                branch_low=[float(x) for _, x in lo],
                branch_mid=[float(x) for _, x in mid],
                branch_high=[float(x) for _, x in hi],
                commitment_min=float(T0), donor_curves=curves,
                rescue_latest={str(k): (None if np.isnan(v) else float(v))
                               for k, v in rb.items()})


def e10_dose_timing(n_cells=None):
    n_cells = N_CELLS if n_cells is None else n_cells
    cells = PopMod.draw_population(n_cells, seed=SEED + 4)
    fast = _dose_curve(cells, DOSES_FAST, "bolus", 90.)
    slow = _dose_curve(cells, DOSES_SLOW, "slow", 0.)
    times = [0., 30., 60., 75., 90., 105., 120., 150., 180., 240.]
    tcurve = []
    for tt in times:
        v = _dose_curve(cells, [200.], "bolus", tt)
        tcurve.append(v[0])
    return dict(doses_fast=DOSES_FAST, viab_fast=fast,
                doses_slow=DOSES_SLOW, viab_slow=slow,
                times=times, viab_timing=tcurve)


def e11_robustness(dataset):
    """Noise level and structural misspecification."""
    out = {"noise": {}, "misspecified": {}}
    for scale in (0.5, 1.0, 2.0):
        old = dict(D.NOISE)
        try:
            for k in D.NOISE:
                D.NOISE[k] = old[k] * scale
            ds = D.reference_dataset(seed=SEED + int(100 * scale))
            f = I.fit("tessera", ds, max_nfev=120, perturb=0.2, seed=5)
            xt = I.pack("tessera")[1]
            err = float(np.sqrt(np.mean((np.asarray(f['x']) - xt) ** 2)))
            out["noise"][str(scale)] = dict(cost=f['cost'], rmse_param=err)
        finally:
            for k in old:
                D.NOISE[k] = old[k]
    # data generated with an altered oxidant phasing the model does not know
    p, b, g = _nom()
    p2 = dict(p); p2['tau_nox'] = p['tau_nox'] * 2.2; p2['q_xo'] = p['q_xo'] * 1.8
    y0 = M.normoxic_steady_state(p2, b, g)
    rng = np.random.default_rng(SEED + 9)
    data = {}
    for name, kw in D.conditions():
        ob = D.forward(p2, b, g, kw, y0=y0)
        rec = {}
        for key in D.OBSERVABLES:
            mu = np.maximum(np.asarray(ob[key], float), 1e-9)
            s = np.sqrt(np.log1p(D.NOISE[key] ** 2))
            rec[key] = mu[None, :] * np.exp(rng.normal(-.5 * s * s, s, (3, mu.size)))
        data[name] = rec
    ds2 = dict(times=D.SAMPLE_TIMES, data=data, truth={}, n_rep=3)
    f = I.fit("tessera", ds2, max_nfev=120, perturb=0.2, seed=6)
    bb, gg, vp = I.unpack("tessera", np.asarray(f['x']))
    y0b = M.normoxic_steady_state(dict(P.NOMINAL), bb, gg)
    T = B.commitment_threshold(dict(P.NOMINAL), bb, gg, y0b)
    out["misspecified"] = dict(cost=f['cost'], commitment_min=float(T))
    return out


def e12_cost():
    p, b, g = _nom()
    y0 = M.normoxic_steady_state(p, b, g)
    t0 = time.time()
    for _ in range(20):
        M.simulate(p=p, barriers=b, gstate=g,
                   prot=M.Protocol(t_ogd=90., t_end=720.),
                   t_eval=np.linspace(0, 720, 121), y0=y0)
    per = (time.time() - t0) / 20
    t0 = time.time()
    C.equilibrium_check(b)
    eq = time.time() - t0
    return dict(seconds_per_trajectory=per, seconds_equilibrium_check=eq,
                n_states=M.NVAR, n_occupancy_states=P.NSTATE, n_edges=P.NEDGE)
