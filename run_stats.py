"""
run_stats.py
------------
Statistical comparison of population survival with and without sulfide
delivery.

The simulations are deterministic given a parameter draw, so there is no
measurement noise to test against.  There is, however, genuine sampling
variability: the ensemble is a finite sample from the modelled distribution
of cells, and each cell returns a binary outcome.  The same cells are
simulated under every condition, so the design is paired and McNemar's exact
test is the appropriate procedure.  Risk differences carry bootstrap
intervals over cells.
"""

import json
from math import comb

import numpy as np

import tessera_model as M
import tessera_params as P
import tessera_experiments as E
import tessera_population as PopMod

OUT = "results.json"
N_CELLS = 200
N_BOOT = 20000


def survival_vector(cells, prepared, dose, kind, t_dose, t_ogd=90.):
    prot = M.Protocol(t_ogd=t_ogd, t_end=720., donor_dose=float(dose),
                      donor_time=float(t_dose), donor_kind=kind)
    out = []
    for (p, b), y0 in zip(cells, prepared):
        if y0 is None:
            out.append(np.nan)
            continue
        o = M.simulate(p=p, barriers=b, prot=prot,
                       t_eval=np.array([0., 720.]), y0=y0)
        out.append(np.nan if o is None
                   else float(o[1][M.IDX['D']][-1]) < 0.5)
    return np.array(out, dtype=float)


def mcnemar_exact(a, b):
    """Two-sided exact McNemar test on paired binary outcomes."""
    n01 = int(np.sum((a == 0) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n = n01 + n10
    if n == 0:
        return n01, n10, 1.0
    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
    return n01, n10, float(min(1.0, 2.0 * tail))


def boot_diff(a, b, n_boot=N_BOOT, seed=5):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, a.size, size=(n_boot, a.size))
    d = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    with open(OUT) as fh:
        res = json.load(fh)
    cells = PopMod.draw_population(N_CELLS, seed=E.SEED + 11)
    prepared = []
    for p, b in cells:
        try:
            prepared.append(M.normoxic_steady_state(p, b, P.G_STATE,
                                                    t_relax=2500.))
        except Exception:
            prepared.append(None)

    dt = res.get("dose_timing", {})
    if dt:
        i_f = int(np.argmax(dt["viab_fast"]))
        dose_f = dt["doses_fast"][i_f]
        i_s = int(np.argmax(dt["viab_slow"]))
        dose_s = dt["doses_slow"][i_s]
    else:
        dose_f, dose_s = 200.0, 550.0

    base = survival_vector(cells, prepared, 0.0, "bolus", 90.)
    fast = survival_vector(cells, prepared, dose_f, "bolus", 90.)
    slow = survival_vector(cells, prepared, dose_s, "slow", 0.)
    late = survival_vector(cells, prepared, dose_f, "bolus", 180.)

    out = dict(n_cells=int(base.size), dose_fast=float(dose_f),
               dose_slow=float(dose_s),
               survival=dict(untreated=float(np.nanmean(base)),
                             rapid=float(np.nanmean(fast)),
                             sustained=float(np.nanmean(slow)),
                             rapid_late=float(np.nanmean(late))))
    for label, arr in (("rapid_vs_untreated", fast), ("sustained_vs_untreated", slow),
                       ("late_vs_untreated", late)):
        n01, n10, p = mcnemar_exact(base, arr)
        lo, hi = boot_diff(base, arr)
        out[label] = dict(discordant_gain=n01, discordant_loss=n10, p_value=p,
                          risk_difference=float(np.nanmean(arr) - np.nanmean(base)),
                          ci_low=lo, ci_high=hi)
        print("%-24s gain %3d loss %3d  RD %+0.3f [%+0.3f, %+0.3f]  p = %.2g"
              % (label, n01, n10, out[label]["risk_difference"], lo, hi, p),
              flush=True)
    n01, n10, p = mcnemar_exact(late, fast)
    lo, hi = boot_diff(late, fast)
    out["rapid_vs_late"] = dict(discordant_gain=n01, discordant_loss=n10,
                                p_value=p,
                                risk_difference=float(np.nanmean(fast) - np.nanmean(late)),
                                ci_low=lo, ci_high=hi)
    print("%-24s gain %3d loss %3d  RD %+0.3f [%+0.3f, %+0.3f]  p = %.2g"
          % ("timely_vs_late", n01, n10, out["rapid_vs_late"]["risk_difference"],
             lo, hi, p), flush=True)

    res["statistics"] = out
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print("saved")


if __name__ == "__main__":
    main()
