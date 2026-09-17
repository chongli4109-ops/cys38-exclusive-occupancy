"""
run_window.py
-------------
Intervention window at a deprivation that is lethal without treatment.

The commitment threshold of the untreated cell is located first, and the
protocol is then set a short interval beyond it so that the untreated
outcome is death.  Only under those conditions does the timing of sulfide
delivery decide the outcome, and the latest administration that still
rescues defines the window.  Both the single cell and the population are
reported.
"""

import json

import numpy as np

import tessera_bifurcation as B
import tessera_model as M
import tessera_params as P
import tessera_experiments as E
import tessera_population as PopMod

OUT = "results.json"
N_CELLS = 80


def main():
    with open(OUT) as fh:
        res = json.load(fh)
    p, b, g = dict(P.NOMINAL), P.edge_barriers(), P.G_STATE.copy()
    y0 = M.normoxic_steady_state(p, b, g)
    T0 = res.get("commitment", {}).get("commitment_min") \
        or B.commitment_threshold(p, b, g, y0)
    # Place the protocol between the untreated threshold and the best
    # threshold any delivery achieves, so that the untreated cell dies while
    # a well-timed donor can still rescue it.  Outside that interval the
    # timing question has no content: either everything survives or nothing
    # does.
    dc = res.get("commitment", {}).get("donor_curves", {})
    best = max([max(v["thresholds"]) for v in dc.values()] or [T0 + 4.0])
    t_ogd = float(np.round(0.5 * (T0 + best), 1))
    if t_ogd <= T0:
        t_ogd = T0 + 1.0
    ok, _d = B.survives(t_ogd, p, b, g, y0)
    print("untreated threshold %.1f min; best treated threshold %.1f min; "
          "protocol set to %.1f min (untreated survives: %s)"
          % (T0, best, t_ogd, ok), flush=True)

    times = np.arange(0.0, 241.0, 20.0)
    single = {}
    for kind, dose in (("bolus", 200.0), ("slow", 550.0)):
        latest, curve = np.nan, []
        for tt in times:
            donor = dict(donor_dose=dose, donor_time=float(tt),
                         donor_kind=kind)
            s, d = B.survives(t_ogd, p, b, g, y0, donor)
            curve.append(bool(s))
            if s:
                latest = float(tt)
        single[kind] = dict(dose=dose, times=times.tolist(),
                            survives=curve, latest=latest)
        print("  %-5s dose %.0f: rescues when given up to %s min"
              % (kind, dose, "never" if latest != latest else "%.0f" % latest),
              flush=True)

    cells = PopMod.draw_population(N_CELLS, seed=E.SEED + 7)
    pop = {}
    for kind, dose in (("bolus", 200.0), ("slow", 550.0)):
        vals = [E._dose_curve(cells, [dose], kind, float(tt), t_ogd=t_ogd)[0]
                for tt in times]
        untreated = E._dose_curve(cells, [0.0], kind, 0.0, t_ogd=t_ogd)[0]
        pop[kind] = dict(dose=dose, times=times.tolist(), viability=vals,
                         untreated=float(untreated))
        print("  %-5s population: untreated %.3f, best %.3f at %.0f min"
              % (kind, untreated, max(vals), times[int(np.argmax(vals))]),
              flush=True)

    res["window"] = dict(t_ogd=t_ogd, threshold_untreated=float(T0),
                         threshold_best=float(best),
                         single=single, population=pop)
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print("saved")


if __name__ == "__main__":
    main()
