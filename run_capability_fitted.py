"""
run_capability_fitted.py
------------------------
Repeat the structural capability test with every alternative structure first
calibrated against the same in silico observations, so that differences
between structures cannot be attributed to an arbitrary choice of nominal
parameters.  Run after run_all.py has produced the fits.
"""

import json
import time

import numpy as np

import tessera_model as M
import tessera_inference as I
import tessera_experiments as E
import tessera_population as PopMod

OUT = "results.json"


def main():
    with open(OUT) as fh:
        res = json.load(fh)
    if "fits" not in res:
        print("fits not available yet")
        return
    cells = PopMod.draw_population(48, seed=E.SEED + 1)
    out = {}
    for v in ["tessera", "unconstrained", "lumped", "independent", "hill"]:
        if v not in res["fits"]:
            continue
        t0 = time.time()
        x = np.asarray(res["fits"][v]["x"])
        b, g, vpar = I.unpack(v, x)
        fast = E._dose_curve(cells, E.DOSES_FAST, "bolus", 90.,
                             barriers=b, variant=v, vpar=vpar)
        slow = E._dose_curve(cells, E.DOSES_SLOW, "slow", 0.,
                             barriers=b, variant=v, vpar=vpar)
        base = fast[0]
        out[v] = dict(doses_fast=E.DOSES_FAST, viab_fast=fast,
                      doses_slow=E.DOSES_SLOW, viab_slow=slow,
                      baseline=float(base), best_fast=float(max(fast)),
                      best_slow=float(max(slow)),
                      gain_fast=float(max(fast) - base),
                      gain_slow=float(max(slow) - base),
                      slow_advantage=float(max(slow) - max(fast)),
                      biphasic=bool(max(fast) > base + 0.02
                                    and fast[-1] < base - 0.02))
        print("%-14s base %.3f best %.3f gain %.3f biphasic %s (%.0f s)"
              % (v, base, max(fast), max(fast) - base, out[v]["biphasic"],
                 time.time() - t0), flush=True)
    res["capability_fitted"] = out
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print("saved")


if __name__ == "__main__":
    main()
