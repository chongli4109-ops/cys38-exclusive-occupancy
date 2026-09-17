"""
run_all.py
----------
Reproduce every numerical result in the manuscript.  Results are written
incrementally to results.json so a long run can be inspected while in
progress and resumed after interruption.

    python run_all.py            run everything that is not already present
    python run_all.py --force    recompute everything
"""

import json
import os
import sys
import time

import numpy as np

import tessera_data as D
import tessera_experiments as E

OUT = "results.json"


def load():
    if os.path.exists(OUT):
        with open(OUT) as fh:
            return json.load(fh)
    return {}


def save(r):
    with open(OUT, "w") as fh:
        json.dump(r, fh)


def stage(res, key, fn, force=False):
    if key in res and not force:
        print("[skip] %s" % key, flush=True)
        return res[key]
    t0 = time.time()
    print("[run ] %s ..." % key, flush=True)
    res[key] = fn()
    save(res)
    print("[done] %s in %.1f s" % (key, time.time() - t0), flush=True)
    return res[key]


def main():
    force = "--force" in sys.argv
    res = load()
    stage(res, "thermodynamics", E.e1_thermodynamics, force)
    stage(res, "reference", E.e2_reference, force)
    stage(res, "capability", E.e3_capability, force)
    stage(res, "ablation", E.e4_ablation, force)

    print("[run ] dataset ...", flush=True)
    ds = D.reference_dataset(seed=E.SEED)

    fits = stage(res, "fits", lambda: E.e5_fits(ds), force)
    stage(res, "identifiability", lambda: E.e6_identifiability(ds, fits), force)
    stage(res, "profiles", lambda: E.e7_profiles(ds, fits), force)
    stage(res, "commitment", E.e9_commitment, force)
    stage(res, "dose_timing", E.e10_dose_timing, force)
    stage(res, "sobol", lambda: E.e8_sobol(N=256), force)
    stage(res, "robustness", lambda: E.e11_robustness(ds), force)
    stage(res, "cost", E.e12_cost, force)
    save(res)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
