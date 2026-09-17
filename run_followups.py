"""
run_followups.py
----------------
Analyses that depend on the main run having finished: the absolute
identifiability recomputation, the structural comparison at fitted
parameters, the intervention window at a lethal deprivation, the paired
statistical comparison, and the stress test on the assumption that the
sulfinate is terminal.
"""

import json
import runpy
import time

import numpy as np

import tessera_data as D
import tessera_experiments as E
import tessera_inference as I

OUT = "results.json"


def add_reference_objective():
    with open(OUT) as fh:
        res = json.load(fh)
    ds = D.reference_dataset(seed=E.SEED)
    x0 = I.pack("tessera")[1]
    res["fits"]["_chi2_truth"] = float(2.0 * I.cost(x0, "tessera", ds))
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print("objective at generating parameters: %.1f"
          % res["fits"]["_chi2_truth"], flush=True)


STEPS = [
    ("reference objective", add_reference_objective),
    ("identifiability (absolute)", lambda: runpy.run_path(
        "recompute_identifiability.py", run_name="__main__")),
    ("capability at fitted parameters", lambda: runpy.run_path(
        "run_capability_fitted.py", run_name="__main__")),
    ("intervention window", lambda: runpy.run_path(
        "run_window.py", run_name="__main__")),
    ("paired statistics", lambda: runpy.run_path(
        "run_stats.py", run_name="__main__")),
    ("sulfinate reversal stress test", lambda: runpy.run_path(
        "run_srx_sweep.py", run_name="__main__")),
]


def main():
    for name, fn in STEPS:
        t0 = time.time()
        print("=== %s ===" % name, flush=True)
        try:
            fn()
        except Exception as exc:
            print("  FAILED: %s: %s" % (type(exc).__name__, exc), flush=True)
        print("=== %s done in %.0f s ===" % (name, time.time() - t0), flush=True)
    print("FOLLOWUPS COMPLETE", flush=True)


if __name__ == "__main__":
    main()
