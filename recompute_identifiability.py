"""
recompute_identifiability.py
----------------------------
Recompute the identifiability analysis from the stored fits using the
absolute criterion, and refresh that entry in results.json.  Run after
run_all.py has finished so the two do not write the file concurrently.
"""

import json

import numpy as np

import tessera_data as D
import tessera_experiments as E
import tessera_inference as I

OUT = "results.json"


def main():
    with open(OUT) as fh:
        res = json.load(fh)
    ds = D.reference_dataset(seed=E.SEED)
    panels = {"full": D.FIT_PANEL, "feasible": D.FEASIBLE_PANEL,
              "minimal": D.MINIMAL_PANEL}
    out = {}
    for v in ("tessera", "unconstrained"):
        x = np.asarray(res["fits"][v]["x"])
        names = res["fits"][v]["names"]
        out[v] = {}
        for pname, panel in panels.items():
            S = I.sensitivity_matrix(x, v, ds, panel=panel)
            d = I.identifiability(S, names)
            out[v][pname] = dict(
                n_identifiable=d["n_identifiable"], n_param=d["n_param"],
                condition=float(d["condition"]),
                collinearity=float(d["collinearity"]),
                lambda_min=d["lambda_min"], lambda_max=d["lambda_max"],
                n_well_determined=d["n_well_determined"],
                median_se=d["median_se"],
                eigenvalues=[float(z) for z in d["eigenvalues"]],
                stderr=[float(z) for z in d["stderr"]])
            print("%-14s %-9s ident %2d/%2d  well %2d  med.se %.3f  "
                  "lam_max %.3g" % (v, pname, d["n_identifiable"],
                                    d["n_param"], d["n_well_determined"],
                                    d["median_se"], d["lambda_max"]),
                  flush=True)
    res["identifiability"] = out
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print("saved")


if __name__ == "__main__":
    main()
