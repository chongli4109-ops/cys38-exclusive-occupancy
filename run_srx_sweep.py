"""
run_srx_sweep.py
----------------
Stress the assumption on which the commitment result rests.

The model treats the sulfinate of Cys38 as terminal, because the only
characterised enzymatic reversal of a cysteine sulfinic acid acts on the
2-Cys peroxiredoxins and no activity toward p65 has been reported.  If that
assumption is wrong the terminal branch is not terminal.  The barrier on the
reduction edge is swept from effectively infinite to fast, and the
commitment threshold, peak sulfinate occupancy and rescue window are
recorded as functions of the resulting turnover.
"""

import json

import numpy as np

import tessera_bifurcation as B
import tessera_model as M
import tessera_params as P

OUT = "results.json"


def main():
    with open(OUT) as fh:
        res = json.load(fh)
    eng = [e[P.E_NAME] for e in P.EDGES]
    i_srx = eng.index("srx")
    b0 = P.edge_barriers()
    g = P.G_STATE

    # effective first-order rate out of the sulfinate at resting activities
    def srx_rate(b):
        return float(np.exp(g[P.S_SO2H] - b) * 0.9 * 0.72 / (P.K_SRX + 0.72))

    barriers_grid = [11.5, 9.0, 7.5, 6.0, 5.0, 4.0, 3.0, 2.0]
    rows = []
    for bv in barriers_grid:
        b = b0.copy()
        b[i_srx] = bv
        p = dict(P.NOMINAL)
        y0 = M.normoxic_steady_state(p, b, g)
        out = M.simulate(p=p, barriers=b, gstate=g,
                         prot=M.Protocol(t_ogd=90., t_end=720.),
                         t_eval=np.linspace(0, 720, 241), y0=y0)
        peak = float(out[1][M.IDX['th3']].max())
        end = float(out[1][M.IDX['th3']][-1])
        T = B.commitment_threshold(p, b, g, y0)
        donor = dict(donor_dose=200.0, donor_time=90.0, donor_kind="bolus")
        Td = B.commitment_threshold(p, b, g, y0, donor=donor)
        k = srx_rate(bv)
        rows.append(dict(barrier=float(bv), rate=k,
                         half_life=float(np.log(2) / k) if k > 0 else float('inf'),
                         peak_sulfinate=peak, end_sulfinate=end,
                         commitment_min=float(T), commitment_donor=float(Td),
                         resting_occupancy=float(y0[M.IDX['th3']])))
        print("b_srx %5.2f  k %.3g /min  t1/2 %9.1f min  peak SO2H %.3f  "
              "T* %.0f  T*(donor) %.0f"
              % (bv, k, rows[-1]['half_life'], peak, T, Td), flush=True)
    res["srx_sweep"] = rows
    with open(OUT, "w") as fh:
        json.dump(res, fh)
    print("saved")


if __name__ == "__main__":
    main()
