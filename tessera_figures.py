# -*- coding: utf-8 -*-
"""
tessera_figures.py
------------------
Publication figures.  Everything is rendered in grayscale so that the
figures survive monochrome reproduction; series are distinguished by line
style, marker and hatch rather than by colour.  Legends are placed outside
the data area wherever a legend could otherwise overlap plotted elements.
"""

import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

FIGDIR = "figures"
COL = 3.45          # single column width in inches
DCOL = 7.16         # full text width in inches

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
    "font.size": 7.2,
    "axes.labelsize": 7.6,
    "axes.titlesize": 7.8,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.6,
    "axes.linewidth": 0.6,
    "grid.linewidth": 0.35,
    "lines.linewidth": 1.0,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "axes.grid": False,
})

GRAY = ["0.0", "0.35", "0.55", "0.72", "0.45", "0.2"]
LS = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 1, 1, 1, 1))]
MK = ["o", "s", "^", "D", "v", "P", "X", "*"]
HATCH = ["", "///", "...", "xxx", "\\\\\\", "|||", "+++"]


def _save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
    return path


def _clean(ax, grid=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid:
        ax.grid(True, color="0.88", linestyle="-", linewidth=0.35)
        ax.set_axisbelow(True)


# ----------------------------------------------------------------------
def fig1_mechanism():
    """Exclusive occupancy graph of Cys38 with the sulfenic hub."""
    fig, ax = plt.subplots(figsize=(DCOL, 2.85))
    ax.set_xlim(0, 10.9); ax.set_ylim(0, 5.55); ax.axis("off")

    nodes = {
        "SH":   (1.30, 2.95, "Cys38\u2013SH\nthiolate", "1.00"),
        "SOH":  (4.30, 2.95, "Cys38\u2013SOH\nsulfenic", "0.88"),
        "SSH":  (7.55, 4.05, "Cys38\u2013SSH\npersulfide", "0.94"),
        "SO2H": (7.55, 1.60, "Cys38\u2013SO$_2$H\nsulfinate", "0.60"),
        "SSG":  (4.30, 1.05, "Cys38\u2013SSG", "0.90"),
        "SNO":  (1.30, 1.05, "Cys38\u2013SNO", "0.90"),
    }
    W, H = 1.72, 0.84
    for k, (x, y, lab, shade) in nodes.items():
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H,
                                    boxstyle="round,pad=0.045,rounding_size=0.07",
                                    linewidth=0.9, edgecolor="0.1",
                                    facecolor=shade, zorder=3))
        ax.text(x, y, lab, ha="center", va="center", fontsize=6.6, zorder=4)

    def arrow(p1, p2, lw=0.9, ls="-", rad=0.0, col="0.15"):
        ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>",
                                     mutation_scale=7, linewidth=lw,
                                     linestyle=ls, color=col,
                                     connectionstyle="arc3,rad=%.2f" % rad,
                                     shrinkA=1, shrinkB=1, zorder=2))

    xs, xo, xr = nodes["SH"][0], nodes["SOH"][0], nodes["SSH"][0]
    # thiol <-> sulfenic
    arrow((xs + W / 2, 3.12), (xo - W / 2, 3.12))
    arrow((xo - W / 2, 2.78), (xs + W / 2, 2.78), ls="--")
    # sulfenic -> persulfide and back
    arrow((xo + W / 2, 3.22), (xr - W / 2, 3.78))
    arrow((xr - W / 2 - 0.14, 3.96), (xo + W / 2 - 0.02, 3.34), ls="--")
    # sulfenic -> sulfinate (heavy: the terminal branch)
    arrow((xo + W / 2, 2.68), (xr - W / 2, 1.90), lw=1.7)
    # persulfide -> thiol, routed above every box
    ax.add_patch(FancyArrowPatch((xr - W / 2, 4.32), (xs, 3.37),
                                 arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.9, linestyle="-.", color="0.15",
                                 connectionstyle="arc3,rad=0.30",
                                 shrinkA=1, shrinkB=1, zorder=2))
    # glutathionylation and nitrosylation
    arrow((xo - 0.28, 2.53), (xo - 0.28, 1.47))
    arrow((xo - W / 2, 1.20), (xs + W / 2 - 0.10, 2.55), ls="-.")
    arrow((xs - 0.30, 2.53), (xs - 0.30, 1.47))
    arrow((xs + 0.34, 1.47), (xs + 0.34, 2.53), ls="-.")

    ax.text(2.80, 3.30, "H$_2$O$_2$", fontsize=6.4, ha="center")
    ax.text(2.80, 2.50, "Trx", fontsize=6.4, ha="center")
    ax.text(5.72, 3.76, "H$_2$S", fontsize=6.4, ha="center", va="center")
    ax.text(6.02, 3.14, "H$_2$O$_2$", fontsize=6.4, ha="center", va="center")
    ax.text(5.78, 2.02, "H$_2$O$_2$", fontsize=6.4, ha="center", va="center")
    ax.text(4.08, 5.22, "Trx", fontsize=6.4, ha="center")
    ax.text(3.58, 1.95, "GSH", fontsize=6.2, ha="right", va="center")
    ax.text(0.64, 1.95, "NO", fontsize=6.2, ha="right", va="center")

    ax.text(7.55, 4.92, "reversible: the reservoir", fontsize=6.5,
            ha="center", style="italic")
    ax.add_patch(Rectangle((8.72, 1.06), 2.04, 1.08, linewidth=0.7,
                           edgecolor="0.35", facecolor="none",
                           linestyle=(0, (3, 2)), zorder=1))
    ax.text(9.74, 1.60, "terminal: no known\nreductase acts on\np65 Cys38",
            fontsize=6.2, ha="center", va="center")
    ax.annotate("", xy=(8.70, 1.60), xytext=(8.43, 1.60),
                arrowprops=dict(arrowstyle="-|>", linewidth=0.8, color="0.35"))
    ax.text(5.45, 0.16, "every route competes for one sulfur atom: occupancy "
                        "is mutually exclusive",
            fontsize=6.6, ha="center", style="italic")
    return _save(fig, "fig1_mechanism.png")


# ----------------------------------------------------------------------
def fig2_architecture():
    fig, ax = plt.subplots(figsize=(DCOL, 2.25))
    ax.set_xlim(0, 11.4); ax.set_ylim(0, 3.5); ax.axis("off")
    boxes = [
        (1.25, "Oxygen–glucose\ndeprivation", "energy charge, oxidant\nphases, thiol pools", "0.93"),
        (3.95, "Exclusive site\noccupancy", "six states of Cys38 on\nthe probability simplex", "0.80"),
        (6.65, "NF-κB module", "IKK / IκBα shuttling,\ncompetence-weighted output", "0.93"),
        (9.35, "Survival decision", "Bcl-xL, caspase feedback,\nirreversible execution", "0.93"),
    ]
    W, H = 2.34, 1.42
    for x, title, sub, shade in boxes:
        ax.add_patch(FancyBboxPatch((x - W / 2, 1.30), W, H,
                                    boxstyle="round,pad=0.05,rounding_size=0.08",
                                    linewidth=0.9, edgecolor="0.1",
                                    facecolor=shade))
        ax.text(x, 2.42, title, ha="center", va="center", fontsize=7.4,
                fontweight="bold")
        ax.text(x, 1.79, sub, ha="center", va="center", fontsize=6.3)
    for i in range(3):
        x1 = boxes[i][0] + W / 2
        x2 = boxes[i + 1][0] - W / 2
        ax.add_patch(FancyArrowPatch((x1, 2.01), (x2, 2.01), arrowstyle="-|>",
                                     mutation_scale=8, linewidth=1.0,
                                     color="0.12"))
    labels = ["H$_2$O$_2$, H$_2$S, Trx,\nATP, glutathione",
              "DNA-binding competence,\nRPS3 recruitment gate",
              "Bcl-xL abundance"]
    for i, lab in enumerate(labels):
        xm = 0.5 * (boxes[i][0] + boxes[i + 1][0])
        ax.text(xm, 2.20, lab, ha="center", va="bottom", fontsize=6.0)
    ax.add_patch(FancyArrowPatch((9.35, 1.28), (1.25, 0.62),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9, linestyle="--", color="0.4",
                                 connectionstyle="arc3,rad=0.11"))
    ax.text(5.30, 0.28, "mitochondrial damage feeds back on energy charge "
                        "and oxidant clearance",
            ha="center", fontsize=6.3, style="italic", color="0.25")
    return _save(fig, "fig2_architecture.png")


# ----------------------------------------------------------------------
def fig3_reference(res):
    r = res["reference"]
    t = np.array(r["t"]) / 60.0
    Y = np.array(r["traj"])
    t_reox = 1.5
    fig, axes = plt.subplots(2, 2, figsize=(DCOL, 4.05))

    ax = axes[0, 0]
    series = [(0, "ATP charge", 1.0), (1, "mitochondrial pool", 1.0),
              (2, "H$_2$O$_2$ ($\\mu$M)/10", 0.1),
              (4, "sulfide ($\\mu$M)/40", 0.025), (5, "reduced Trx", 1.0)]
    for k, (i, lab, sc) in enumerate(series):
        ax.plot(t, Y[i] * sc, color=GRAY[k % 6], linestyle=LS[k % 6],
                label=lab)
    ax.axvspan(0, t_reox, color="0.91", zorder=0, linewidth=0)
    ax.set_xlabel("time (h)"); ax.set_ylabel("normalised level")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1.32)
    ax.set_title("(a) upstream state", fontsize=7.2, pad=2)
    _clean(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.34), ncol=2,
              frameon=False, handlelength=2.0, columnspacing=1.0,
              labelspacing=0.28)

    ax = axes[0, 1]
    labels = ["SH", "SOH", "SSH", "SO$_2$H", "SSG", "SNO"]
    shades = ["0.97", "0.86", "0.74", "0.60", "0.44", "0.26"]
    base = np.zeros_like(t)
    for i in range(6):
        top = base + Y[6 + i]
        ax.fill_between(t, base, top, facecolor=shades[i], edgecolor="0.12",
                        linewidth=0.35, hatch=HATCH[i], label=labels[i])
        base = top
    ax.axvline(t_reox, color="0.05", linewidth=0.9, linestyle=(0, (4, 2)))
    ax.text(t_reox + 0.25, 0.035, "restoration", fontsize=6.2, ha="left",
            va="bottom", color="0.05")
    ax.set_xlabel("time (h)"); ax.set_ylabel("Cys38 occupancy")
    ax.set_xlim(0, 12); ax.set_ylim(0, 1)
    ax.set_title("(b) site occupancy", fontsize=7.2, pad=2)
    _clean(ax, grid=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.48, -0.34), ncol=3,
              frameon=False, handlelength=1.5, columnspacing=0.8,
              labelspacing=0.28, handleheight=1.1)

    ax = axes[1, 0]
    for k, (i, lab, sc) in enumerate([(14, "nuclear NF-$\\kappa$B", 4.0),
                                      (22, "Bcl-xL", 1.0),
                                      (23, "active caspase", 0.30),
                                      (24, "executed fraction", 1.0)]):
        ax.plot(t, Y[i] * sc, color=GRAY[k], linestyle=LS[k], label=lab)
    ax.axvspan(0, t_reox, color="0.91", zorder=0, linewidth=0)
    ax.set_xlabel("time (h)"); ax.set_ylabel("normalised level")
    ax.set_xlim(0, 12)
    ax.set_title("(c) transcriptional and decision layers", fontsize=7.2, pad=2)
    _clean(ax)
    ax.legend(loc="upper right", frameon=False, handlelength=2.0,
              labelspacing=0.28, borderaxespad=0.2)

    ax = axes[1, 1]
    d = np.array(r["durations"]); v = np.array(r["pop_viability"])
    sd = np.array(r["pop_sd"])
    ax.errorbar(d, v, yerr=sd / np.sqrt(80), color="0.1", marker="o",
                markersize=2.8, linewidth=1.0, capsize=1.8,
                markerfacecolor="white", markeredgewidth=0.7)
    ax.set_xlabel("deprivation duration (min)")
    ax.set_ylabel("surviving fraction")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title("(d) population response", fontsize=7.2, pad=2)
    _clean(ax)
    fig.subplots_adjust(hspace=0.78, wspace=0.40)
    return _save(fig, "fig3_reference.png")


# ----------------------------------------------------------------------
def fig4_capability(res):
    cap = res.get("capability_fitted") or res["capability"]
    order = ["tessera", "unconstrained", "independent", "lumped", "hill"]
    names = {"tessera": "Exclusive, thermodynamically constrained",
             "unconstrained": "Exclusive, unconstrained",
             "independent": "Independent modifications",
             "lumped": "Lumped oxidation",
             "hill": "Phenomenological modifier"}
    fig, axes = plt.subplots(1, 2, figsize=(DCOL, 2.15))
    for ax, key, dk, ttl in ((axes[0], "viab_fast", "doses_fast",
                              "rapid-release donor"),
                             (axes[1], "viab_slow", "doses_slow",
                              "slow-release donor")):
        for i, v in enumerate(order):
            if v not in cap:
                continue
            d = np.array(cap[v][dk]); y = np.array(cap[v][key])
            x = np.arange(len(d))
            ax.plot(x, y, color=GRAY[i % 6], linestyle=LS[i % 6],
                    marker=MK[i], markersize=2.5, markerfacecolor="white",
                    markeredgewidth=0.6, label=names[v])
            ax.set_xticks(x)
            ax.set_xticklabels([("0" if z == 0 else "%g" % z) for z in d],
                               rotation=45, ha="right")
        ax.set_xlabel("donor dose (µM equivalent)")
        ax.set_title(ttl, pad=3)
        _clean(ax)
    axes[0].set_ylabel("surviving fraction")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.09), ncol=2,
               frameon=False, handlelength=2.4, columnspacing=1.4,
               labelspacing=0.3)
    fig.subplots_adjust(wspace=0.28, bottom=0.34)
    return _save(fig, "fig4_capability.png")


# ----------------------------------------------------------------------
def fig5_ablation(res):
    ab = res["ablation"]
    keys = [k for k in ab if k != "full"]
    labels = {"no_persulfide_oxidation": "persulfide\noxidation",
              "no_sulfide_respiratory_inhibition": "respiratory\ninhibition",
              "no_RPS3_gate": "RPS3\ngate",
              "no_energy_gated_repair": "energy-gated\nsynthesis",
              "no_site_turnover": "site\nturnover"}
    fig, ax = plt.subplots(figsize=(COL, 1.80))
    x = np.arange(len(keys))
    gains = [ab[k]["gain"] for k in keys]
    ref = ab["full"]["gain"]
    ax.bar(x, gains, width=0.58, facecolor="0.82", edgecolor="0.1",
           linewidth=0.7, hatch="///")
    ax.axhline(ref, color="0.1", linestyle="--", linewidth=0.9)
    ax.text(len(keys) - 0.45, ref * 1.03, "complete model", fontsize=5.8,
            ha="right", va="bottom")
    for xi, gg in zip(x, gains):
        ax.text(xi, max(gg, 0) + ref * 0.04, "%.3f" % gg, fontsize=5.2,
                ha="center", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[k] for k in keys], fontsize=5.4)
    ax.set_ylabel("maximum survival gain", fontsize=6.8)
    ax.set_ylim(0, ref * 1.38)
    _clean(ax)
    return _save(fig, "fig5_ablation.png")


# ----------------------------------------------------------------------
def fig6_identifiability(res):
    idn = res["identifiability"]
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    for i, v in enumerate(["tessera", "unconstrained"]):
        ev = np.array(idn[v]["full"]["eigenvalues"], dtype=float)
        ax.semilogy(np.arange(1, ev.size + 1), np.maximum(ev, 1e-16),
                    color=GRAY[i], linestyle=LS[i], marker=MK[i],
                    markersize=2.5, markerfacecolor="white",
                    markeredgewidth=0.6,
                    label=("constrained, %d parameters" % ev.size
                           if v == "tessera"
                           else "unconstrained, %d parameters" % ev.size))
    ax.axhline(1.0, color="0.45", linewidth=0.8, linestyle=":")
    ax.text(ev.size * 0.98, 1.6, "one unit of RT resolved", fontsize=5.8,
            color="0.3", va="bottom", ha="right")
    ax.set_xlabel("eigenvalue index", fontsize=6.8)
    ax.set_ylabel("eigenvalue of $\\mathcal{F}$", fontsize=6.8)
    _clean(ax)
    ax.legend(loc="lower left", frameon=False, handlelength=2.0,
              fontsize=6.0, labelspacing=0.25)
    return _save(fig, "fig6_identifiability.png")


# ----------------------------------------------------------------------
def fig7_profiles(res):
    pr = res["profiles"]
    keys = sorted(pr["tessera"].keys())
    n = len(keys)
    fig, axes = plt.subplots(1, max(n, 2), figsize=(DCOL, 2.05),
                             sharey=True)
    if n == 1:
        axes = [axes]
    nice = {"b_ox": r"$b_{\mathrm{ox}}$", "b_psulf": r"$b_{\mathrm{psulf}}$",
            "b_overox": r"$b_{\mathrm{overox}}$", "b_psox": r"$b_{\mathrm{psox}}$"}
    for j, k in enumerate(keys):
        ax = axes[j]
        for i, v in enumerate(["tessera", "unconstrained"]):
            if k not in pr.get(v, {}):
                continue
            g = np.array(pr[v][k]["grid"]); p = np.array(pr[v][k]["profile"])
            base = pr[v][k]["base"]
            ax.plot(g - g[len(g) // 2], 2 * (p - base), color=GRAY[i],
                    linestyle=LS[i], marker=MK[i], markersize=2.2,
                    markerfacecolor="white", markeredgewidth=0.5,
                    label=("constrained" if v == "tessera" else "unconstrained"))
        ax.axhline(3.84, color="0.45", linewidth=0.7, linestyle=":")
        ax.set_xlabel(nice.get(k, k) + " offset (RT)")
        _clean(ax)
    axes[0].set_ylabel("$2\\,\\Delta$ negative log-likelihood")
    axes[0].set_ylim(bottom=-0.5)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=2,
               frameon=False, handlelength=2.2)
    fig.subplots_adjust(wspace=0.22, bottom=0.36)
    return _save(fig, "fig7_profiles.png")


# ----------------------------------------------------------------------
def fig8_commitment(res):
    cm = res["commitment"]
    fig, ax = plt.subplots(figsize=(COL, 1.85))
    u = np.array(cm["u_grid"])
    lo = np.array(cm["branch_low"], dtype=float)
    mid = np.array(cm["branch_mid"], dtype=float)
    hi = np.array(cm["branch_high"], dtype=float)
    ax.plot(u, lo, color="0.1", linestyle="-", label="stable")
    ax.plot(u, hi, color="0.1", linestyle="-")
    ax.plot(u, mid, color="0.45", linestyle="--", label="unstable")
    for fv in cm["saddle_nodes"]:
        ax.axvline(fv, color="0.55", linewidth=0.6, linestyle=":")
        ax.text(fv, 1.4e-3, "fold", fontsize=5.8, ha="center", va="bottom")
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 60)
    ax.set_xlabel("composite death drive $u$", fontsize=6.8)
    ax.set_ylabel("caspase activity", fontsize=6.8)
    _clean(ax)
    ax.legend(loc="lower right", frameon=False, handlelength=2.0,
              fontsize=6.2)
    return _save(fig, "fig8_commitment.png")


# ----------------------------------------------------------------------
def fig9_dose_timing(res):
    dt = res["dose_timing"]
    wn = res.get("window")
    ncol = 3 if wn else 2
    fig, axes = plt.subplots(1, ncol, figsize=(DCOL, 2.10))

    ax = axes[0]
    for i, (dk, vk, lab) in enumerate([("doses_fast", "viab_fast", "rapid release"),
                                       ("doses_slow", "viab_slow", "slow release")]):
        d = np.array(dt[dk], dtype=float); y = np.array(dt[vk], dtype=float)
        ax.plot(d, y, color=GRAY[i], linestyle=LS[i], marker=MK[i],
                markersize=2.6, markerfacecolor="white", markeredgewidth=0.6,
                label=lab)
    ax.set_xscale("symlog", linthresh=50.0, linscale=0.5)
    ax.set_xticks([0, 100, 1000, 10000])
    ax.set_xticklabels(["0", "$10^2$", "$10^3$", "$10^4$"])
    ax.minorticks_off()
    ax.axhline(dt["viab_fast"][0], color="0.55", linewidth=0.7, linestyle=":")
    ax.text(0.0, dt["viab_fast"][0] + 0.004, "untreated", fontsize=5.9,
            ha="left", va="bottom")
    ax.set_xlabel("delivered dose (\u00b5M equiv.)")
    ax.set_ylabel("surviving fraction")
    ax.set_title("(a) dose, survivable insult", fontsize=6.8, pad=3)
    _clean(ax)
    ax.legend(loc="lower left", frameon=False, handlelength=1.9,
              labelspacing=0.25)

    ax = axes[1]
    tt = np.array(dt["times"], dtype=float); y = np.array(dt["viab_timing"],
                                                          dtype=float)
    ax.plot(tt, y, color="0.1", linestyle="-", marker="o", markersize=2.6,
            markerfacecolor="white", markeredgewidth=0.7)
    ax.axvline(90, color="0.35", linewidth=0.8, linestyle="--")
    ax.annotate("restoration", xy=(90, y.min()), xytext=(112, y.min()),
                fontsize=5.9, va="center", ha="left",
                arrowprops=dict(arrowstyle="->", linewidth=0.6, color="0.4"))
    ax.axhline(dt["viab_fast"][0], color="0.55", linewidth=0.7, linestyle=":")
    ax.set_xlabel("time of delivery (min)")
    ax.set_ylabel("surviving fraction")
    ax.set_title("(b) timing, survivable insult", fontsize=6.8, pad=3)
    _clean(ax)

    if wn:
        ax = axes[2]
        for i, (kind, lab) in enumerate([("bolus", "rapid release"),
                                         ("slow", "slow release")]):
            q = wn["population"].get(kind)
            if not q:
                continue
            t = np.array(q["times"], dtype=float)
            v = np.array(q["viability"], dtype=float)
            ax.plot(t, v, color=GRAY[i], linestyle=LS[i], marker=MK[i],
                    markersize=2.6, markerfacecolor="white",
                    markeredgewidth=0.6, label=lab)
            ax.axhline(q["untreated"], color="0.55", linewidth=0.7,
                       linestyle=":")
        ax.axvline(wn["t_ogd"], color="0.35", linewidth=0.8, linestyle="--")
        ax.annotate("restoration", xy=(wn["t_ogd"], ax.get_ylim()[0]),
                    xytext=(wn["t_ogd"] + 22, ax.get_ylim()[0]),
                    fontsize=5.9, va="bottom", ha="left",
                    arrowprops=dict(arrowstyle="->", linewidth=0.6,
                                    color="0.4"))
        ax.set_xlabel("time of delivery (min)")
        ax.set_ylabel("surviving fraction")
        ax.set_title("(c) timing, lethal insult", fontsize=6.8, pad=3)
        _clean(ax)
        ax.legend(loc="upper right", frameon=False, handlelength=1.9,
                  labelspacing=0.25)
    fig.subplots_adjust(wspace=0.46, bottom=0.26)
    return _save(fig, "fig9_dose_timing.png")


# ----------------------------------------------------------------------
def fig10_sobol(res):
    sb = res["sobol"]
    names = sb["names"]
    outs = ["peak_sulfinate", "mean_persulfide", "death_fraction"]
    titles = ["peak sulfinate occupancy", "mean persulfide occupancy",
              "executed fraction"]
    fig, axes = plt.subplots(1, 3, figsize=(DCOL, 2.05), sharey=True)
    nice = {"b_ox": r"$b_{\mathrm{ox}}$", "b_psulf": r"$b_{\mathrm{psulf}}$",
            "b_overox": r"$b_{\mathrm{overox}}$", "b_psox": r"$b_{\mathrm{psox}}$",
            "b_depsulf": r"$b_{\mathrm{depsulf}}$", "b_glut": r"$b_{\mathrm{glut}}$",
            "g_SOH": r"$g_{\mathrm{SOH}}$", "g_SSH": r"$g_{\mathrm{SSH}}$",
            "g_SO2H": r"$g_{\mathrm{SO_2H}}$", "k_turn": r"$k_{\mathrm{t}}$",
            "v_cbs": r"$v_{S}$", "k_sqr": r"$k_{q}$",
            "K_sqr_inh": r"$K_{S}$", "q_nox": r"$q_{n}$", "k_px": r"$k_{x}$",
            "K_rps3": r"$K_{R}$"}
    pretty = [nice.get(n, n.replace("_", " ")) for n in names]
    y = np.arange(len(names))
    for j, (o, ttl) in enumerate(zip(outs, titles)):
        ax = axes[j]
        S1 = np.array(sb["results"][o]["S1"])
        ST = np.array(sb["results"][o]["ST"])
        ax.barh(y + 0.19, np.clip(ST, 0, None), height=0.36, facecolor="0.62",
                edgecolor="0.1", linewidth=0.5, hatch="///", label="total")
        ax.barh(y - 0.19, np.clip(S1, 0, None), height=0.36, facecolor="0.88",
                edgecolor="0.1", linewidth=0.5, label="first order")
        ax.set_title(ttl, fontsize=6.9, pad=3)
        ax.set_xlabel("Sobol index")
        _clean(ax)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(pretty, fontsize=6.0)
    axes[0].invert_yaxis()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.015), ncol=2,
               frameon=False, handlelength=1.8, columnspacing=1.6)
    fig.subplots_adjust(wspace=0.14, bottom=0.22)
    return _save(fig, "fig10_sobol.png")


def build_all(path="results.json"):
    with open(path) as fh:
        res = json.load(fh)
    made = [fig1_mechanism(), fig2_architecture()]
    for key, fn in [("reference", fig3_reference), ("capability", fig4_capability),
                    ("ablation", fig5_ablation),
                    ("identifiability", fig6_identifiability),
                    ("profiles", fig7_profiles), ("commitment", fig8_commitment),
                    ("dose_timing", fig9_dose_timing), ("sobol", fig10_sobol)]:
        if key in res:
            try:
                made.append(fn(res))
            except Exception as exc:
                print("figure for %s failed: %s" % (key, exc))
    return made


if __name__ == "__main__":
    for p in build_all():
        print(p)
