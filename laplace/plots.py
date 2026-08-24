"""Figures. Kept free of any torch dependency so that they can be redrawn from
results.json alone (see make_figures.py) without re-running the experiment.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- validated palette (categorical slots 1-3); distinct markers = secondary encoding ---
C = {"random": "#2a78d6", "curiosity": "#eb6834", "curiosity_safe": "#1baf7a"}
M = {"random": "o", "curiosity": "s", "curiosity_safe": "^"}
LBL = {"random": "random", "curiosity": "curiosity", "curiosity_safe": "curiosity + safety"}
INK, INK2 = "#0b0b0b", "#52514e"
BG = "#fcfcfb"


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left", pad=8)
    ax.grid(True, color="#e6e5e1", lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#cfcec9")
    ax.tick_params(colors=INK2, labelsize=8)


def fig_efficiency(hists, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for ax, key, title in [(axes[0], "rmse_id", "In-distribution test"),
                           (axes[1], "rmse_ood", "Out of distribution (unseen topologies and regimes)")]:
        for st, h in hists.items():
            x = [r["n_exp"] for r in h]
            y = [r[key]["vm"] for r in h]
            ax.plot(x, y, color=C[st], lw=2, marker=M[st], ms=5, label=LBL[st])
            ax.annotate(LBL[st], (x[-1], y[-1]), textcoords="offset points", xytext=(6, 0),
                        color=INK2, fontsize=8, va="center")
        style(ax, "experiments performed", "RMSE on |V| (p.u.)", title)
        ax.set_xlim(right=max(x) * 1.35)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Experiment efficiency: choosing beats collecting",
                 color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def fig_calibration(sd, err, path):
    """sd: stated uncertainty, err: observed error -- both flat arrays."""
    corr = float(np.corrcoef(sd, err)[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(sd, err, s=9, alpha=0.28, color="#2a78d6", edgecolors="none")
    lim = max(sd.max(), err.max()) * 1.02
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
    ax.annotate(f"correlation = {corr:.2f}", (0.04, 0.93), xycoords="axes fraction",
                color=INK, fontsize=9)
    style(ax, "stated uncertainty (ensemble standard deviation)", "observed error on |V|",
          "Does the model know what it does not know?")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
    return corr


def fig_frontier(hists, path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for st in ("curiosity", "curiosity_safe"):
        h = hists[st]
        axes[0].plot([r["n_exp"] for r in h], [r["rmse_ood"]["vm"] for r in h],
                     color=C[st], lw=2, marker=M[st], ms=5, label=LBL[st])
        axes[1].plot([r["n_exp"] for r in h], [r["n_unsafe"] for r in h],
                     color=C[st], lw=2, marker=M[st], ms=5, label=LBL[st])
    style(axes[0], "experiments performed", "RMSE on |V| (out of distribution)",
          "What caution costs")
    style(axes[1], "experiments performed", "interventions outside the envelope (cumulative)",
          "What it avoids")
    for ax in axes:
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.suptitle("Information / safety frontier", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)


def fig_law(pairs, path):
    true_v = np.array([p[0] for p in pairs])
    hat_v = np.array([p[1] for p in pairs])
    fig, ax = plt.subplots(figsize=(5, 4.4))
    lim = [min(true_v.min(), hat_v.min()) * 1.05, max(true_v.max(), hat_v.max()) * 1.05]
    ax.plot(lim, lim, color=INK2, lw=1, ls="--")
    ax.scatter(true_v, hat_v, s=26, color="#eb6834", alpha=0.85, edgecolors="none")
    style(ax, "true simulator coefficient (Ybus)", "recovered coefficient",
          "Discovered law against ground truth")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=BG)
    plt.close(fig)
