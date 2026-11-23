"""Rebuild the figures in figures/ from the recorded result files.

    python scripts/make_figures.py

Needs matplotlib (not needed for anything else here). Reads only files in
results/, so every plotted value is also in results/TABLES.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT = ROOT / "figures"

SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE = "#2a78d6", "#eb6834"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def frame(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def titles(fig, title, subtitle):
    fig.text(0.012, 0.975, title, fontsize=11.5, fontweight="semibold", va="top", ha="left")
    fig.text(0.012, 0.915, subtitle, fontsize=9, color=INK2, va="top", ha="left")


def planted_rule_following():
    """Agreement with the hidden rule against the share of markers that obey it."""
    local = load(RES / "planted" / "local_baselines.json")

    def model_acc(folder):
        return load(RES / "planted" / folder / "history.json")["evals"][-1]["marker_acc"]

    strengths = sorted(float(p.name[len("rule_p_"):]) for p in (RES / "planted").glob("rule_p_*")
                       if "_seed" not in p.name)
    name = lambda rp: f"rule_p_{rp:g}" if (RES / "planted" / f"rule_p_{rp:g}").exists() else f"rule_p_{rp}"
    model = [model_acc(name(rp)) for rp in strengths]
    base = [local[name(rp)]["best_local_vs_rule"] for rp in strengths]
    seed_points = []
    for s in (1, 2):
        for rp in strengths:
            folder = f"{name(rp)}_seed{s}"
            if (RES / "planted" / folder).exists():
                seed_points.append((rp, model_acc(folder)))

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    fig.subplots_adjust(left=0.085, right=0.975, top=0.80, bottom=0.13)
    line = dict(linewidth=1.8, solid_capstyle="round", solid_joinstyle="round",
                marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.4)
    ax.plot(strengths, base, color=ORANGE, label="best of four local predictors", **line)
    ax.plot(strengths, model, color=BLUE, label="4-layer transformer (seed 0)", **line)
    ax.plot([p[0] for p in seed_points], [p[1] for p in seed_points], linestyle="none", color=BLUE,
            marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.4,
            label="the same, two further seeds")
    i = strengths.index(0.05)
    ax.annotate(f"{model[i]:.3f} against {base[i]:.3f}\nwhen 1 marker in 20 obeys the rule",
                xy=(0.058, model[i] - 0.012), xytext=(0.13, 0.80), fontsize=9, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(0.35, 1.04)
    ax.set_xlabel("share of training markers set by the hidden rule")
    ax.set_ylabel("agreement with the rule, held-out markers")
    frame(ax)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    titles(fig, "A small transformer follows a hidden rule far better than four local predictors",
           "Planted counter, synthetic corpus. Both curves are scored against the rule on held-out sequences.")
    fig.savefig(OUT / "planted_rule_following.png")
    plt.close(fig)


def depth_removal_comparison():
    """The removal test: the depth subspace against two kinds of comparison subspace."""
    models = [("gpt2", "GPT-2"), ("gpt2-medium", "GPT-2-medium"), ("gpt2-large", "GPT-2-large"),
              ("gpt2-xl", "GPT-2-XL"), ("microsoft_Phi-3.5-mini-instruct", "Phi-3.5-mini"),
              ("Qwen_Qwen2.5-7B-Instruct", "Qwen2.5-7B"), ("Qwen_Qwen3-8B", "Qwen3-8B")]
    rows = []
    for key, label in models:
        d = load(RES / f"causal_depth_by_depth_{key}.json")
        depth = d["increase_depth_subspace"]["all"]
        cell = lambda s: (depth - s["mean"]["all"], depth - s["max"]["all"], depth - s["min"]["all"])
        rows.append((label, cell(d["increase_random_subspace"]), cell(d["increase_shuffled_probe_subspace"])))

    fig, ax = plt.subplots(figsize=(8, 4.4), dpi=200)
    fig.subplots_adjust(left=0.155, right=0.975, top=0.745, bottom=0.14)
    ys = list(range(len(rows)))[::-1]
    for y, (label, rand, shuf) in zip(ys, rows):
        for (mid, lo, hi), color, dy in ((rand, BLUE, 0.14), (shuf, ORANGE, -0.14)):
            ax.plot([lo, hi], [y + dy, y + dy], color=color, linewidth=1.8, solid_capstyle="round")
            ax.plot([mid], [y + dy], marker="o", markersize=7, color=color,
                    markeredgecolor=SURFACE, markeredgewidth=1.4, linestyle="none")
    ax.axvline(0, color=MUTED, linewidth=0.9)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlabel("loss increase with the depth subspace removed, minus the increase for the comparison subspace")
    frame(ax, grid_axis="x")
    lo, hi = ax.get_xlim()
    ax.text(-0.06, len(rows) - 0.42, "depth subspace costs less than the comparison", fontsize=8.5,
            color=MUTED, ha="right", va="center")
    ax.text(0.06, len(rows) - 0.42, "costs more", fontsize=8.5, color=MUTED, ha="left", va="center")
    handles = [plt.Line2D([], [], color=BLUE, marker="o", markersize=7, markeredgecolor=SURFACE,
                          markeredgewidth=1.4, linewidth=1.8, label="against random subspaces (mean and range of ten)"),
               plt.Line2D([], [], color=ORANGE, marker="o", markersize=7, markeredgecolor=SURFACE,
                          markeredgewidth=1.4, linewidth=1.8,
                          label="against the same probe fit to shuffled labels (mean and range of three)")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.875), frameon=False,
               fontsize=9, labelcolor=INK2, ncol=1, handlelength=1.6)
    titles(fig, "Whether the depth subspace looks used depends on the comparison",
           "Removal of a 128-dimensional subspace at the middle layer; seven pretrained models, 360 held-out sentences.")
    fig.savefig(OUT / "depth_removal_comparison.png")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    planted_rule_following()
    depth_removal_comparison()
    print(f"wrote {OUT / 'planted_rule_following.png'}")
    print(f"wrote {OUT / 'depth_removal_comparison.png'}")


if __name__ == "__main__":
    main()
