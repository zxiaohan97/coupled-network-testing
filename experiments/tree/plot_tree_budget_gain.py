"""Plot random-tree budget-gain experiment output."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a budget-gain CSV produced by ``random_tree_budget_gain.py``."""

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def plot_budget_gain(input_path: Path, output_path: Path, threshold: float = 0.1) -> None:
    """Plot conditional paired gains versus correlation for one uncertainty target."""

    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"{input_path} contains no rows")

    required = {"threshold", "ci95_low", "ci95_high"}
    if not required.issubset(rows[0]):
        raise ValueError("rerun random_tree_budget_gain.py to generate thresholds and intervals")
    # Select a single target even when the input contains an exploratory sweep.
    rows = [row for row in rows if float(row["threshold"]) == threshold]
    if not rows:
        raise ValueError(f"{input_path} contains no rows for threshold {threshold:g}")
    settings = rows[0]
    for key in ("n_nodes", "n_trees", "p", "q", "physical_error", "max_budget", "random_seed"):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"input must use a single {key} value")
    social_errors = sorted({float(row["social_error"]) for row in rows})
    correlations = sorted({float(row["r"]) for row in rows})
    colors = ("#26749B", "#8A527B", "#617B30", "#A55F22")

    fig, ax = plt.subplots(figsize=(8, 5.7))
    any_gain = False
    for index, social_error in enumerate(social_errors):
        points = sorted(
            (row for row in rows if float(row["social_error"]) == social_error),
            key=lambda row: float(row["r"]),
        )
        xs = np.array([float(row["r"]) for row in points])
        gains = 100 * np.array([float(row["mean_budget_gain"]) for row in points])
        lows = 100 * np.array([float(row["ci95_low"]) for row in points])
        highs = 100 * np.array([float(row["ci95_high"]) for row in points])
        color = colors[index % len(colors)]
        label = (
            f"Social error = {social_error:g}"
            if len(social_errors) > 1
            else "Greedy vs contact tracing"
        )
        ax.plot(xs, gains, "o-", color=color, linewidth=2, markersize=5, label=label)
        ax.fill_between(xs, lows, highs, color=color, alpha=0.16)
        any_gain |= bool(np.isfinite(gains).any())

    ax.axhline(0, color="0.4", linewidth=0.8)
    ax.set_xlabel("Social correlation r", fontsize=11)
    ax.set_ylabel("Mean testing-budget gain (%)", fontsize=11)
    ax.set_xticks(correlations)
    ax.margins(x=0.03, y=0.1)
    ax.legend(frameon=False, fontsize=10, loc="upper left")
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    if not any_gain:
        ax.text(
            0.5,
            0.55,
            "No eligible paired budgets",
            transform=ax.transAxes,
            ha="center",
            color="0.35",
        )

    error_label = ", ".join(f"{value:g}" for value in social_errors)
    fig.suptitle("Budget gain versus social correlation", fontsize=16, y=0.97)
    fig.text(
        0.5,
        0.91,
        f"{settings['n_trees']} paired trees | {settings['n_nodes']} nodes | "
        f"uncertainty threshold={threshold:g} | max budget={settings['max_budget']}\n"
        f"p={float(settings['p']):g}, q={float(settings['q']):g} | "
        f"physical error={float(settings['physical_error']):g}, social error={error_label}",
        ha="center",
        va="top",
        fontsize=10,
    )
    fig.text(
        0.5,
        0.04,
        "Shading: approximate pointwise 95% confidence intervals across trees.\n"
        "Means use finite paired budgets with a positive contact-tracing baseline.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0.015, 0.12, 0.985, 0.85))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/tree_budget_gain.csv"))
    parser.add_argument("--threshold", type=float, default=0.1, help="Uncertainty target to plot")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/selected_results/tree/tree_budget_gain.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_budget_gain(args.input, args.output, threshold=args.threshold)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
