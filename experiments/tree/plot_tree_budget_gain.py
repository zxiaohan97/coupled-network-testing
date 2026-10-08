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


def plot_budget_gain(input_path: Path, output_path: Path) -> None:
    """Plot conditional paired gains alongside each policy's target reach rate."""

    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"{input_path} contains no rows")

    required = {"ci95_low", "ci95_high", "greedy_reach_fraction", "contact_reach_fraction"}
    if not required.issubset(rows[0]):
        raise ValueError("rerun random_tree_budget_gain.py to generate intervals and reach rates")
    settings = rows[0]
    for key in ("n_nodes", "n_trees", "p", "q", "physical_error", "max_budget", "random_seed"):
        if len({row[key] for row in rows}) != 1:
            raise ValueError(f"input must use a single {key} value")
    targets = sorted({float(row["threshold"]) for row in rows}, reverse=True)
    social_errors = sorted({float(row["social_error"]) for row in rows})
    correlations = sorted({float(row["r"]) for row in rows})
    colors = ("#26749B", "#8A527B", "#617B30", "#A55F22")

    fig, axes = plt.subplots(
        2,
        len(targets),
        figsize=(6.0 * len(targets), 7.5),
        sharex="col",
        squeeze=False,
    )
    for column, target in enumerate(targets):
        gain_ax, reach_ax = axes[:, column]
        gain_ax.set_title(f"Uncertainty target = {target:g}", fontsize=13, pad=12)
        any_gain = False
        for index, social_error in enumerate(social_errors):
            points = sorted(
                (
                    row
                    for row in rows
                    if float(row["threshold"]) == target
                    and float(row["social_error"]) == social_error
                ),
                key=lambda row: float(row["r"]),
            )
            xs = np.array([float(row["r"]) for row in points])
            gains = 100 * np.array([float(row["mean_budget_gain"]) for row in points])
            lows = 100 * np.array([float(row["ci95_low"]) for row in points])
            highs = 100 * np.array([float(row["ci95_high"]) for row in points])
            color = colors[index % len(colors)]
            label = f"Social error = {social_error:g}" if len(social_errors) > 1 else "Paired mean"
            gain_ax.plot(xs, gains, "o-", color=color, linewidth=2, markersize=4, label=label)
            gain_ax.fill_between(xs, lows, highs, color=color, alpha=0.16)
            any_gain |= bool(np.isfinite(gains).any())

            if len(social_errors) == 1:
                for x, row in zip(xs, points, strict=True):
                    gain_ax.text(
                        x,
                        -0.15,
                        f"n={row['num_reached']}",
                        transform=gain_ax.get_xaxis_transform(),
                        ha="center",
                        fontsize=7.5,
                        color="0.35",
                    )

            suffix = f" (social error {social_error:g})" if len(social_errors) > 1 else ""
            for policy, policy_label, policy_color, line_style in (
                ("greedy", "Greedy", "#187D68", "-"),
                ("contact", "Contact tracing", "#C65B4B", "--"),
            ):
                rates = [100 * float(row[f"{policy}_reach_fraction"]) for row in points]
                reach_ax.plot(
                    xs,
                    rates,
                    marker="o",
                    markersize=4,
                    linewidth=2,
                    color=policy_color,
                    linestyle=line_style,
                    label=policy_label + suffix,
                )

        gain_ax.axhline(0, color="0.4", linewidth=0.8)
        gain_ax.set_ylabel("Mean testing-budget gain (%)")
        gain_ax.legend(frameon=False, fontsize=9)
        if not any_gain:
            gain_ax.text(
                0.5,
                0.55,
                "No eligible paired budgets",
                transform=gain_ax.transAxes,
                ha="center",
                color="0.35",
            )
        reach_ax.set_title(f"Target reached within {settings['max_budget']} tests", fontsize=11)
        reach_ax.set_xlabel("Social correlation r")
        reach_ax.set_ylabel("Trees reaching target (%)")
        reach_ax.set_ylim(-4, 104)
        reach_ax.legend(frameon=False, fontsize=9)
        reach_ax.set_xticks(correlations)
        for ax in (gain_ax, reach_ax):
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=9)

    error_label = ", ".join(f"{value:g}" for value in social_errors)
    fig.suptitle(
        "Random-tree testing: budget savings and target reachability\n"
        f"{settings['n_trees']} paired trees | {settings['n_nodes']} nodes | "
        f"p={float(settings['p']):g}, q={float(settings['q']):g} | "
        f"physical error={float(settings['physical_error']):g}, social error={error_label}",
        fontsize=14,
    )
    fig.text(
        0.5,
        0.035,
        "Shading: approximate 95% confidence intervals across trees; n: eligible paired trees.\n"
        "Gains use trees where both policies reach the target "
        "and the baseline budget is positive.\n"
        "Targets apply to exact outcome-averaged posterior uncertainty; reach rates use all trees.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    fig.tight_layout(rect=(0, 0.11, 1, 0.98), h_pad=3.2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/tree_budget_gain.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/selected_results/tree/tree_budget_gain.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_budget_gain(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
