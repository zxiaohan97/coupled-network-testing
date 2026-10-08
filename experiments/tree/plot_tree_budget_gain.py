"""Plot random-tree budget-gain experiment output."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a budget-gain CSV produced by ``random_tree_budget_gain.py``."""

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def plot_budget_gain(input_path: Path, output_path: Path) -> None:
    """Create a compact budget-gain figure grouped by social test error."""

    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"{input_path} contains no rows")

    grouped: defaultdict[float, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[float(row["social_error"])].append(
            (float(row["r"]), float(row["mean_budget_gain"])),
        )

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for social_error, points in sorted(grouped.items()):
        points.sort()
        xs = [point[0] for point in points]
        ys = [100.0 * point[1] for point in points]
        ax.plot(xs, ys, marker="o", linewidth=2, label=f"social error={social_error:g}")

    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("social correlation r")
    ax.set_ylabel("mean budget gain vs contact tracing (%)")
    ax.set_title("Random-tree budget gain")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

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
