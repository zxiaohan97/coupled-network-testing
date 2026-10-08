"""Plot spanning-tree approximation experiment output."""

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


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV produced by ``spanning_tree_approximation.py``."""

    with path.open(newline="") as file:
        return list(csv.DictReader(file))


def plot_spanning_tree_approximation(input_path: Path, output_path: Path) -> None:
    """Plot approximation error and average number of excess edges."""

    rows = read_rows(input_path)
    if not rows:
        raise ValueError(f"{input_path} contains no rows")

    rows.sort(key=lambda row: float(row["edge_probability"]))
    edge_probabilities = [float(row["edge_probability"]) for row in rows]
    relative_differences = [
        100.0 * abs(float(row["mean_relative_difference"]))
        for row in rows
    ]
    relative_errors = [
        100.0 * float(row.get("stderr_relative_difference") or 0.0)
        for row in rows
    ]
    cyclomatic_numbers = [float(row["mean_cyclomatic_number"]) for row in rows]

    fig, ax_error = plt.subplots(figsize=(6.5, 4.0))
    ax_cycles = ax_error.twinx()

    error_line = ax_error.errorbar(
        edge_probabilities,
        relative_differences,
        yerr=relative_errors,
        marker="o",
        linewidth=2,
        capsize=4,
        color="#2a6fbb",
        label="relative difference",
    )
    cycle_line = ax_cycles.plot(
        edge_probabilities,
        cyclomatic_numbers,
        marker="s",
        linewidth=2,
        color="#9a4d1c",
        label="excess edges",
    )

    ax_error.set_xlabel("ER edge probability")
    ax_error.set_ylabel("|tree approximation difference| (%)", color="#2a6fbb")
    ax_cycles.set_ylabel("mean excess edges", color="#9a4d1c")
    ax_error.set_title("Spanning-tree approximation")
    ax_error.grid(alpha=0.25)

    lines = [error_line, *cycle_line]
    ax_error.legend(lines, [line.get_label() for line in lines], frameon=False)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/spanning_tree_approximation.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("figures/selected_results/tree/spanning_tree_approximation.png"),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    plot_spanning_tree_approximation(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
