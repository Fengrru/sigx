"""Generate README visual assets (benchmark chart) with matplotlib.

Usage:
    python scripts/generate_assets.py

Outputs:
    docs/assets/benchmark.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

ORANGE = "#ff7a29"
GREEN = "#4caf7d"
BLUE = "#4d9de0"


def make_benchmark_chart() -> Path:
    # Real numbers from: pipeline.evaluate("tests/benchmark.json")
    types = ["abandon", "positive", "correction", "negative", "rephrase", "overall"]
    precision = [1.00, 1.00, 0.67, 0.75, 1.00, 0.88]
    recall = [1.00, 1.00, 0.67, 0.60, 0.67, 0.79]
    f1 = [1.00, 1.00, 0.67, 0.67, 0.80, 0.83]

    x = np.arange(len(types))
    width = 0.26

    fig, ax = plt.subplots(figsize=(10.5, 4.6), dpi=150)
    fig.patch.set_facecolor("white")

    b1 = ax.bar(x - width, precision, width, label="Precision", color=BLUE, zorder=3)
    b2 = ax.bar(x, recall, width, label="Recall", color=ORANGE, zorder=3)
    b3 = ax.bar(x + width, f1, width, label="F1", color=GREEN, zorder=3)

    for bars in (b1, b2, b3):
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2, color="#444444")

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [t if t != "overall" else "OVERALL" for t in types],
        fontsize=10,
    )
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.set_ylabel("Score")
    ax.set_title(
        "SigX benchmark — regex extractors, 20 labeled conversations",
        fontsize=12,
        pad=12,
    )
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.axvline(x=4.5, color="#cccccc", linewidth=1, linestyle="--")

    out = ASSETS / "benchmark.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print("benchmark ->", make_benchmark_chart())
