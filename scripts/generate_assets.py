"""Generate README visual assets (banner + benchmark chart) with matplotlib.

Usage:
    python scripts/generate_assets.py

Outputs:
    docs/assets/banner.png
    docs/assets/benchmark.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ASSETS = Path(__file__).resolve().parent.parent / "docs" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = "#0d1b2a"
NAVY_LIGHT = "#1b263b"
ORANGE = "#ff7a29"
WHITE = "#f5f7fa"
GRAY = "#9fb0c3"
GREEN = "#4caf7d"
RED = "#e05d5d"
BLUE = "#4d9de0"


def _bubble(ax, x, y, w, h, color, mark=None):
    """Draw a rounded chat bubble with an optional check/cross mark."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.008,rounding_size=0.02",
            linewidth=0,
            facecolor=color,
            alpha=0.9,
        )
    )
    if mark == "x":
        ax.text(
            x + w - 0.018,
            y + h / 2,
            "✕",
            color=RED,
            fontsize=15,
            ha="center",
            va="center",
            fontweight="bold",
        )
    elif mark == "ok":
        ax.text(
            x + w - 0.018,
            y + h / 2,
            "✓",
            color=GREEN,
            fontsize=15,
            ha="center",
            va="center",
            fontweight="bold",
        )


def _block(ax, x, y, w, h, color):
    """Draw a structured-data block."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            linewidth=1.2,
            edgecolor=color,
            facecolor="none",
            alpha=0.95,
        )
    )
    for i in range(3):
        ax.plot(
            [x + 0.008, x + w - 0.008],
            [y + h * (0.28 + 0.22 * i)] * 2,
            color=color,
            linewidth=1.6,
            alpha=0.55,
            solid_capstyle="round",
        )


def make_banner() -> Path:
    fig, ax = plt.subplots(figsize=(19.2, 6.4), dpi=100)
    fig.patch.set_facecolor(NAVY)
    ax.set_facecolor(NAVY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Background gradient (vertical, navy -> lighter navy)
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(
        grad,
        extent=(0, 1, 0, 1),
        aspect="auto",
        cmap=plt.matplotlib.colors.LinearSegmentedColormap.from_list("bg", [NAVY, NAVY_LIGHT]),
        alpha=0.9,
        zorder=0,
    )

    # Signal waveform behind the wordmark
    t = np.linspace(0, 1, 800)
    wave = 0.5 + 0.045 * np.sin(t * 34) * np.exp(-((t - 0.5) ** 2) / 0.06)
    ax.plot(t, wave, color=ORANGE, linewidth=1.4, alpha=0.35, zorder=1)

    # Left: chat bubbles (conversation logs)
    _bubble(ax, 0.045, 0.62, 0.13, 0.10, "#25344a", mark="x")
    _bubble(ax, 0.075, 0.45, 0.13, 0.10, "#25344a", mark="ok")
    _bubble(ax, 0.045, 0.28, 0.13, 0.10, "#25344a", mark="x")

    # Flow arrows
    for y in (0.67, 0.50, 0.33):
        ax.annotate(
            "",
            xy=(0.265, 0.50),
            xytext=(0.215, y),
            arrowprops={"arrowstyle": "-", "color": GRAY, "alpha": 0.35, "lw": 1.2},
        )

    # Right: structured data blocks (training pairs)
    _block(ax, 0.80, 0.58, 0.085, 0.16, GREEN)
    _block(ax, 0.845, 0.40, 0.085, 0.16, BLUE)
    _block(ax, 0.80, 0.22, 0.085, 0.16, ORANGE)
    for y in (0.66, 0.48, 0.30):
        ax.annotate(
            "",
            xy=(0.795, y),
            xytext=(0.735, 0.50),
            arrowprops={"arrowstyle": "-", "color": GRAY, "alpha": 0.35, "lw": 1.2},
        )

    # Wordmark: SigX (X in orange)
    ax.text(
        0.478,
        0.56,
        "Sig",
        color=WHITE,
        fontsize=110,
        ha="right",
        va="center",
        fontweight="bold",
        family="DejaVu Sans",
        zorder=3,
    )
    ax.text(
        0.478,
        0.56,
        "X",
        color=ORANGE,
        fontsize=110,
        ha="left",
        va="center",
        fontweight="bold",
        family="DejaVu Sans",
        zorder=3,
    )

    # Tagline
    ax.text(
        0.50,
        0.30,
        "Implicit Feedback Signal Extraction for LLM Alignment",
        color=GRAY,
        fontsize=22,
        ha="center",
        va="center",
        family="DejaVu Sans",
        zorder=3,
    )

    out = ASSETS / "banner.png"
    fig.savefig(out, dpi=100, bbox_inches="tight", pad_inches=0, facecolor=NAVY)
    plt.close(fig)
    return out


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
    print("banner    ->", make_banner())
    print("benchmark ->", make_benchmark_chart())
