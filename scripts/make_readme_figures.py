"""Regenerate the figures embedded in README.MD.

Usage:
    py scripts/make_readme_figures.py

Writes light and dark variants to docs/images/ so the README can serve each
through a <picture> element.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "docs" / "images"

FILE_STEMS = {
    2020: "h20201332",
    2021: "h20211332",
    2022: "h20221331",
    2023: "h20231331",
    2024: "h20241331",
}

SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}
SEVERITY_ORDER = ["Fatal", "Serious", "Slight"]

# Northern edge of the empty Arava stretch, in ITM metres.
SOUTH_CUTOFF = 520_000

# Ordinal blue ramp: severity order is encoded as lightness, so the most severe
# class always carries the most contrast against the surface. Both sets pass
# scripts/validate_palette.js --ordinal against their own surface.
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "ink": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "series": {"Fatal": "#0d366b", "Serious": "#2a78d6", "Slight": "#86b6ef"},
        "sequential": ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"],
    },
    "dark": {
        "surface": "#1a1a19",
        "ink": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "series": {"Fatal": "#cde2fb", "Serious": "#5598e7", "Slight": "#184f95"},
        "sequential": ["#184f95", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"],
    },
}

plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans", "sans-serif"]


def load() -> pd.DataFrame:
    if not (DATA_DIR / f"{FILE_STEMS[2020]}data.csv").exists():
        raise SystemExit(
            "No data found. Run:  py scripts/download_data.py"
        )
    frames = [
        pd.read_csv(DATA_DIR / f"{stem}data.csv").assign(year=year)
        for year, stem in FILE_STEMS.items()
    ]
    return pd.concat(frames, ignore_index=True)


def style_axes(ax, theme: dict) -> None:
    ax.set_facecolor(theme["surface"])
    ax.grid(True, color=theme["grid"], linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme["axis"])
        ax.spines[side].set_linewidth(1)
    ax.tick_params(colors=theme["muted"], labelsize=10, length=0)


def figure_severity_trend(df: pd.DataFrame, theme_name: str) -> None:
    theme = THEMES[theme_name]

    counts = pd.crosstab(df["year"], df["HUMRAT_TEUNA"].map(SEVERITY_LABELS))
    counts = counts[SEVERITY_ORDER]
    indexed = counts / counts.loc[2020] * 100

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=theme["surface"])
    style_axes(ax, theme)

    # 2020 baseline: everything is read as a departure from this line.
    ax.axhline(100, color=theme["axis"], linewidth=1, linestyle=(0, (4, 4)), zorder=1)

    for label in SEVERITY_ORDER:
        ax.plot(
            indexed.index,
            indexed[label],
            color=theme["series"][label],
            linewidth=2,
            marker="o",
            markersize=8,
            markeredgecolor=theme["surface"],
            markeredgewidth=2,
            zorder=3,
        )
        # Direct labels carry identity, so it never rests on color alone.
        ax.annotate(
            f"  {label}  {indexed[label].iloc[-1]:.0f}",
            xy=(indexed.index[-1], indexed[label].iloc[-1]),
            va="center",
            fontsize=11,
            fontweight="bold",
            color=theme["series"][label],
        )

    ax.set_xticks(list(indexed.index))
    ax.set_xlim(2019.85, 2024.95)
    ax.set_ylabel("Accidents, indexed to 2020 = 100", color=theme["secondary"], fontsize=10)
    ax.set_ylim(50, 155)

    fig.suptitle(
        "Recorded slight-injury accidents fell 37% while fatal accidents rose 42%",
        x=0.055,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=theme["ink"],
    )
    ax.set_title(
        "Israel, accidents with casualties, 2020-2024. Counts, not rates - the rising\n"
        "share of severe accidents is driven by slight accidents leaving the record.",
        loc="left",
        fontsize=10.5,
        color=theme["secondary"],
        pad=12,
    )
    fig.text(
        0.055,
        0.02,
        "Source: Israel CBS road accidents with casualties (PUF), 2020-2024.",
        fontsize=9,
        color=theme["muted"],
    )

    fig.tight_layout(rect=(0, 0.035, 0.87, 0.94))
    fig.savefig(
        OUTPUT_DIR / f"severity-trend-{theme_name}.png",
        dpi=200,
        facecolor=theme["surface"],
    )
    plt.close(fig)


def figure_severity_map(df: pd.DataFrame, theme_name: str) -> None:
    theme = THEMES[theme_name]
    cmap = LinearSegmentedColormap.from_list("seq", theme["sequential"])

    located = df.dropna(subset=["X", "Y"])
    severe = located["HUMRAT_TEUNA"].isin([1, 2]).to_numpy(dtype=float) * 100

    fig, ax = plt.subplots(figsize=(5.6, 9.4), facecolor=theme["surface"])
    ax.set_facecolor(theme["surface"])

    # A tall, narrow grid matches Israel's extent, so the hexagons stay regular
    # once the aspect is locked. mincnt guards against the 0%/100% noise that
    # tiny cells produce - the same caution the EDA applies to sparse GEO_GRID
    # cells.
    shared = dict(
        C=severe,
        reduce_C_function=np.mean,
        gridsize=(26, 46),
        mincnt=25,
    )
    rates = ax.hexbin(located["X"], located["Y"], **shared).get_array()
    ax.clear()

    # A handful of sparse cells would otherwise stretch the ramp and flatten
    # everything else, so the top of the scale is clipped and marked as such.
    upper = float(np.percentile(rates.compressed(), 97))

    hexes = ax.hexbin(
        located["X"],
        located["Y"],
        **shared,
        cmap=cmap,
        vmin=0,
        vmax=upper,
        linewidths=0.4,
        edgecolors=theme["surface"],
    )

    # Eilat and the Arava sit ~120km south of everything else, so plotting the
    # full extent spends a third of the frame on empty desert. The frame stops
    # north of the gap and the caption reports what that leaves out.
    ax.set_ylim(SOUTH_CUTOFF, located["Y"].max() + 6000)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.margins(x=0.01)
    for spine in ax.spines.values():
        spine.set_visible(False)

    bar = fig.colorbar(
        hexes,
        ax=ax,
        orientation="horizontal",
        location="bottom",
        shrink=0.62,
        pad=0.01,
        aspect=26,
        extend="max",
    )
    bar.set_label(
        "Fatal or serious, % of accidents", color=theme["secondary"], fontsize=10
    )
    bar.ax.tick_params(colors=theme["muted"], labelsize=9, length=0)
    bar.outline.set_visible(False)

    fig.suptitle(
        "Severe-accident rate varies widely across Israel",
        x=0.04,
        y=0.975,
        ha="left",
        fontsize=14,
        fontweight="bold",
        color=theme["ink"],
    )
    omitted = (located["Y"] < SOUTH_CUTOFF).mean() * 100
    fig.text(
        0.04,
        0.938,
        "Cells with at least 25 accidents, 2020-2024. Israeli Transverse Mercator grid.",
        fontsize=9.5,
        color=theme["secondary"],
    )
    fig.text(
        0.04,
        0.915,
        f"Eilat and the Arava ({omitted:.1f}% of located accidents) fall south of this frame.",
        fontsize=9.5,
        color=theme["muted"],
    )

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.06)
    fig.savefig(
        OUTPUT_DIR / f"severity-map-{theme_name}.png",
        dpi=200,
        facecolor=theme["surface"],
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()

    for theme_name in THEMES:
        figure_severity_trend(df, theme_name)
        figure_severity_map(df, theme_name)

    for path in sorted(OUTPUT_DIR.glob("*.png")):
        print(f"  {path.relative_to(PROJECT_ROOT)}  ({path.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
