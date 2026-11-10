"""Charts for the ground profiles and results, saved as PNG images."""

import matplotlib
import numpy as np

matplotlib.use("Agg")  # draw straight to files; no window needed
import matplotlib.pyplot as plt  # noqa: E402

# Colours: one hue for single measures; two opposite hues and a neutral grey for win / draw / loss.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
BAR = "#2a78d6"
WIN = "#2a78d6"
DRAW = "#c3c2b7"
LOSS = "#e34948"


def _style(ax):
    """Quiet axes: no box, light vertical gridlines, muted tick labels."""
    ax.set_facecolor(SURFACE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASE)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for label in ax.get_yticklabels():
        label.set_color(INK)


def plot_ground_profiles(profiles, path, title):
    """Two small panels side by side: draw rate and average first-innings score, one bar per ground."""
    labels = [f"{g}  (n={n})" for g, n in zip(profiles["ground"], profiles["tests"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    panels = [("draw_rate", "Draw rate", 100, "{:.0f}%"),
              ("avg_first_innings_runs", "Average first-innings score (runs)", 1, "{:.0f}")]
    for ax, (column, heading, scale, fmt) in zip(axes, panels):
        values = profiles[column].fillna(0) * scale
        ax.barh(labels, values, color=BAR, height=0.55)
        ax.invert_yaxis()  # first Test of the series at the top
        _style(ax)
        ax.set_title(heading, loc="left", color=INK, fontsize=11)
        for y, (v, n) in enumerate(zip(values, profiles["tests"])):
            shown = fmt.format(v) if n > 0 else "no Tests"
            ax.text(v, y, " " + shown, va="center", color=INK_2, fontsize=9)
        ax.set_xlim(0, values.max() * 1.15 if values.max() > 0 else 1)
    axes[1].set_yticklabels([])
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_results(records, path, title, labels=("Win", "Draw", "Loss"), as_share=False):
    """Stacked horizontal bars: wins, draws and losses at each ground, every segment labelled."""
    names = [f"{g}  (n={n})" for g, n in zip(records["ground"], records["tests"])]
    parts = records[["wins", "draws", "losses"]].astype(float)
    if as_share:
        parts = parts.div(records["tests"].where(records["tests"] > 0), axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(10, 3.8), facecolor=SURFACE)
    left = [0.0] * len(records)
    for column, colour, label, text_colour in [("wins", WIN, labels[0], "white"),
                                                 ("draws", DRAW, labels[1], INK),
                                                 ("losses", LOSS, labels[2], "white")]:
        widths = parts[column].tolist()
        ax.barh(names, widths, left=left, color=colour, label=label, height=0.55,
                edgecolor=SURFACE, linewidth=2)
        for y, (start, width) in enumerate(zip(left, widths)):
            if width > 0:
                shown = f"{width:.0f}%" if as_share else f"{width:.0f}"
                ax.text(start + width / 2, y, shown, ha="center", va="center", color=text_colour, fontsize=9)
        left = [a + b for a, b in zip(left, widths)]
    ax.invert_yaxis()
    _style(ax)
    if as_share:
        ax.set_xlim(0, 100)
        ax.set_xlabel("Share of Tests (%)", color=MUTED)
    else:
        ax.set_xlabel("Tests", color=MUTED)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path

def plot_strength(strength, years, teams, path, title, highlight=None):
    """Each team's strength over time: highlighted teams with a 90% band, the rest as thin grey lines."""
    highlight = highlight or {"England": WIN, "Australia": LOSS}
    median = np.median(strength, axis=0)
    low, high = np.percentile(strength, 5, axis=0), np.percentile(strength, 95, axis=0)
    fig, ax = plt.subplots(figsize=(10, 5), facecolor=SURFACE)
    for i, team in enumerate(teams):
        if team not in highlight:
            ax.plot(years, median[:, i], color=BASE, linewidth=1)
    for team, colour in highlight.items():
        i = teams.index(team)
        ax.fill_between(years, low[:, i], high[:, i], color=colour, alpha=0.15, linewidth=0)
        ax.plot(years, median[:, i], color=colour, linewidth=2, label=team)
        ax.text(years[-1] + 0.3, median[-1, i], team, va="center", color=INK, fontsize=10)
    ax.axhline(0, color=BASE, linewidth=0.8)
    _style(ax)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.xaxis.grid(False)
    ax.set_xlim(years[0], years[-1] + 2.5)
    ax.set_ylabel("Strength (0 = average Test side)", color=MUTED)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path


def plot_forecast(rows, path, title, labels=("England win", "Draw", "Australia win")):
    """One stacked bar per row of chances (p_home, p_draw, p_away), each piece labelled as a percentage."""
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(rows) + 1.6), facecolor=SURFACE)
    left = np.zeros(len(rows))
    for column, colour, label, text_colour in [("p_home", WIN, labels[0], "white"),
                                                 ("p_draw", DRAW, labels[1], INK),
                                                 ("p_away", LOSS, labels[2], "white")]:
        widths = rows[column].to_numpy() * 100
        ax.barh(rows["label"], widths, left=left, color=colour, label=label, height=0.6,
                edgecolor=SURFACE, linewidth=2)
        for y, (start, width) in enumerate(zip(left, widths)):
            if width >= 4:
                ax.text(start + width / 2, y, f"{width:.0f}%", ha="center", va="center",
                        color=text_colour, fontsize=9)
        left += widths
    ax.invert_yaxis()
    _style(ax)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Chance (%)", color=MUTED)
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, labelcolor=INK)
    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return path