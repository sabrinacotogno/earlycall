"""Matplotlib figures used by the scripts and the app.

Every function returns the Figure, so the same code works in a notebook, in a script
(fig.savefig) and in Streamlit (st.pyplot(fig)).
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from earlycall.batch_data import BatchData


def plot_trajectories(data: BatchData, param: str, labels: np.ndarray | None = None):
    """All batches for one parameter; coloured by group label when given (-1 in grey)."""
    p = data.params.index(param)
    fig, ax = plt.subplots(figsize=(8, 4))
    for b in range(data.n_batches):
        label = -1 if labels is None else labels[b]
        color = "lightgrey" if label < 0 else plt.cm.tab10(label % 10)
        ax.plot(data.time, data.X[b, p], color=color, linewidth=0.8, alpha=0.7)
    ax.set_xlabel("time")
    ax.set_ylabel(param)
    ax.set_title(f"{param}: all batches" + ("" if labels is None else " (colour = group, grey = none)"))
    fig.tight_layout()
    return fig


def plot_reference_band(percent: np.ndarray, reference: np.ndarray, band_std: np.ndarray, param: str,
                        run: np.ndarray | None = None, t_index: int | None = None,
                        band_k: float = 2.0):
    """One group's mean +- band_k * std for one parameter, and optionally a run up to t_index.

    percent: x-axis, each time point as % of the batch duration (BatchData.percent).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(percent, reference - band_k * band_std, reference + band_k * band_std,
                    color="tab:blue", alpha=0.2, label=f"group band (mean +- {band_k:g} std)")
    ax.plot(percent, reference, color="tab:blue", label="group mean")
    if run is not None:
        t_index = t_index or len(percent)
        ax.plot(percent[:t_index], run[:t_index], color="black", linewidth=2, label="run so far")
        ax.axvline(percent[t_index - 1], color="grey", linestyle="--")
    ax.set_xlim(0, 100)
    ax.set_xlabel("time into the batch (% of batch end)")
    ax.set_ylabel(param)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_vs_time(summary: pd.DataFrame, metric: str = "auc", title: str = ""):
    """One line per method: score against % of the batch seen (output of evaluation.summarize).

    Percent instead of hours or days, so all datasets share the same axis (0 = start, 100 = end).
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for method, rows in summary.groupby("method"):
        ax.plot(rows["percent"], rows[metric], marker="o", label=method)
    if metric == "auc":
        ax.axhline(0.5, color="grey", linestyle=":", label="coin flip")
        ax.set_ylim(0.2, 1.02)
    ax.set_xlim(0, 100)
    ax.set_xlabel("time into the batch (% of batch end)")
    ax.set_ylabel(metric)
    ax.set_title(title or f"{metric} vs time into the batch")
    ax.legend()
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ solar views (app)
def plot_group_references(time: np.ndarray, references: np.ndarray, share_faulty: np.ndarray):
    """Typical relative-power run of each group, darker red = more faulty runs in the group."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for g in np.argsort(share_faulty):
        ax.plot(time, references[g], linewidth=1.5, color=plt.cm.Reds(0.15 + 0.85 * share_faulty[g]))
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1.5)
    ax.set_xlabel("hour of the day")
    ax.set_ylabel("relative power")
    ax.set_title("Typical trajectory of each group (darker red = more faulty runs in the group)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ online app
INK, GOOD, LOW, MUTED = "#14213d", "#0f7b6c", "#d9822b", "#9aa3ab"
METHOD_STYLE = {   # display name -> (colour, line width, line style)
    "BatchGrouper": (GOOD, 3.0, "-"),
    "ridge": ("#3d6fb6", 2.0, "-"),
    "logistic": ("#3d6fb6", 2.0, "-"),
    "PLS": ("#8e5ea2", 2.0, "-"),
    "base rate": (MUTED, 1.5, ":"),
}


def clean_axes(ax) -> None:
    """Light style shared by the app figures: no box, faint grid."""
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.grid(alpha=0.25)
    ax.tick_params(colors=INK, labelsize=9)


def plot_tunnels(percent: np.ndarray, references: np.ndarray, band_std: np.ndarray, group_means: np.ndarray,
                 low_threshold: float, param: str, highlight: int | None = None,
                 run: np.ndarray | None = None, t_index: int | None = None, band_k: float = 2.0):
    """Every tunnel (group mean +- band_k * std) for one parameter.

    Green = the group's mean titer is above the low threshold, orange = below.
    highlight: the tunnel of the monitored batch, drawn strong; the others are faint.
    run, t_index: the monitored batch, drawn up to now.
    """
    fig, ax = plt.subplots(figsize=(9, 4))
    for g in range(len(references)):
        colour = LOW if group_means[g] < low_threshold else GOOD
        strong = highlight is None or g == highlight
        ax.fill_between(percent, references[g] - band_k * band_std[g], references[g] + band_k * band_std[g],
                        color=colour, alpha=0.28 if strong else 0.06, linewidth=0)
        ax.plot(percent, references[g], color=colour, linewidth=1.8 if strong else 0.8,
                alpha=1.0 if strong else 0.4)
    if run is not None:
        ax.plot(percent[:t_index], run[:t_index], color=INK, linewidth=2.5, label="monitored batch")
        ax.axvline(percent[t_index - 1], color=INK, linestyle="--", linewidth=1)
        ax.legend(frameon=False, loc="upper left")
    ax.set_xlim(0, 100)
    ax.set_xlabel("time into the batch (%)", color=INK)
    ax.set_ylabel(param, color=INK)
    clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_scores(summary: pd.DataFrame, metric: str, ylabel: str):
    """One line per method (display names): score against % of the batch seen."""
    fig, ax = plt.subplots(figsize=(9, 4))
    for method, rows in summary.groupby("method"):
        colour, width, style = METHOD_STYLE.get(method, (INK, 2.0, "-"))
        ax.plot(rows["percent"], rows[metric], color=colour, linewidth=width, linestyle=style,
                marker="o", markersize=4, label=method)
    if metric == "auc":
        ax.axhline(0.5, color=MUTED, linewidth=1)
        ax.set_ylim(0.2, 1.02)
    else:
        ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, 100)
    ax.set_xlabel("time into the run (%)", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_runs_by_quality(percent: np.ndarray, curves: np.ndarray, quality: np.ndarray, param: str):
    """Every run for one parameter, coloured by its final quality (dark = high, light = low)."""
    fig, ax = plt.subplots(figsize=(8, 3.8))
    known = ~np.isnan(quality)
    low, high = np.nanmin(quality), np.nanmax(quality)
    for curve, q in zip(curves[known], quality[known]):
        ax.plot(percent, curve, color=plt.cm.viridis_r(0.1 + 0.8 * (high - q) / (high - low)), linewidth=1.1, alpha=0.8)
    scale = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(low, high))
    fig.colorbar(scale, ax=ax, pad=0.01).set_label("final titer (g/L)", color=INK)
    ax.set_xlim(0, 100)
    ax.set_xlabel("time into the batch (%)", color=INK)
    ax.set_ylabel(param, color=INK)
    clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_quality_distribution(quality: np.ndarray, low_threshold: float):
    """Histogram of the final quality, with the low threshold."""
    fig, ax = plt.subplots(figsize=(5, 3.8))
    quality = quality[~np.isnan(quality)]
    bins = np.linspace(quality.min(), quality.max(), 16)
    ax.hist(quality[quality < low_threshold], bins=bins, color=LOW, alpha=0.85, label="low")
    ax.hist(quality[quality >= low_threshold], bins=bins, color=GOOD, alpha=0.85, label="good")
    ax.axvline(low_threshold, color=INK, linestyle="--", linewidth=1)
    ax.set_xlabel("final titer (g/L)", color=INK)
    ax.set_ylabel("batches", color=INK)
    ax.legend(frameon=False)
    clean_axes(ax)
    fig.tight_layout()
    return fig


def plot_quality_ranges(group_range: np.ndarray, group_mean: np.ndarray, low_threshold: float):
    """One bar per tunnel: final quality from min to max, dot = mean. Orange = mean below threshold."""
    n = len(group_mean)
    fig, ax = plt.subplots(figsize=(5, 0.4 * n + 1.2))
    for g in range(n):
        colour = LOW if group_mean[g] < low_threshold else GOOD
        ax.plot(group_range[g], [g, g], color=colour, linewidth=8, alpha=0.35, solid_capstyle="round")
        ax.plot(group_mean[g], g, "o", color=colour, markersize=8)
    ax.axvline(low_threshold, color=INK, linestyle="--", linewidth=1)
    ax.set_yticks(range(n), [f"tunnel {g + 1}" for g in range(n)])
    ax.invert_yaxis()
    ax.set_xlabel("final titer (g/L)", color=INK)
    clean_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return fig
