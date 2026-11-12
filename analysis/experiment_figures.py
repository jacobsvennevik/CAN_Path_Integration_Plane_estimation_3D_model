"""Figures that compare stored experiment runs across plane settings.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, LogLocator, NullLocator
from matplotlib.transforms import blended_transform_factory

from config import normalized_to_metres

from . import gongyu_scoring as gy
from .gridness2d import percentile_cells
from .rate_maps import SMOOTH_SIGMA, _rate_map_from_accumulator, _spatial_info
from .summaries import mra_curve, mra_radii_m, pooled_ifd, pooled_ifd_cv


# Lines through the κ_sens × τ grid. A third line appears when some τ
# other than 1 was run at more than one κ_sens.
_KAPPA_COLOR = "#3D7EBF"
_TAU_COLOR = "#E07A4C"
_LINE3_COLOR = "#3C9A6E"
_ON_PLANE = "#6E6E6E"


def _run(runs, seed, kappa, tau):
    """One stored run, or None when that seed was not run.

    The true plane is saved under ('true', 'true').
    """
    if kappa is None:
        key = (int(seed), "true", "true")
    else:
        key = (int(seed), float(kappa), int(tau))
    return runs.get(key)


class _Line:
    """One curve, and one column, through the plane conditions."""

    def __init__(self, conditions, which, title, xlabel, color, marker):
        self.conditions = list(conditions)
        self.which = which
        self.title = title
        self.xlabel = xlabel
        self.color = color
        self.marker = marker


def _slices(plane_runs):
    """Lines through the κ_sens × τ grid.
    """
    line1 = [(k, t) for k, t in plane_runs if t == 1]
    line2 = [(k, t) for k, t in plane_runs if k is not None and float(k) == 1_000_000.0]
    by_tau = {}
    tau_order = []
    for kappa, tau in plane_runs:
        if kappa is None or tau is None or int(tau) == 1:
            continue
        tau = int(tau)
        if tau not in by_tau:
            by_tau[tau] = []
            tau_order.append(tau)
        if not any(float(kappa) == float(k) for k, _ in by_tau[tau]):
            by_tau[tau].append((kappa, tau))
    lines = []
    if line1:
        lines.append(_Line(
            line1, "kappa", r"κ_sens varies  (τ = 1)", r"κ_sens", _KAPPA_COLOR, "o",
        ))
    if line2:
        lines.append(_Line(
            line2, "tau", r"τ varies  (κ_sens = $10^{6}$)", r"τ", _TAU_COLOR, "o",
        ))
    for tau in tau_order:
        conds = by_tau[tau]
        if len(conds) <= 1:
            continue
        conds = sorted(conds, key=lambda pair: -float(pair[0]))
        lines.append(_Line(
            conds, "kappa", rf"κ_sens varies  (τ = {tau})", r"κ_sens", _LINE3_COLOR, "^",
        ))
    return lines


def _unique_conditions(plane_runs):
    """Each estimated condition once, in the order the lines first use it."""
    found = []
    for line in _slices(plane_runs):
        for pair in line.conditions:
            if pair not in found:
                found.append(pair)
    return found


def _setting_label(kappa, tau, which):
    if which == "kappa":
        return r"$10^{6}$" if float(kappa) >= 1e5 else f"{float(kappa):g}"
    return f"{int(tau)}"


def _values(runs, seeds, kappa, tau, value_of):
    found = []
    for seed in seeds:
        run = _run(runs, seed, kappa, tau)
        if run is None:
            continue
        found.append(float(value_of(run)))
    return np.asarray(found, dtype=float)


def _offsets(n):
    if n <= 1:
        return np.zeros(n)
    return np.linspace(-0.12, 0.12, n)


def _deg_label(values):
    """Held-normal error. A range when the seeds disagree, otherwise one value."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return ""
    lo, hi = float(np.min(vals)), float(np.max(vals))

    def fmt(x):
        return f"{x:.2f}" if x < 1 else f"{x:.1f}"

    if hi - lo < 0.05:
        return f"{fmt(float(np.mean(vals)))}°"
    return f"{fmt(lo)}–{fmt(hi)}°"


def _plane_rms(run):
    return float(run["filter_stats"]["rms_err_deg"])


def _score(run, name):
    """A summary stored on the run. The HGS floor lives beside the row."""
    row = run.get("row") or {}
    if name in row:
        return float(row[name])
    if name == "hgs_floor":
        return float(run["floor"])
    raise KeyError(name)


def _wrapped_final(run):
    """Last-step wrapped error, in radians. Cannot pass about 4.4 rad."""
    return float(run["result"].error_rad[-1])


def _unwrapped_total(run):
    """Last-step unwrapped error, in radians. Drift is not folded."""
    return float(run["result"].unwrapped_error_rad)


def _rad_ylim(values):
    """Log limits that include every positive value, and the 4.4 rad wrap ceiling."""
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return (0.02, 6)
    lo = min(0.02, float(np.min(vals)) / 1.8)
    hi = max(float(np.max(vals)) * 1.8, 6.0)
    return (lo, hi)


def _plot_two_slices(runs, plane_runs, seeds, rows, show_rms, guides=None):
    """One row per error. Dots are the seeds that were run, the bar is their mean.
    """
    lines = _slices(plane_runs)
    n_cols = len(lines)
    n = len(rows)
    guides = list(guides) if guides is not None else [None] * n
    width = 9.8 if n_cols == 2 else 9.8 * n_cols / 2
    fig, axes = plt.subplots(n, n_cols, figsize=(width, 3.35 * n), sharey="row", squeeze=False)
    for row, (value_of, ylabel, yticks, ylabels, ylim) in enumerate(rows):
        true_vals = _values(runs, seeds, None, None, value_of)
        true_mean = float(np.mean(true_vals)) if true_vals.size else np.nan
        show_plane = np.isfinite(true_mean) and ylim[0] < true_mean < ylim[1]
        last = row == n - 1
        for col, line in enumerate(lines):
            ax = axes[row, col]
            ax.set_yscale("log")
            ax.set_axisbelow(True)
            if show_plane:
                ax.axhline(true_mean, color="#9A9A9A", lw=1.05, zorder=2)
            for i, (kappa, tau) in enumerate(line.conditions):
                vals = _values(runs, seeds, kappa, tau, value_of)
                if vals.size == 0:
                    continue
                mean = float(np.mean(vals))
                mark = _ON_PLANE if mean < true_mean * 1.8 else line.color
                ax.plot([i - 0.28, i + 0.28], [mean, mean], color=mark, lw=2.4,
                        solid_capstyle="butt", zorder=3)
                ax.scatter(
                    np.full(len(vals), i) + _offsets(len(vals)), vals,
                    s=22, color=mark, zorder=4, linewidths=0, marker=line.marker,
                )
                if show_rms and last:
                    ax.annotate(
                        _deg_label(_values(runs, seeds, kappa, tau, _plane_rms)),
                        xy=(i, 0), xycoords=ax.get_xaxis_transform(),
                        xytext=(0, -17), textcoords="offset points",
                        ha="center", va="top", fontsize=8, color="0.2",
                    )
            ax.set_xticks(range(len(line.conditions)))
            ax.set_xticklabels([_setting_label(k, t, line.which) for k, t in line.conditions])
            ax.set_xlim(-0.55, len(line.conditions) - 0.45)
            if last:
                ax.set_xlabel(line.xlabel, labelpad=22 if show_rms else 4)
            if row == 0:
                ax.set_title(line.title, fontsize=11)
            if yticks is None:
                ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 3.0)))
            else:
                ax.yaxis.set_major_locator(FixedLocator(list(yticks)))
                ax.yaxis.set_major_formatter(FixedFormatter(list(ylabels)))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.grid(axis="y", which="major", color="0.87", lw=0.8)
        axes[row, 0].set_ylabel(ylabel)
        axes[row, 0].set_ylim(*ylim)
        for col in range(1, n_cols):
            axes[row, col].tick_params(labelleft=False)
        right = axes[row, n_cols - 1]
        guide = guides[row]
        if guide is not None and ylim[0] < guide[0] < ylim[1]:
            for col in range(n_cols):
                axes[row, col].axhline(
                    guide[0], color="0.25", lw=1.0, ls=(0, (4, 2)), zorder=1,
                )
            trans = blended_transform_factory(right.transAxes, right.transData)
            right.text(
                0.98, guide[0] * 1.25, guide[1], transform=trans,
                ha="right", va="bottom", fontsize=8, color="0.25",
            )
        if show_plane:
            trans = blended_transform_factory(right.transAxes, right.transData)
            right.text(
                0.98, true_mean * 1.35, "true plane", transform=trans,
                ha="right", va="bottom", fontsize=8, color="0.25",
            )
    n_seeds = len(tuple(seeds))
    axes[0, 0].legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_KAPPA_COLOR, markersize=5, label="seed"),
        Line2D([0], [0], color=_KAPPA_COLOR, lw=2.2, label=f"mean of {n_seeds} seeds"),
    ], frameon=False, fontsize=8, loc="upper left")
    if show_rms:
        axes[-1, 0].annotate(
            r"E$_{RMS}$", xy=(0, 0), xycoords=axes[-1, 0].get_xaxis_transform(),
            xytext=(-32, -17), textcoords="offset points",
            ha="right", va="top", fontsize=8, color="0.15",
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=(0.62 / fig.get_size_inches()[1]) if show_rms else 0.08)
    return fig


def plot_final_error(runs, plane_runs, seeds, half_spacing):
    """Wrapped final error and unwrapped total, in radians, for the two slices."""
    series = (
        (_wrapped_final, "wrapped final error\n(rad)"),
        (_unwrapped_total, "unwrapped total\n(rad)"),
    )
    half_grid = float(half_spacing)
    rows = []
    guides = []
    for value_of, ylabel in series:
        collected = [
            _values(runs, seeds, kappa, tau, value_of)
            for kappa, tau in [(None, None), *_unique_conditions(plane_runs)]
        ]
        vals = np.concatenate([*collected, [half_grid]])
        rows.append((value_of, ylabel, None, None, _rad_ylim(vals)))
        guides.append((half_grid, "½ bump spacing"))
    return _plot_two_slices(runs, plane_runs, seeds, rows, show_rms=True, guides=guides)


def plot_held_normal_error(runs, plane_runs, seeds):
    """Held-normal RMS for the same two slices. The true plane lies below this axis."""
    return _plot_two_slices(
        runs, plane_runs, seeds,
        rows=[(_plane_rms, "plane error\n(RMS deg)", (0.1, 1, 10), ("0.1", "1", "10"), (0.04, 60))],
        show_rms=False,
    )


def _pad_ylim(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        pad = max(abs(hi), 1.0) * 0.15
        lo, hi = hi - pad, hi + pad
    if lo >= 0:
        lo = 0.0
    return lo, hi + 0.18 * max(hi - lo, 1e-6)


def plot_score_sweeps(runs, plane_runs, seeds, metrics):
    """One row per score. One column per line. The line is the mean of the seeds.
    """
    lines = _slices(plane_runs)
    n_cols = len(lines)
    n = len(metrics)
    width = 9.2 if n_cols == 2 else 9.2 * n_cols / 2
    fig, axes = plt.subplots(n, n_cols, figsize=(width, 2.55 * n), sharey="row", squeeze=False)
    for col, line in enumerate(lines):
        for row, (name, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            true_level = float(np.mean(_values(
                runs, seeds, None, None, lambda run: _score(run, name),
            )))
            means, rms_row, rate_row = [], [], []
            for i, (kappa, tau) in enumerate(line.conditions):
                vals = _values(runs, seeds, kappa, tau, lambda run, name=name: _score(run, name))
                if vals.size == 0:
                    means.append(np.nan)
                    rms_row.append("")
                    rate_row.append(np.nan)
                    continue
                ax.scatter(
                    i + _offsets(len(vals)), vals, s=22, color=line.color,
                    alpha=0.45, zorder=3, linewidths=0, marker=line.marker,
                )
                means.append(float(np.mean(vals)))
                rms_row.append(_deg_label(_values(runs, seeds, kappa, tau, _plane_rms)))
                rate_row.append(float(np.mean(_values(
                    runs, seeds, kappa, tau,
                    lambda run: float(run["filter_stats"]["mean_rate_deg"]),
                ))))
            xs = np.arange(len(line.conditions))
            ax.plot(xs, means, color=line.color, lw=2, marker=line.marker, ms=6.5, zorder=4)
            ax.axhline(true_level, color="0.75", lw=1.2, zorder=0)
            ax.set_xticks(xs)
            ax.set_xticklabels([_setting_label(k, t, line.which) for k, t in line.conditions])
            ax.set_xlim(-0.45, len(line.conditions) - 0.55)
            ax.grid(True, color="0.92", lw=0.6)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(line.title)
                ax.text(0.98, true_level, "true plane", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=8, color="0.45")
            if row == n - 1:
                _label_plane_under_axis(ax, xs, rms_row, rate_row, line.xlabel, col == 0)
    for row, (name, _) in enumerate(metrics):
        collected = [_values(runs, seeds, k, t, lambda run, name=name: _score(run, name))
                     for k, t in _unique_conditions(plane_runs) + [(None, None)]]
        axes[row, 0].set_ylim(*_pad_ylim(np.concatenate(collected)))
    n_seeds = len(tuple(seeds))
    axes[0, 0].legend(handles=[
        Line2D([0], [0], marker="o", color="0.45", lw=0, alpha=0.55, label="seed"),
        Line2D([0], [0], marker="o", color="0.25", lw=2, label=f"mean of {n_seeds} seeds"),
    ], fontsize=8, frameon=False, loc="best")
    height = fig.get_size_inches()[1]
    fig.subplots_adjust(
        left=0.16, right=0.98, bottom=1.15 / height, top=0.94,
        hspace=0.48, wspace=0.18,
    )
    return fig


def _label_plane_under_axis(ax, xs, rms_row, rate_row, xlabel, label_rows):
    """Held-normal RMS, then its rate, sitting below the axis name."""
    ax.set_xlabel(xlabel, labelpad=2)
    for x, rms, rate in zip(xs, rms_row, rate_row):
        ax.annotate(
            rms, xy=(x, 0), xycoords=ax.get_xaxis_transform(),
            xytext=(0, -36), textcoords="offset points",
            ha="center", va="top", fontsize=8, color="0.35", annotation_clip=False,
        )
        ax.annotate(
            f"{rate:.2f}", xy=(x, 0), xycoords=ax.get_xaxis_transform(),
            xytext=(0, -50), textcoords="offset points",
            ha="center", va="top", fontsize=8, color="0.35", annotation_clip=False,
        )
    if label_rows:
        ax.annotate(
            "E_RMS", xy=(0, 0), xycoords=ax.get_xaxis_transform(),
            xytext=(-10, -36), textcoords="offset points",
            ha="right", va="top", fontsize=8, color="0.35", annotation_clip=False,
        )
        ax.annotate(
            "(°/step)", xy=(0, 0), xycoords=ax.get_xaxis_transform(),
            xytext=(-10, -50), textcoords="offset points",
            ha="right", va="top", fontsize=8, color="0.35", annotation_clip=False,
        )


def _rows_for(score_df, kappa, tau):
    frame = score_df
    if kappa is None:
        return frame[frame["kappa_sens"].astype(str) == "true"]
    kappa_num = frame["kappa_sens"].map(lambda v: np.nan if str(v) == "true" else float(v))
    tau_num = frame["tau_refresh"].map(lambda v: np.nan if str(v) == "true" else float(v))
    return frame[np.isclose(kappa_num, float(kappa)) & np.isclose(tau_num, float(tau))]


def _off_axis_label(kappa, tau):
    """A condition past the right edge of the error axis."""
    if float(kappa) >= 1e5:
        return f"τ = {int(tau)}, off axis"
    return f"κ = {float(kappa):g}, τ = {int(tau)}, off axis"


def _line_legend_label(line):
    if line.marker == "^":
        return rf"κ sweep  (τ = {int(line.conditions[0][1])})"
    if line.which == "kappa":
        return r"κ sweep  (τ = 1)"
    return r"τ sweep  (κ = $10^{6}$)"


def plot_scores_vs_plane_error(score_df, plane_runs, metrics, xmax=None):
    """Each score against held-normal RMS. One curve per line, and the true plane.

    xmax moves only the right edge. The left edge stays at -0.8. A condition
    whose mean RMS is above xmax is left off its curve and named in the legend.
    A condition that sits on two lines has its seed dots drawn once.
    """
    lines = _slices(plane_runs)
    n = len(metrics)
    ncols = 2 if n <= 2 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.45 * nrows), sharex=True, squeeze=False)
    flat = list(axes.ravel())
    off_labels = []
    seen_off = set()
    if xmax is not None:
        for line in lines:
            for kappa, tau in line.conditions:
                key = (float(kappa), int(tau))
                if key in seen_off:
                    continue
                group = _rows_for(score_df, kappa, tau)
                if group.empty:
                    continue
                if float(group["rms_err_deg"].mean()) > float(xmax):
                    seen_off.add(key)
                    off_labels.append(_off_axis_label(kappa, tau))
    for ax, (name, ylabel) in zip(flat, metrics):
        true = _rows_for(score_df, None, None)
        ax.scatter(true["rms_err_deg"], true[name], s=22, color="0.55", alpha=0.8, zorder=3, linewidths=0)
        if not true.empty:
            ax.scatter([true["rms_err_deg"].mean()], [true[name].mean()], s=42, color="0.25", zorder=5)
        drawn = set()
        for line in lines:
            mx, my = [], []
            for kappa, tau in line.conditions:
                group = _rows_for(score_df, kappa, tau)
                if group.empty:
                    continue
                mean_x = float(group["rms_err_deg"].mean())
                mean_y = float(group[name].mean())
                key = (float(kappa), int(tau))
                if xmax is not None and mean_x > float(xmax):
                    continue
                if key not in drawn:
                    ax.scatter(
                        group["rms_err_deg"], group[name], s=22, color=line.color,
                        alpha=0.45, zorder=3, linewidths=0, marker=line.marker,
                    )
                    drawn.add(key)
                mx.append(mean_x)
                my.append(mean_y)
            ax.plot(mx, my, color=line.color, lw=2, marker=line.marker, ms=6.5, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("E_RMS (deg)")
        ax.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        if xmax is None:
            ax.set_xlim(left=-0.8)
        else:
            ax.set_xlim(left=-0.8, right=xmax)
        ax.set_ylim(*_pad_ylim(score_df[name].to_numpy(dtype=float)))
    for ax in flat[n:]:
        ax.axis("off")
    legend_ax = flat[n] if n < len(flat) else flat[0]
    handles = [
        Line2D([0], [0], marker=line.marker, color=line.color, lw=2, label=_line_legend_label(line))
        for line in lines
    ]
    handles.extend(
        Line2D([0], [0], lw=0, color="none", label=label) for label in off_labels
    )
    handles.append(Line2D([0], [0], marker="o", color="0.25", lw=0, label="true plane"))
    legend_ax.legend(
        handles=handles, fontsize=8, frameon=False,
        loc="center" if n < len(flat) else "best",
    )
    fig.tight_layout()
    return fig


# (label, azimuth, altitude) in gy.oblique_slice's convention. Vertical slices have z pointing up.
_SLICES = (
    ("horizontal (x–y)", 0.0, 0.0),
    ("vertical (y–z)", 0.0, np.pi / 2),
    ("vertical (x–z)", np.pi / 2, np.pi / 2),
)


def percentile_slices(sums, counts, title, active_mask=None, percentiles=(10, 50, 90),
                      vmax=3.0, sigma=SMOOTH_SIGMA):
    """Scored cells at spatial-information percentiles, one column per cell, one row per slice."""
    scored_idx = np.where(active_mask)[0] if active_mask is not None else np.arange(sums.shape[-1])
    f = _rate_map_from_accumulator(sums[..., scored_idx], counts, sigma=sigma)
    p = counts / (counts.sum() + 1e-30)
    si = _spatial_info(p, f)
    idx = percentile_cells(si, percentiles)
    visited = counts > 0
    scale = dict(vmin=0, vmax=vmax) if vmax is not None else {}
    fig, axes = plt.subplots(
        len(_SLICES), len(idx), figsize=(3.2 * len(idx), 3.0 * len(_SLICES)), squeeze=False,
    )
    for col, k in enumerate(idx):
        g = f[..., k] / f[..., k][visited].mean()
        cv = f[..., k][visited].std() / f[..., k][visited].mean()
        for row, (name, az, al) in enumerate(_SLICES):
            sl = gy.oblique_slice(g, az, al, bins=g.shape[0])[..., 0]
            ax = axes[row, col]
            im = ax.imshow(sl, origin="lower", cmap="viridis", **scale)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"rate  p{percentiles[col]}\nSI {si[k]:.2f}  CV {cv:.2f}", fontsize=10)
            if col == 0:
                ax.set_ylabel(name)
    fig.suptitle(title)
    if vmax is not None:
        fig.colorbar(im, ax=axes, shrink=0.6, label="rate / map mean")
    return fig


#colloirs pr condition
_CONDITION_COLOR = ("#1F1F1F", "#3D7EBF", "#E07A4C", "#3C9A6E")
_PROTO_COLOR = {
    "fcc": "#A7C4DE",
    "hcp": "#E3B89A",
    "col": "#A3CDB0",
    "rnd": "#5C5C5C",
}
_PROTO_LABEL = {"fcc": "FCC", "hcp": "HCP", "col": "COL", "rnd": "RND"}
_PROTO_ORDER = ("fcc", "hcp", "col", "rnd")
# Diagonal of the 1.29 m kept cube. The bin edges are 0.1 m, the rate-map bin.
_IFD_BINS = np.linspace(0.0, 2.3, 24)


def _plane_label(kappa, tau):
    if kappa is None:
        return "true plane"
    kappa_s = r"$10^{6}$" if float(kappa) >= 1e5 else f"{float(kappa):g}"
    return rf"κ_sens = {kappa_s}, τ = {int(tau)}"


def _condition_scores(runs, seeds, kappa, tau):
    found = []
    for seed in seeds:
        run = _run(runs, seed, kappa, tau)
        if run is None:
            continue
        scores = run.get("scores")
        if scores is not None:
            found.append(scores)
    return found


def _mean_mra_curves(score_list):
    """One mean MRA per score dict, and the rate-map bin count.

    A prototype is one dict, so its curve is the mean across cells. A network
    condition is one dict per seed. The band is the range of those means.
    A band across cells would be far too tight, because the cells share one sheet.
    """
    means = []
    bins = None
    for scores in score_list:
        mra = mra_curve(scores)
        if mra is None:
            continue
        with np.errstate(invalid="ignore"):
            means.append(np.nanmean(mra, axis=0))
        bins = int(scores["bins"])
    if not means or bins is None:
        return None
    width = min(len(m) for m in means)
    return np.vstack([m[:width] for m in means]), bins


def _join(score_list, take):
    """Stack one stored vector from each score dict."""
    parts = [p for p in (take(s) for s in score_list) if len(p)]
    return np.concatenate(parts) if parts else np.empty(0)


def _step_hist(ax, values, edges, **kwargs):
    values = np.asarray(values, dtype=float).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return
    ax.hist(values, bins=edges, density=True, histtype="step", **kwargs)


def plot_mra_curves(runs, plane_runs, seeds, proto_scores, env_size):
    """Mean MRA against distance. One solid line per plane condition. """
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    ax.axhline(0.0, color="0.75", lw=1.0, zorder=1)
    for kind in _PROTO_ORDER:
        proto = proto_scores.get(kind) if proto_scores else None
        if proto is None:
            continue
        pooled = _mean_mra_curves([proto])
        if pooled is None:
            continue
        curves, bins = pooled
        ax.plot(
            mra_radii_m(curves.shape[1], bins, env_size), curves[0],
            color=_PROTO_COLOR[kind], lw=1.3, ls="--" if kind != "rnd" else ":",
            zorder=2, label=_PROTO_LABEL[kind],
        )
    for i, (kappa, tau) in enumerate(plane_runs):
        pooled = _mean_mra_curves(_condition_scores(runs, seeds, kappa, tau))
        if pooled is None:
            continue
        stacked, bins = pooled
        with np.errstate(invalid="ignore"):
            mid = np.nanmean(stacked, axis=0)
            lo = np.nanmin(stacked, axis=0)
            hi = np.nanmax(stacked, axis=0)
        x = mra_radii_m(len(mid), bins, env_size)
        color = _CONDITION_COLOR[i % len(_CONDITION_COLOR)]
        if stacked.shape[0] > 1:
            ax.fill_between(x, lo, hi, color=color, alpha=0.18, lw=0, zorder=2)
        ax.plot(x, mid, color=color, lw=2.0, zorder=3, label=_plane_label(kappa, tau))
    ax.set_xlabel("distance (m)")
    ax.set_ylabel("MRA")
    ax.set_xlim(left=0.0)
    ax.grid(True, color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig


# First two distances of a 0.71 m hexagonal lattice. 1.23 m is √3 × 0.71.
_IFD_GUIDES = (0.71, 1.23)


def _lattice_reference(proto_scores):
    """The lattice drawn behind each distance panel. FCC when it was scored."""
    for kind in ("fcc", "hcp", "col"):
        proto = proto_scores.get(kind)
        if proto is not None:
            return kind, proto
    return None


def plot_ifd_histograms(runs, plane_runs, seeds, proto_scores, env_size):
    """All-pairs inter-field distances, one panel per plane condition..
    Every curve uses the same bins on [0, 2.3] m and is normalised on its own.
    """
    panels = list(plane_runs)
    n = len(panels)
    ncols = 2 if n > 2 else max(n, 1)
    nrows = int(np.ceil(n / ncols)) if n else 1
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 2.8 * nrows),
        sharex=True, sharey=True, squeeze=False,
    )
    proto_scores = proto_scores or {}
    lattice = _lattice_reference(proto_scores)
    rnd = proto_scores.get("rnd")
    flat = axes.ravel()
    for i, (kappa, tau) in enumerate(panels):
        ax = flat[i]
        for guide in _IFD_GUIDES:
            ax.axvline(guide, color="0.45", lw=0.8, zorder=1)
        if rnd is not None:
            _step_hist(
                ax, normalized_to_metres(pooled_ifd(rnd), env_size), _IFD_BINS,
                color="0.55", lw=1.2, zorder=2,
            )
        if lattice is not None:
            kind, proto = lattice
            _step_hist(
                ax, normalized_to_metres(pooled_ifd(proto), env_size), _IFD_BINS,
                color=_PROTO_COLOR[kind], lw=1.3, linestyle="--", zorder=2,
            )
        distances = _join(_condition_scores(runs, seeds, kappa, tau), pooled_ifd)
        _step_hist(
            ax, normalized_to_metres(distances, env_size), _IFD_BINS,
            color=_CONDITION_COLOR[i % len(_CONDITION_COLOR)], lw=1.8, zorder=3,
        )
        ax.set_title(_plane_label(kappa, tau), fontsize=10)
        ax.set_xlim(0.0, 2.3)
        ax.grid(True, axis="y", color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        if i % ncols == 0:
            ax.set_ylabel("density")
        if i >= n - ncols:
            ax.set_xlabel("inter-field distance (m)")
    for ax in flat[n:]:
        ax.set_visible(False)
    if n:
        top = flat[0].get_ylim()[1]
        for ax in flat[:n]:
            for guide in _IFD_GUIDES:
                ax.text(
                    guide, top, f"{guide:.2f}", ha="center", va="top",
                    fontsize=7, color="0.4",
                )
    def _legend_handles():
        handles = []
        if rnd is not None:
            handles.append(Line2D([0], [0], color="0.55", lw=1.2, label="RND"))
        if lattice is not None:
            kind, _proto = lattice
            handles.append(Line2D(
                [0], [0], color=_PROTO_COLOR[kind], lw=1.3, ls="--",
                label=_PROTO_LABEL[kind],
            ))
        return handles

    if _legend_handles():
        for ax in flat[:n]:
            ax.legend(handles=_legend_handles(), frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    return fig


def _ecdf(values):
    values = np.asarray(values, dtype=float).ravel()
    values = np.sort(values[np.isfinite(values)])
    if values.size == 0:
        return None
    return values, np.arange(1, values.size + 1) / values.size


def plot_cv_distributions(runs, plane_runs, seeds, proto_scores):
    """Cumulative distribution of the 3 nearest neighbour spacing CV"""
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    rnd = np.empty(0)
    if proto_scores and proto_scores.get("rnd") is not None:
        rnd = pooled_ifd_cv(proto_scores["rnd"])
    curve = _ecdf(rnd)
    if curve is not None:
        ax.step(
            curve[0], curve[1], where="post", color=_PROTO_COLOR["rnd"],
            lw=1.6, linestyle=":", label="RND", zorder=2,
        )
    for i, (kappa, tau) in enumerate(plane_runs):
        curve = _ecdf(_join(_condition_scores(runs, seeds, kappa, tau), pooled_ifd_cv))
        if curve is None:
            continue
        ax.step(
            curve[0], curve[1], where="post",
            color=_CONDITION_COLOR[i % len(_CONDITION_COLOR)], lw=2.0,
            label=_plane_label(kappa, tau), zorder=3,
        )
    ax.set_xlabel("local spacing CV")
    ax.set_ylabel("cumulative fraction")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", color="0.92", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig

