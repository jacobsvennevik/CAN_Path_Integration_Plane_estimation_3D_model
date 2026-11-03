"""Figures that compare stored experiment runs across plane settings.
"""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedFormatter, FixedLocator, NullLocator
from matplotlib.transforms import blended_transform_factory


# The experiments compare two slices, not a full grid of κ_sens and τ.
_KAPPA_COLOR = "#3D7EBF"
_TAU_COLOR = "#E07A4C"
_ON_PLANE = "#6E6E6E"


def _run(runs, seed, kappa, tau):
    """One stored run. The true plane is saved under ('true', 'true')."""
    if kappa is None:
        key = (int(seed), "true", "true")
    else:
        key = (int(seed), float(kappa), int(tau))
    return runs[key]


def _kappa_slice(plane_runs):
    """κ_sens settings that were run at τ = 1."""
    return [(k, t) for k, t in plane_runs if t == 1]


def _tau_slice(plane_runs):
    """τ settings that were run at κ_sens = 10^6."""
    return [(k, t) for k, t in plane_runs if k is not None and float(k) == 1_000_000.0]


def _setting_label(kappa, tau, which):
    if which == "kappa":
        return r"$10^{6}$" if float(kappa) >= 1e5 else f"{float(kappa):g}"
    return f"{int(tau)}"


def _values(runs, seeds, kappa, tau, value_of):
    return np.asarray([
        float(value_of(_run(runs, seed, kappa, tau))) for seed in seeds
    ])


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


def _plot_two_slices(runs, plane_runs, seeds, value_of, ylabel, yticks, ylabels, ylim, show_rms):
    """Plot the path integretion error. The 3 different dots are seeds, the bar is their mean. The horizontal axis lists the settings that were run."""
    panels = (
        (_kappa_slice(plane_runs), "kappa", r"κ_sens varies  (τ = 1)", r"κ_sens", _KAPPA_COLOR),
        (_tau_slice(plane_runs), "tau", r"τ varies  (κ_sens = $10^{6}$)", r"τ", _TAU_COLOR),
    )
    true_mean = float(np.mean(_values(runs, seeds, None, None, value_of)))
    show_plane = ylim[0] < true_mean < ylim[1]
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.55), sharey=True)
    offsets = np.linspace(-0.12, 0.12, len(tuple(seeds)))
    for ax, (conditions, which, title, xlabel, color) in zip(axes, panels):
        ax.set_yscale("log")
        ax.set_axisbelow(True)
        if show_plane:
            ax.axhline(true_mean, color="#9A9A9A", lw=1.05, zorder=2)
        for i, (kappa, tau) in enumerate(conditions):
            vals = _values(runs, seeds, kappa, tau, value_of)
            mean = float(np.mean(vals))
            mark = _ON_PLANE if mean < true_mean * 1.8 else color
            ax.plot([i - 0.28, i + 0.28], [mean, mean], color=mark, lw=2.4,
                    solid_capstyle="butt", zorder=3)
            ax.scatter(np.full(len(vals), i) + offsets, vals, s=22, color=mark, zorder=4, linewidths=0)
            if show_rms:
                rms = float(np.mean(_values(runs, seeds, kappa, tau, _plane_rms)))
                ax.annotate(
                    f"{rms:.1f}°", xy=(i, 0), xycoords=ax.get_xaxis_transform(),
                    xytext=(0, -17), textcoords="offset points",
                    ha="center", va="top", fontsize=8, color="0.2",
                )
        ax.set_xticks(range(len(conditions)))
        ax.set_xticklabels([_setting_label(k, t, which) for k, t in conditions])
        ax.set_xlim(-0.55, len(conditions) - 0.45)
        ax.set_xlabel(xlabel, labelpad=22 if show_rms else 4)
        ax.set_title(title, fontsize=11)
        ax.yaxis.set_major_locator(FixedLocator(list(yticks)))
        ax.yaxis.set_major_formatter(FixedFormatter(list(ylabels)))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.grid(axis="y", which="major", color="0.87", lw=0.8)
    axes[0].set_ylabel(ylabel)
    axes[0].set_ylim(*ylim)
    axes[1].tick_params(labelleft=False)
    if show_plane:
        trans = blended_transform_factory(axes[1].transAxes, axes[1].transData)
        axes[1].text(0.98, true_mean * 1.35, "true plane", transform=trans,
                     ha="right", va="bottom", fontsize=8, color="0.25")
    axes[0].legend(handles=[
        Line2D([0], [0], marker="o", color="none", markerfacecolor=_KAPPA_COLOR, markersize=5, label="seed"),
        Line2D([0], [0], color=_KAPPA_COLOR, lw=2.2, label="mean of 3 seeds"),
    ], frameon=False, fontsize=8, loc="upper left")
    if show_rms:
        axes[0].annotate(
            r"E$_{RMS}$", xy=(0, 0), xycoords=axes[0].get_xaxis_transform(),
            xytext=(-32, -17), textcoords="offset points",
            ha="right", va="top", fontsize=8, color="0.15",
        )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.22 if show_rms else 0.14)
    return fig


def plot_final_error(runs, plane_runs, seeds):
    """Final wrapped error for the two slices that were run"""
    return _plot_two_slices(
        runs, plane_runs, seeds, lambda run: float(run["result"].error_rad[-1]),
        ylabel="final error (rad)\nwrapped, last step",
        yticks=(0.03, 0.3, 1, 3), ylabels=("0.03", "0.3", "1", "3"),
        ylim=(0.02, 6), show_rms=True,
    )


def plot_held_normal_error(runs, plane_runs, seeds):
    """Held-normal RMS for the same two slices. The true plane lies below this axis."""
    return _plot_two_slices(
        runs, plane_runs, seeds, _plane_rms,
        ylabel="plane error\n(RMS deg)",
        yticks=(0.1, 1, 10), ylabels=("0.1", "1", "10"),
        ylim=(0.08, 60), show_rms=False,
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
    """One row per score. Left varies kappa_sens, right varies tau_refresh. The line is the mean of the seeds.
    """
    panels = (
        (_kappa_slice(plane_runs), "kappa", r"κ_sens varies  (τ = 1)", r"κ_sens", _KAPPA_COLOR),
        (_tau_slice(plane_runs), "tau", r"τ varies  (κ_sens = $10^{6}$)", r"τ", _TAU_COLOR),
    )
    n = len(metrics)
    fig, axes = plt.subplots(n, 2, figsize=(9.2, 2.55 * n), sharey="row", squeeze=False)
    for col, (conditions, which, title, xlabel, color) in enumerate(panels):
        for row, (name, ylabel) in enumerate(metrics):
            ax = axes[row, col]
            true_level = float(np.mean(_values(
                runs, seeds, None, None, lambda run: _score(run, name),
            )))
            means, rms_row, rate_row = [], [], []
            for i, (kappa, tau) in enumerate(conditions):
                vals = _values(runs, seeds, kappa, tau, lambda run, name=name: _score(run, name))
                offset = np.linspace(-0.12, 0.12, len(vals))
                ax.scatter(i + offset, vals, s=22, color=color, alpha=0.45, zorder=3, linewidths=0)
                means.append(float(np.mean(vals)))
                rms_row.append(float(np.mean(_values(runs, seeds, kappa, tau, _plane_rms))))
                rate_row.append(float(np.mean(_values(
                    runs, seeds, kappa, tau,
                    lambda run: float(run["filter_stats"]["mean_rate_deg"]),
                ))))
            xs = np.arange(len(conditions))
            ax.plot(xs, means, color=color, lw=2, marker="o", ms=6.5, zorder=4)
            ax.axhline(true_level, color="0.75", lw=1.2, zorder=0)
            ax.set_xticks(xs)
            ax.set_xticklabels([_setting_label(k, t, which) for k, t in conditions])
            ax.set_xlim(-0.45, len(conditions) - 0.55)
            ax.grid(True, color="0.92", lw=0.6)
            ax.set_axisbelow(True)
            if col == 0:
                ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(title)
                ax.text(0.98, true_level, "true plane", transform=ax.get_yaxis_transform(),
                        ha="right", va="bottom", fontsize=8, color="0.45")
            if row == n - 1:
                _label_plane_under_axis(ax, xs, rms_row, rate_row, xlabel, col == 0)
    for row, (name, _) in enumerate(metrics):
        collected = [_values(runs, seeds, k, t, lambda run, name=name: _score(run, name))
                     for k, t in _kappa_slice(plane_runs) + _tau_slice(plane_runs) + [(None, None)]]
        axes[row, 0].set_ylim(*_pad_ylim(np.concatenate(collected)))
    axes[0, 0].legend(handles=[
        Line2D([0], [0], marker="o", color="0.45", lw=0, alpha=0.55, label="seed"),
        Line2D([0], [0], marker="o", color="0.25", lw=2, label="mean of 3 seeds"),
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
            f"{rms:.1f}°", xy=(x, 0), xycoords=ax.get_xaxis_transform(),
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


def plot_scores_vs_plane_error(score_df, plane_runs, metrics):
    """Each score against held-normal RMS. One line per slice, and the true plane at the left."""
    kappa_slice = _kappa_slice(plane_runs)
    tau_slice = _tau_slice(plane_runs)
    n = len(metrics)
    ncols = 2 if n <= 2 else 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 3.45 * nrows), sharex=True, squeeze=False)
    flat = list(axes.ravel())
    for ax, (name, ylabel) in zip(flat, metrics):
        true = _rows_for(score_df, None, None)
        ax.scatter(true["rms_err_deg"], true[name], s=22, color="0.55", alpha=0.8, zorder=3, linewidths=0)
        ax.scatter([true["rms_err_deg"].mean()], [true[name].mean()], s=42, color="0.25", zorder=5)
        for conditions, color in ((kappa_slice, _KAPPA_COLOR), (tau_slice, _TAU_COLOR)):
            mx, my = [], []
            for kappa, tau in conditions:
                group = _rows_for(score_df, kappa, tau)
                if not (color == _TAU_COLOR and int(tau) == 1):
                    ax.scatter(group["rms_err_deg"], group[name], s=22, color=color,
                               alpha=0.45, zorder=3, linewidths=0)
                mx.append(float(group["rms_err_deg"].mean()))
                my.append(float(group[name].mean()))
            ax.plot(mx, my, color=color, lw=2, marker="o", ms=6.5, zorder=4)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("E_RMS (deg)")
        ax.grid(True, color="0.92", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlim(left=-0.8)
        ax.set_ylim(*_pad_ylim(score_df[name].to_numpy(dtype=float)))
    for ax in flat[n:]:
        ax.axis("off")
    legend_ax = flat[n] if n < len(flat) else flat[0]
    legend_ax.legend(handles=[
        Line2D([0], [0], marker="o", color=_KAPPA_COLOR, lw=2, label=r"κ sweep  (τ = 1)"),
        Line2D([0], [0], marker="o", color=_TAU_COLOR, lw=2, label=r"τ sweep  (κ = $10^{6}$)"),
        Line2D([0], [0], marker="o", color="0.25", lw=0, label="true plane"),
    ], fontsize=8, frameon=False, loc="center" if n < len(flat) else "best")
    fig.tight_layout()
    return fig


def _active_grid_cells(run):
    return [c for c in run["lattice"]["cells"] if c["n_spikes"] > 0 and np.isfinite(c["grid_fit"])]


def _finite(values):
    if values is None:
        return np.array([])
    a = np.asarray(values, dtype=float).ravel()
    return a[np.isfinite(a)]


def _hist_edges(values, n=21, include=()):
    """Shared bin edges. ``include`` keeps a reference line inside the axis."""
    vals = _finite(values)
    extra = _finite(include)
    pool = np.concatenate([vals, extra]) if extra.size else vals
    if pool.size == 0:
        return np.linspace(0.0, 1.0, n)
    lo, hi = float(pool.min()), float(pool.max())
    if hi - lo < 1e-9:
        pad = max(abs(hi), 1.0) * 0.05
        lo, hi = hi - pad, hi + pad
    return np.linspace(lo, hi, n)


def _stored_scores(runs, seed, kappa, tau):
    return _run(runs, seed, kappa, tau).get("scores") or {}


_CHI_COLORS = (("#2166ac", "FCC"), ("#d6604d", "HCP"), ("#67a9cf", "COL"))


def _pooled_chi(runs, plane_runs, seeds):
    """Three structure-score rows per condition, seeds pooled."""
    out = {}
    for kappa, tau in plane_runs:
        rows = [[], [], []]
        for seed in seeds:
            chi = _stored_scores(runs, seed, kappa, tau).get("chi")
            if chi is None:
                continue
            chi = np.asarray(chi, dtype=float)
            if chi.ndim != 2:
                continue
            for i in range(min(3, chi.shape[0])):
                rows[i].append(_finite(chi[i]))
        out[(kappa, tau)] = [
            np.concatenate(parts) if parts else np.array([]) for parts in rows
        ]
    return out


def _pooled_score(runs, plane_runs, seeds, name):
    out = {}
    for kappa, tau in plane_runs:
        parts = [
            _finite(_stored_scores(runs, seed, kappa, tau).get(name))
            for seed in seeds
        ]
        parts = [p for p in parts if p.size]
        out[(kappa, tau)] = np.concatenate(parts) if parts else np.array([])
    return out


def plot_field_separations(runs, plane_runs, seeds, grid_spacing, env_size):
    """All-pairs separations, and the nearest-neighbour CV, one panel per condition.

    The first figure is every remaining pair of centres, which is what Gong &
    Yu's code stores, with a line at one grid spacing. The second is the CV of
    each field's three nearest neighbours. That CV is defined only for cells
    with at least four fields, and a low value is local consistency.
    """
    spacing_n = float(grid_spacing) / (float(env_size) / 2.0)
    pooled = _pooled_ifd(runs, plane_runs, seeds)
    pair_values = [v for v in pooled.values() if v.size]
    pair_edges = _hist_edges(
        np.concatenate(pair_values) if pair_values else np.array([]),
        include=(spacing_n,),
    )

    def draw_pairs(ax, kappa, tau):
        values = pooled[(kappa, tau)]
        if values.size:
            ax.hist(values, bins=pair_edges, color="#4d4d4d", alpha=0.85)
        ax.axvline(spacing_n, color="#d6604d", ls="--", lw=1)

    pairs = _condition_panels(
        plane_runs,
        "every pair of field centres, three seeds pooled",
        "inter-field distance (normalised)",
        draw_pairs,
        handles=[Line2D([0], [0], color="#d6604d", ls="--", lw=1.2, label="one grid spacing")],
    )

    cv = _pooled_score(runs, plane_runs, seeds, "ifd_cv")
    cv_values = [v for v in cv.values() if v.size]
    cv_edges = _hist_edges(np.concatenate(cv_values) if cv_values else np.array([]))

    def draw_cv(ax, kappa, tau):
        values = cv[(kappa, tau)]
        if values.size:
            ax.hist(values, bins=cv_edges, color="#2166ac", alpha=0.85)

    local = _condition_panels(
        plane_runs,
        "local spacing consistency, three seeds pooled",
        "CV of three nearest neighbours",
        draw_cv,
    )
    return pairs, local


def _pooled_ifd(runs, plane_runs, seeds):
    out = {}
    for kappa, tau in plane_runs:
        parts = []
        for seed in seeds:
            ifd = _stored_scores(runs, seed, kappa, tau).get("ifd") or {}
            parts.extend(
                _finite(v) for v in ifd.values() if len(v)
            )
        parts = [p for p in parts if p.size]
        out[(kappa, tau)] = np.concatenate(parts) if parts else np.array([])
    return out


def _condition_panels(plane_runs, title, xlabel, draw, handles=None, legend_on_first=False):
    """One panel per plane condition. ``draw(ax, kappa, tau)`` fills that panel."""
    n = len(plane_runs)
    nrows = int(np.ceil(n / 4))
    fig, axes = plt.subplots(nrows, 4, figsize=(12, 2.7 * nrows), sharex=True, sharey=True, squeeze=False)
    axes = list(axes.ravel())
    for i, (ax, (kappa, tau)) in enumerate(zip(axes, plane_runs)):
        draw(ax, kappa, tau)
        ax.set_title("true plane" if kappa is None else f"κ={kappa:g}  τ={int(tau)}")
        if i == 0 and legend_on_first:
            ax.legend(fontsize=8)
    for ax in axes[n:]:
        ax.axis("off")
    if handles:
        fig.legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(title, y=1.02)
    fig.supxlabel(xlabel)
    fig.tight_layout()
    return fig


def plot_lattice_histograms(runs, plane_runs, seeds, grid_spacing):
    """Grid fit, fitted spacing, and orientation. One panel per plane condition.

    Seeds are pooled for the first two figures. Orientation stays split by seed,
    because each seed flies a different plane.
    """
    cells = {
        (k, t): [c for seed in seeds for c in _active_grid_cells(_run(runs, seed, k, t))]
        for k, t in plane_runs
    }
    fits = [c["grid_fit"] for group in cells.values() for c in group]
    fit_edges = np.linspace(min(fits), max(fits), 21)
    spacing = [c["spacing"] for group in cells.values() for c in group]
    space_edges = np.linspace(min(min(spacing), grid_spacing), max(max(spacing), grid_spacing), 21)

    def draw_fit(ax, kappa, tau):
        group = cells[(kappa, tau)]
        rest = [c["grid_fit"] for c in group if not c["significant"]]
        sig = [c["grid_fit"] for c in group if c["significant"]]
        if rest:
            ax.hist(rest, bins=fit_edges, color="#67a9cf", alpha=0.85)
        if sig:
            ax.hist(sig, bins=fit_edges, color="#d6604d", alpha=0.75)

    def draw_spacing(ax, kappa, tau):
        values = [c["spacing"] for c in cells[(kappa, tau)]]
        if values:
            ax.hist(values, bins=space_edges, color="#2166ac", alpha=0.8)
        ax.axvline(grid_spacing, color="#d6604d", ls="--", lw=1)

    def draw_orientation(ax, kappa, tau):
        for seed in seeds:
            run = _run(runs, seed, kappa, tau)
            angles = [
                float(run["lattice"]["orientations"][int(idx)])
                for c in _active_grid_cells(run) if c["significant"]
                for idx in np.asarray(c["orientation"]).ravel() if int(idx) >= 0
            ]
            if angles:
                ax.hist(angles, bins=np.arange(0, 61, 5), alpha=0.45, label=f"seed {seed}")

    return (
        _condition_panels(plane_runs, "per-cell grid fit, three seeds pooled", "grid fit", draw_fit, handles=[
            Patch(facecolor="#67a9cf", alpha=0.85, label="other"),
            Patch(facecolor="#d6604d", alpha=0.75, label="significant"),
        ]),
        _condition_panels(plane_runs, "fitted spacing, three seeds pooled", "fitted spacing (m)", draw_spacing),
        _condition_panels(
            plane_runs, "orientation of significant cells", "orientation (deg)",
            draw_orientation, legend_on_first=True,
        ),
    )

