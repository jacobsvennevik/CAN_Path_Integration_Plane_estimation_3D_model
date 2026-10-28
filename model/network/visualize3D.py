"""
Extending the visualization module of the MADE package for manifolds, CANs, and QANs.
Because it is hard to visualize a 4D object (3D volume plotted in one more dimension).
We break the 3D space into 2D volumes. Look in notebooks for simulations

This module provides functions to visualize:
1. CAN activity states
2. QAN trajectories, bump tracking, and path-integration error
"""
from made.visuals import clean_axes
from model.metrics import wrapped_angle_diff, unwrap_torus

import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

# defined once
def slice_specs(d: int):
    """Axis-aligned 2-D slices of T^d. Each spec is (d0, d1, d_fixed, xlabel, ylabel)."""
    names = [f"θ{i+1}" for i in range(d)]
    specs = []
    for d0, d1 in combinations(range(d), 2):
        fixed = [i for i in range(d) if i not in (d0, d1)]
        d_fixed = fixed[0] if fixed else None
        specs.append((d0, d1, d_fixed, names[d0], names[d1]))
    return specs


SLICE_SPECS = slice_specs(3)
_AXIS_NAMES = ["θ₁", "θ₂", "θ₃"]


    
def _format_torus_ax(ax, xlabel, ylabel, period=None):
    """Standard torus axis formating"""
    if period is None:
        period = 2 * np.pi
    period = float(period)
    clean_axes(ax, title=f"{xlabel} vs {ylabel}", ylabel=ylabel)
    ax.set_xlabel(xlabel)
    ax.set_xlim(0, period)
    ax.set_ylim(0, period)
    ticks = [0, period / 2, period]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if np.isclose(period, 2 * np.pi):
        labels = ["0", "π", "2π"]
    else:
        labels = [f"{t:.0f}" for t in ticks]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_aspect("equal")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=8)
    

def _break_periodic_jumps_2d(x, y, threshold=np.pi):
    """
    Insert NaNs where a 2D projected torus trajectory crosses a periodic boundary wrapping back from 2pi to 0.
    Needed so that matplotlib does not draw line between them.
    """
    x_plot = x.copy().astype(float)
    y_plot = y.copy().astype(float)

    dx = np.abs(np.diff(x_plot))
    dy = np.abs(np.diff(y_plot))

    jumps = np.where((dx > threshold) | (dy > threshold))[0] + 1

    x_plot[jumps] = np.nan
    y_plot[jumps] = np.nan

    return x_plot, y_plot


def _fold_onto_period(theta, period=2 * np.pi):
    """Fold angles into [0, period) with wrapped_angle_diff.

    The tracker accumulates an unwrapped phase. Ground truth is already on
    one period. Both have to sit in the same interval before a torus plot.
    """
    theta = np.asarray(theta, dtype=float)
    period = np.asarray(period, dtype=float)
    return np.mod(wrapped_angle_diff(theta, 0.0, period=period), period)


def visualize_trajectory_projections(traj, decoded=None, title="Torus trajectory",
                                     period=None):
    """
    Plot the different slices, works like the rest of the code fixing one of the dimensions
    """
    sheet = 2 * np.pi if period is None else period
    traj = _fold_onto_period(traj, sheet)
    if decoded is not None:
        decoded = _fold_onto_period(decoded, sheet)
    d = traj.shape[1]
    specs = slice_specs(d)
    n_panels = max(len(specs), 1)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    axes = np.atleast_1d(axes)
    thresh = None if period is None else 0.5 * float(np.mean(np.asarray(period)))

    for ax, spec in zip(axes, specs):
        d0, d1, d_fixed, xlabel, ylabel = spec
        kw = {} if thresh is None else {"threshold": thresh}
        # Ground-truth trajectory
        x, y = _break_periodic_jumps_2d(traj[:, d0], traj[:, d1], **kw)
        ax.plot(
            x, y, color="#2166ac", linewidth=1.2, alpha=0.8, label="ground truth",)
        # Mark the ground-truth start point
        ax.scatter(
            traj[0, d0], traj[0, d1],
            color="#2166ac", s=40,
            zorder=5,
        )
        # Optional decoded trajectory, used when we have the grid cell firing decoded positions and maps them on the trajectories
        if decoded is not None:
            # To help visualisation when the plot reaches a periodic boundary
            x_dec, y_dec = _break_periodic_jumps_2d(
                decoded[:, d0], decoded[:, d1], **kw)
            ax.plot(
                x_dec, y_dec,
                color="#d6604d", linewidth=1.0, linestyle="--",
                alpha=0.8, label="decoded",)
            # Mark the decoded start point
            ax.scatter(decoded[0, d0], decoded[0, d1], color="#d6604d", s=40, zorder=5)
        p = None if period is None else np.asarray(period).ravel()
        ax_period = None if p is None else float(p[d0])
        _format_torus_ax(ax, xlabel, ylabel, period=ax_period)

    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig, axes


def plot_pi_error(gt, decoded, title="Path-integration error", steps=True):
    """Torus positions in radians.

    The left panel is the accumulated error. With ``steps=True``, the right
    panel is the typical decoded step against the true step. ``steps=False``
    drops that panel.
    """
    gt, decoded = np.asarray(gt), np.asarray(decoded)
    if max(np.max(np.abs(gt)), np.max(np.abs(decoded))) < 1.5:
        print("WARNING: input looks like metres, not radians.")

    accumulated = unwrap_torus(decoded) - unwrap_torus(gt)
    err = np.linalg.norm(accumulated, axis=1)
    gt_speed = np.linalg.norm(wrapped_angle_diff(gt[1:], gt[:-1]), axis=1)
    path_length = np.concatenate([[0.0], np.cumsum(gt_speed)])
    score = float(err.mean() / (path_length.mean() + 1e-12))
    med = float(np.median(err))

    if steps:
        true_step = wrapped_angle_diff(gt[1:], gt[:-1]).ravel()
        dec_step = wrapped_angle_diff(decoded[1:], decoded[:-1]).ravel()
        # Median inside each bin. 
        lim = float(np.abs(true_step).max()) * 1.1
        nb = 11
        edges = np.linspace(-lim, lim, nb + 1)
        idx = np.clip(np.digitize(true_step, edges) - 1, 0, nb - 1)
        min_count = max(5, int(0.005 * true_step.size))
        ctr, typical = [], []
        for k in range(nb):
            take = idx == k
            if int(take.sum()) < min_count:
                continue
            # Median true step
            ctr.append(float(np.median(true_step[take])))
            typical.append(float(np.median(dec_step[take])))
        ctr = np.asarray(ctr, dtype=float)
        typical = np.asarray(typical, dtype=float)
        if ctr.size and float(ctr @ ctr) > 1e-18:
            integration_gain = float(ctr @ typical / (ctr @ ctr))
        else:
            integration_gain = float("nan")
        fig, ax = plt.subplots(
            1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})
        err_ax = ax[0]
    else:
        integration_gain = None
        fig, err_ax = plt.subplots(figsize=(6, 3.5))
        ax = err_ax

    err_ax.plot(err, color="#d6604d", lw=0.4, alpha=0.45)
    win = max(1, len(err) // 200)
    if win > 1:
        smooth = np.convolve(err, np.ones(win) / win, mode="valid")
        err_ax.plot(np.arange(win - 1, len(err)), smooth, color="#2166ac", lw=1.4,
                    label=f"mean over {win} steps")
        err_ax.legend(fontsize=8)
    err_ax.set(title="accumulated error", xlabel="timestep", ylabel="error (rad)")

    if steps:
        span = lim
        if typical.size:
            span = max(span, float(np.max(np.abs(typical)))) * 1.05
        step_ax = ax[1]
        step_ax.plot(
            [-span, span], [-span, span], "k--", lw=1, label="equal to the true step")
        if typical.size:
            step_ax.plot(
                ctr, typical, "o-", color="#b2182b", ms=5, lw=1.2,
                label="typical decoded step")
        gain_txt = f"{integration_gain:.2f}" if np.isfinite(integration_gain) else "—"
        step_ax.set(
            title=f"typical step is {gain_txt}× the true step",
            xlabel="true step (rad)", ylabel="decoded step (rad)",
            xlim=(-span, span), ylim=(-span, span))
        step_ax.set_box_aspect(1)
        step_ax.legend(fontsize=8, loc="lower right")
        fig.suptitle(
            f"{title} — median error {med:.3f} rad | tracking score {score:.4f} | "
            f"typical gain {gain_txt}",
            y=1.02)
    else:
        fig.suptitle(f"{title} — median accumulated error {med:.3f} rad | "
                     f"tracking score {score:.4f}")
    plt.tight_layout()
    return fig, ax



def plot_trajectory_vs_decoded(traj, decoded, title="Path integration per axis"):
    """
    One panel per torus axis: unwrapped phase against time, ground truth vs decoded.
 
    """
    #normalise both
    traj, decoded = np.asarray(traj, float), np.asarray(decoded, float)
    #unwrap both
    gt_u, dec_u = unwrap_torus(traj), unwrap_torus(decoded)
    #calculate the step difference
    gt_step = wrapped_angle_diff(traj[1:], traj[:-1])
    dec_step = wrapped_angle_diff(decoded[1:], decoded[:-1])
    #sample index
    t = np.arange(gt_u.shape[0])
    
    #plot the unwrapped phase against time, ground truth vs decoded
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for d, ax in enumerate(axes):
        ax.plot(t, gt_u[:, d], color="#2166ac", lw=1.2, alpha=0.85,
                label="ground truth")
        ax.plot(t, dec_u[:, d], color="#d6604d", lw=1.0, ls="--", alpha=0.85,
                label="decoded")
 
        
        ts, ds = gt_step[:, d], dec_step[:, d]
        integration_gain = float(ts @ ds / (ts @ ts + 1e-12))

        clean_axes(ax, title=f"{_AXIS_NAMES[d]}  (integration gain {integration_gain:.3f})",
                   ylabel="unwrapped phase (rad)")
        ax.set_xlabel("timestep")
        ax.legend(fontsize=8)
 
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig, axes
 
 
def plot_bump_tracking(states, decoded, n, title="Bump tracking check",
                       max_frames=600, cmap="inferno", input_stride=1):
    """
    Give a visualosation of the bump beeing tracked, one panel per axis.
    When a switch happen we should be able to see it here.
    """
    S = np.asarray(states)
    if S.ndim == 3:                               # precomputed (frames, d, n)
        margs, stride = S, 1
    else:                                         # raw (T, N), as now
        stride = max(1, S.shape[0] // int(max_frames))
        vol = S[::stride].reshape(-1, n, n, n)
        margs = np.stack([vol.sum(axis=(2, 3)), vol.sum(axis=(1, 3)),
                          vol.sum(axis=(1, 2))], axis=1)
    t = np.arange(margs.shape[0]) * stride * input_stride

    dec = np.asarray(decoded, float)
    if stride > 1:
        dec = dec[::stride]
    cells = (dec / (2 * np.pi) * n) % n

    fig, axes = plt.subplots(1, margs.shape[1], figsize=(5 * margs.shape[1], 4))
    if margs.shape[1] == 1:
        axes = [axes]
    names = [f"θ{i+1}" for i in range(margs.shape[1])]
    for d, ax in enumerate(axes):
        marg = margs[:, d]
        ax.imshow(marg.T, aspect="auto", origin="lower", cmap=cmap,
                  extent=[t[0], t[-1], 0, n])
 
        # break the line where it wraps round the torus, as in _break_periodic_jumps_2d
        y = cells[:, d].copy()
        y[np.where(np.abs(np.diff(y)) > n / 2)[0] + 1] = np.nan
        ax.plot(t, y, color="#00e5ff", lw=1.3, label="decoded")
 
        clean_axes(ax, title=names[d], ylabel="neuron index")
        ax.set_aspect("auto")     # clean_axes forces "equal", which flattens the panel
        ax.set_xlabel("timestep")
        ax.legend(fontsize=8, loc="upper right")
 
    fig.suptitle(title, y=1.02)
    plt.tight_layout()
    return fig, axes


def _snapshot_plane(d, plane=None):
    """(label, xy_axes, fixed_axis). For d=2 there is no fixed axis."""
    specs = slice_specs(d)
    if not specs:
        raise ValueError(f"need d>=2 to slice, got {d}")
    if plane is None:
        spec = specs[0]
    else:
        # plane is the fixed axis, matching the old 3-D API
        spec = next((s for s in specs if s[2] == plane), specs[0])
    d0, d1, d_fixed, x_label, y_label = spec
    if d_fixed is None:
        label = f"{x_label}–{y_label}"
    else:
        label = f"{x_label}–{y_label} at tracked θ{d_fixed+1}"
    return label, (d0, d1), d_fixed


def plot_bump_snapshots(volumes, times, cells=None, radius=None, cmap="inferno",
                        ncols=4, panel=3.2, show_tracker=True, plane=None):
    """Slice through each snapshot at the tracked cell.

    plane=2 is θ₁–θ₂ (default). plane=1 is θ₁–θ₃, plane=0 is θ₂–θ₃.
    Shows discrete neuron pixels (not a smoothed projection), the tracker
    centre (×), and optionally the tracking window (dashed square).
    ``plane`` is the fixed axis. For d=2 the whole sheet is shown.
    """
    vols = np.asarray(volumes)
    cells = None if cells is None else np.asarray(cells, float)
    k = len(vols)
    shape = vols.shape[1:]
    d = len(shape)
    label, xy_ax, fix_ax = _snapshot_plane(d, plane if d > 2 else None)
    ncols = min(ncols, k)
    nrows = int(np.ceil(k / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel * ncols, panel * nrows))
    axes = np.atleast_1d(axes).ravel()

    for i in range(len(axes)):
        ax = axes[i]
        if i >= k:
            ax.axis("off")
            continue
        sl = [slice(None)] * d
        if fix_ax is None:
            ifix = None
            if cells is not None:
                xy = (cells[i, xy_ax[0]], cells[i, xy_ax[1]])
            else:
                peak = np.unravel_index(np.argmax(vols[i]), vols[i].shape)
                xy = (peak[xy_ax[0]], peak[xy_ax[1]])
        else:
            n_fix = shape[fix_ax]
            if cells is not None:
                ifix = int(round(cells[i, fix_ax])) % n_fix
                xy = (cells[i, xy_ax[0]], cells[i, xy_ax[1]])
            else:
                peak = np.unravel_index(np.argmax(vols[i]), vols[i].shape)
                ifix = int(peak[fix_ax])
                xy = None
            sl[fix_ax] = ifix
        ax.imshow(vols[i][tuple(sl)].T, origin="lower", cmap=cmap,
                  extent=[0, shape[xy_ax[0]], 0, shape[xy_ax[1]]],
                  interpolation="nearest")
        if show_tracker and xy is not None:
            ax.plot(xy[0], xy[1], "x", color="#00e5ff", ms=11, mew=2)
            if radius is not None:
                ax.add_patch(plt.Rectangle(
                    (xy[0] - radius - 0.5, xy[1] - radius - 0.5),
                    2 * radius + 1, 2 * radius + 1,
                    fill=False, edgecolor="#00e5ff", lw=1.0, ls="--"))
        ax.set_title(f"t={times[i]}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(f"{label}, possibly with tracker window", y=1.02)
    plt.tight_layout()
    return fig, axes
