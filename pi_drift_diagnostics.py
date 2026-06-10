import numpy as np
import matplotlib.pyplot as plt
from metrics import wrapped_angle_diff
from path_integration import PathIntegrator
# wrapped_angle_diff, np, plt already imported in the notebook


def zero_velocity_diffusion(exp, g_vec, n_steps=150_000, start=None, warmup_steps=500):
    """
    B&F Fig 6C: hold velocity at zero, measure how fast the bump wanders -> D.

    Changes from previous version
    --------------------------------
    1. warmup_steps is now a parameter (default 500, not 100) so the bump is
       fully settled on the attractor before drift is measured.
    2. MSD fit uses only the first 25 % of lags (B&F's implicit approach),
       NOT a (pi/2)^2 saturation threshold.  The unwrapped drift is unbounded
       and does not saturate; the old threshold was conceptually wrong.
    3. D is now decomposed per-axis (D1, D2, D3).  The joint MSD gives
       D_total = D1+D2+D3, but decoherence is triggered by the first axis
       to slip by pi.  Two t_decohere values are reported:
         - isotropic:    pi^2 / (D_total / 3)   -- assumes D1=D2=D3
         - conservative: pi^2 / max(D1,D2,D3)   -- worst-case axis
    4. Single-step displacement sanity check added before unwrapping.
    """
    pi = PathIntegrator(qan=exp.qan, **exp.integrator_kwargs)
    if start is None:
        start = np.array([np.pi, np.pi, np.pi])
    pi.reset(start)
    pi.warmup(warmup_steps)

    pos = np.empty((n_steps, 3))
    zero_v = np.zeros(3)
    for t in range(n_steps):
        pos[t] = pi.step(zero_v, g_vec)

    # Sanity check: individual steps must be < pi/2 for the unwrapping to be reliable.
    max_step = np.abs(wrapped_angle_diff(pos[1:], pos[:-1])).max()
    if max_step > np.pi * 0.5:
        print(f"WARNING: max single-step displacement = {max_step:.3f} rad  "
              "(> pi/2).  Unwrapping may be unreliable at this noise level.")

    # Unwrap by accumulating small signed steps — stays in R^3, not on the torus.
    drift = np.zeros_like(pos)
    for t in range(1, n_steps):
        drift[t] = drift[t - 1] + wrapped_angle_diff(pos[t], pos[t - 1])

    # ------------------------------------------------------------------ #
    #  Joint MSD (all three axes combined) — used for D_total             #
    # ------------------------------------------------------------------ #
    lags = np.unique(np.geomspace(1, n_steps // 4, 40).astype(int))
    msd_joint = np.array(
        [np.mean(np.sum((drift[L:] - drift[:-L]) ** 2, axis=1)) for L in lags]
    )

    # ------------------------------------------------------------------ #
    #  Per-axis MSD — used for D_per_axis and conservative t_decohere     #
    # ------------------------------------------------------------------ #
    msd_axes = np.array(
        [
            [np.mean((drift[L:, ax] - drift[:-L, ax]) ** 2) for L in lags]
            for ax in range(3)
        ]
    )  # shape (3, n_lags)

    # FIT: only the first 25 % of lags (reliable, pre-variance-inflation regime).
    # B&F Fig 6C fits implicitly restrict to this window.
    n_fit = max(3, len(lags) // 4)
    fit_lags = lags[:n_fit]

    if msd_joint.max() < 1e-12:
        print(
            "D ~ 0 : no measurable drift (backend appears noiseless).  "
            "This is the correct B&F result without spiking noise."
        )
        D_total = 0.0
        D_per_axis = np.zeros(3)
    else:
        D_total = float(
            max(np.polyfit(fit_lags, msd_joint[:n_fit], 1)[0], 0.0)
        )
        D_per_axis = np.array(
            [
                float(max(np.polyfit(fit_lags, msd_axes[ax, :n_fit], 1)[0], 0.0))
                for ax in range(3)
            ]
        )

    # ------------------------------------------------------------------ #
    #  Decoherence times (two estimates)                                  #
    # ------------------------------------------------------------------ #
    # Isotropic: assumes D1=D2=D3=D_total/3; standard B&F-style estimate.
    D_iso = D_total / 3.0
    t_decohere_iso = np.pi ** 2 / D_iso if D_iso > 0 else np.inf

    # Conservative: worst-case (fastest-drifting) axis.
    D_max_axis = D_per_axis.max()
    t_decohere_conservative = np.pi ** 2 / D_max_axis if D_max_axis > 0 else np.inf

    # ------------------------------------------------------------------ #
    #  Plots                                                              #
    # ------------------------------------------------------------------ #
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    # Panel 0: joint MSD log-log with fit region highlighted
    axes[0].loglog(
        lags, np.maximum(msd_joint, 1e-30),
        "o", ms=4, color="#2166ac", label="MSD (joint)",
    )
    axes[0].loglog(
        lags[:n_fit], np.maximum(msd_joint[:n_fit], 1e-30),
        "o", ms=5, color="#d6604d", label="fit region",
    )
    if D_total > 0:
        axes[0].loglog(
            fit_lags, D_total * fit_lags,
            "r--", lw=1.5, label=f"fit D_total={D_total:.2e}",
        )
    axes[0].set(
        title="Joint MSD (slope~1 = diffusive)",
        xlabel="lag (steps)", ylabel="rad²",
    )
    axes[0].legend(fontsize=8)

    # Panel 1: per-axis MSD log-log
    colors_ax = ["#1b7837", "#762a83", "#e08214"]
    for ax_i, (c, label) in enumerate(zip(colors_ax, ["θ₁", "θ₂", "θ₃"])):
        axes[1].loglog(
            lags, np.maximum(msd_axes[ax_i], 1e-30),
            "o", ms=3, color=c, alpha=0.8, label=f"{label} D={D_per_axis[ax_i]:.2e}",
        )
    axes[1].set(
        title="Per-axis MSD",
        xlabel="lag (steps)", ylabel="rad²",
    )
    axes[1].legend(fontsize=7)

    # Panel 2: |drift| over time
    stride = max(1, n_steps // 5000)
    t_ax = np.arange(n_steps)
    axes[2].plot(
        t_ax[::stride], np.linalg.norm(drift, axis=1)[::stride],
        color="#d6604d", lw=0.7,
    )
    axes[2].axhline(np.pi, color="k", ls=":", label="half-period (π)")
    axes[2].set(title="|drift| over time", xlabel="step", ylabel="rad")
    axes[2].legend(fontsize=8)

    fig.suptitle(
        f"Zero-velocity diffusion (B&F Fig 6C)\n"
        f"D_total={D_total:.2e}  |  "
        f"t_dec (isotropic)={t_decohere_iso:.0f} steps  |  "
        f"t_dec (conservative)={t_decohere_conservative:.0f} steps",
        y=1.02,
    )
    plt.tight_layout()
    plt.show()

    print(
        f"D_total      = {D_total:.3e} rad²/step\n"
        f"D_per_axis   = [{', '.join(f'{d:.3e}' for d in D_per_axis)}] rad²/step\n"
        f"t_decohere (isotropic)    = {t_decohere_iso:.0f} steps\n"
        f"t_decohere (conservative) = {t_decohere_conservative:.0f} steps"
    )
    return D_total, D_per_axis, t_decohere_iso, t_decohere_conservative


def path_normalised_error(
    exp, g_vec, n_trials=50, n_steps=3000, base_seed=1000, warmup_steps=500
):
    """
    Claudi Fig 6C: 50 random trajectories, error as % of path length.

    Changes from previous version
    --------------------------------
    1. warmup_steps is now a parameter (default 500, matching zero_velocity_diffusion)
       so both functions use a consistent settling criterion.
    2. Error and path-length arrays are aligned correctly: both use indices [1:]
       (shape T-1).  Index 0 is skipped because decoded[0] and torus_gt[0]
       represent the same moment (before any integration step) and comparing
       them measures reset residual, not integration error.
    3. Box-plot white line is now the MEAN (matching Claudi Fig 6C caption:
       "white lines the mean"), not the median.  Median is shown in dark.
    4. Jitter uses a locally-seeded RNG so dot positions are reproducible
       across notebook re-runs with identical data.
    """
    pi = PathIntegrator(qan=exp.qan, **exp.integrator_kwargs)
    errors_pct = np.empty(n_trials)

    saved_seed = exp.config.experiment.seed
    saved_T    = exp.config.experiment.n_steps
    exp.config.experiment.n_steps = n_steps

    for trial in range(n_trials):
        exp.config.experiment.seed = base_seed + trial
        world_pos, v_body_seq, torus_gt = exp.generate_trajectory()

        pi.reset(torus_gt[0])
        pi.warmup(warmup_steps)
        decoded = pi.run(v_body_seq, g_vec, record=False)   # (n_steps, 3)

        # Skip index 0: decoded[0] and torus_gt[0] are the same moment
        # (before any velocity has been integrated). Including them measures
        # reset residual, not path-integration error.
        # Both err and step now have shape (T-1,) — correctly aligned.
        err  = np.linalg.norm(
            wrapped_angle_diff(decoded[1:], torus_gt[1:]), axis=1
        )   # (T-1,)
        step = np.linalg.norm(
            wrapped_angle_diff(torus_gt[1:], torus_gt[:-1]), axis=1
        )   # (T-1,)
        errors_pct[trial] = 100.0 * err.sum() / (step.sum() + 1e-9)

        if (trial + 1) % 10 == 0:
            print(
                f"  trial {trial+1}/{n_trials}  "
                f"running mean = {errors_pct[:trial+1].mean():.2f}%"
            )

    exp.config.experiment.seed = saved_seed
    exp.config.experiment.n_steps = saved_T

    mean_val   = float(errors_pct.mean())
    median_val = float(np.median(errors_pct))

    fig, ax = plt.subplots(figsize=(4, 6))

    ax.boxplot(
        errors_pct,
        positions=[1],
        widths=0.4,
        patch_artist=True,
        # Median shown in dark (not white) — white line is reserved for mean
        medianprops=dict(color="#1a1a1a", lw=1.5),
        boxprops=dict(facecolor="#4393c3", alpha=0.8),
        whiskerprops=dict(lw=1.2),
        capprops=dict(lw=1.2),
        flierprops=dict(marker="o", ms=4, color="#2166ac", alpha=0.6),
    )

    # White line = mean, matching Claudi Fig 6C
    ax.hlines(
        mean_val, 0.8, 1.2,
        colors="white", lw=2.0, label=f"mean = {mean_val:.1f}%", zorder=4,
    )

    # Jitter: locally-seeded RNG for reproducible dot positions
    rng = np.random.default_rng(seed=0)
    ax.scatter(
        1 + rng.uniform(-0.08, 0.08, n_trials),
        errors_pct,
        s=18, alpha=0.5, color="#2166ac", zorder=3,
    )

    ax.set_xticks([1])
    ax.set_xticklabels(["T³ arena"])
    ax.set_ylabel("error (% of path length)")
    ax.set_title(
        f"Path-normalised error (Claudi Fig 6C)\n"
        f"n={n_trials}  mean={mean_val:.2f}%  median={median_val:.2f}%"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()

    print(
        f"mean   = {mean_val:.2f}%\n"
        f"median = {median_val:.2f}%\n"
        f"IQR    = [{np.percentile(errors_pct, 25):.2f}%, "
        f"{np.percentile(errors_pct, 75):.2f}%]"
    )
    return errors_pct