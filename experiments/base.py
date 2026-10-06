from dataclasses import dataclass, asdict
from copy import deepcopy

import numpy as np

from model.metrics import wrapped_angle_diff, torus_error_periods
from analysis.scoring import tracking_score
from model.path_integration import PathIntegrator, build_rotation_matrix
from config import world_to_flat_bins
from model.network.QAN3D import Torus3DQAN


@dataclass
class Trajectory:
    """Walk plus optional plane schedule. ``__iter__`` yields the legacy triple."""
    world_pos: np.ndarray
    v_body_seq: np.ndarray
    torus_gt: np.ndarray
    n_true_seq: np.ndarray = None
    segment_id: np.ndarray = None
    segment_starts: np.ndarray = None
    plane_id: np.ndarray = None
    route_of: np.ndarray = None
    flight_of: np.ndarray = None
    route_table: list = None
    flight_table: list = None

    def __iter__(self):
        yield self.world_pos
        yield self.v_body_seq
        yield self.torus_gt


def world_to_torus_gt(world_pos, scale):
    """Ground-truth torus coordinates for a world path, in radians.

    The metres-to-radians conversion is a plain scaling. PeriodicEuclidean lays
    the torus axes directly on the world axes, so there is no basis change to
    apply.

    The + pi centres the path in the middle of the torus rather than at the
    corner, so a walk starting at the world origin does not sit on the wrap.
    """
    return (np.pi + np.asarray(world_pos) * scale) % (2 * np.pi)


def torus_gt_from_velocity(v_body_seq, n_true_seq, scale, d):
    """Path-dependent GT: (π + scale · cumsum((R(n_t) v_t)[:d])) mod 2π.

    On a flat floor (n ≡ ẑ) this is bit-identical to world_to_torus_gt.
    """
    v_body_seq = np.asarray(v_body_seq, dtype=float)
    n_true_seq = np.asarray(n_true_seq, dtype=float)
    T = v_body_seq.shape[0]
    d = int(d)
    drive = np.empty((T, d))
    if T == 0:
        return drive
    # one R per run of constant n (one matrix per plane-switching segment)
    chg = np.empty(T, dtype=bool)
    chg[0] = True
    chg[1:] = np.any(n_true_seq[1:] != n_true_seq[:-1], axis=1)
    bounds = np.append(np.flatnonzero(chg), T)
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        R = build_rotation_matrix(n_true_seq[t0])
        drive[t0:t1] = (v_body_seq[t0:t1] @ R.T)[:, :d]
    return (np.pi + scale * np.cumsum(drive, axis=0)) % (2 * np.pi)


def bounce_square(uv, heading, speed, limit):
    """One in-plane step at `heading`, reflected so it stays in the ±limit square.

    The new position is the old position plus the (possibly reflected) step.
    No random turn: the caller updates the heading first.
    """
    heading = float(heading)
    duv = speed * np.array([np.cos(heading), np.sin(heading)])
    new_uv = np.asarray(uv, dtype=float) + duv
    for dim in range(2):
        if new_uv[dim] > limit or new_uv[dim] < -limit:
            heading = (np.pi - heading) if dim == 0 else (-heading)
            duv = speed * np.array([np.cos(heading), np.sin(heading)])
            new_uv = np.asarray(uv, dtype=float) + duv
    return new_uv, heading, duv


def plane_step(uv, heading, speed, step_turn_std, limit, rng):
    """One persistent-walk step inside the ±limit square."""
    heading = float(heading) + float(rng.normal(0.0, step_turn_std))
    return bounce_square(uv, heading, speed, limit)


def planar_walk(rng, n_steps, speed, step_turn_std, limit, uv0=None, heading=None):
    """Persistent random walk in plane coordinates. Reflection stays in-plane.

    step_turn_std is the standard deviation of the heading change per step.
    uv0 is the in-plane start (default origin). heading is the initial direction
    in that plane; omitted, it is drawn at random.

    Returns
    -------
    uv : (n_steps, 2) in-plane positions
    duv : (n_steps, 2) in-plane steps. duv[0] = 0, and uv[t] = uv[t-1] + duv[t].
    """
    uv = np.zeros((n_steps, 2))
    duv = np.zeros((n_steps, 2))
    if uv0 is not None:
        uv[0] = np.asarray(uv0, dtype=float)
    if heading is None:
        heading = float(rng.uniform(0, 2 * np.pi))
    for t in range(1, n_steps):
        uv[t], heading, duv[t] = plane_step(
            uv[t - 1], heading, speed, step_turn_std, limit, rng)
    return uv, duv


@dataclass
class WalkParams:
    cfg: object
    n_steps: int
    seed: int
    rng: object
    scale: float
    dt: float
    torus_inc: float
    world_speed: float
    omega_std: float
    step_turn_std: float
    limit: float


def walk_params(experiment_cfg, dt, turn_std=None, n_steps=None, seed=None,
                scale=None):
    """Shared walk numbers: speed, heading diffusion, box limit.

    The speed formula target_speed · dt / scale lives only here.
    """
    n_steps = experiment_cfg.n_steps if n_steps is None else int(n_steps)
    seed = experiment_cfg.seed if seed is None else int(seed)
    rng = np.random.default_rng(seed)
    scale = experiment_cfg.scale if scale is None else float(scale)
    dt = float(dt)
    torus_inc = experiment_cfg.target_speed_rad_per_time * dt   # rad/step
    world_speed = torus_inc / scale              # m/step
    omega_std = experiment_cfg.omega_std if turn_std is None else float(turn_std)
    step_turn_std = omega_std * np.sqrt(dt)
    # Reflect at boundaries ±(env_size/2)
    limit = experiment_cfg.env_size / 2
    return WalkParams(
        cfg=experiment_cfg, n_steps=n_steps, seed=seed, rng=rng,
        scale=scale, dt=dt, torus_inc=torus_inc, world_speed=world_speed,
        omega_std=omega_std, step_turn_std=step_turn_std, limit=limit,
    )


@dataclass
class ExperimentResult:
    world_pos:         np.ndarray
    v_body_seq:        np.ndarray
    torus_gt:          np.ndarray
    theta_hist:        np.ndarray
    n_hat_hist:        np.ndarray
    n_held_hist:       np.ndarray  # sample-and-hold normal that actually drives E
    gap_hist:          np.ndarray
    S_tot_buffer:      np.ndarray   # or None
    vmf_snapshots: list         # or None
    norm_error:        np.ndarray
    mean_norm_error:   float
    condition:         str
    params:            dict
    ratemap_sums:      np.ndarray = None
    ratemap_counts:    np.ndarray = None
    sub_idx: np.ndarray = None   # neuron subsample indices (3-D path)
    shuf_sums:         np.ndarray = None   # (n_shuffle, bins, ..., n_sub) shuffle maps
    active_mask:       np.ndarray = None   # (n_sub,) bool — neurons with non-trivial activity
    n_true_seq:        np.ndarray = None
    segment_id:        np.ndarray = None
    segment_starts:    np.ndarray = None
    plane_id:          np.ndarray = None
    route_of:          np.ndarray = None
    flight_of:         np.ndarray = None
    route_table:       list = None
    flight_table:      list = None
    kappa_hist:        np.ndarray = None  # vMF κ; gap_hist is an alias
    warmup_volume:     np.ndarray = None
    n_gravity_flips:   int = 0
    field_pos:         np.ndarray = None   # (n_samples, 3) metres, volumetric IFD
    field_act:         np.ndarray = None   # (n_samples, n_sub) activity trace
    field_stride:      int = 1


class BaseExperiment:
    condition_label = "base" #overwritten by subclasses
    ratemap_ndim          = 2
    ratemap_n_sub         = 0
    ratemap_n_shuffle     = 0
    ratemap_seed          = 0
    ratemap_active_thresh = 1e-3

    def __init__(self, config, record=None, plane_mode=None):
        self.qan = Torus3DQAN.from_config(config.network)
        N_neurons    = self.qan.cans[0].S.shape[0]
        decode_chunk = max(64, int(256e6 / (4 * N_neurons)))
        plane_kwargs = asdict(config.plane)
        plane_kwargs.pop("refresh_timing", None)
        self.integrator_kwargs = dict(
            **plane_kwargs,
            scale=config.experiment.scale,
            record_stride=config.experiment.record_stride,
            decode_chunk=decode_chunk,
        )
        if plane_mode is not None:
            self.integrator_kwargs["plane_mode"] = plane_mode  # "bayesian" (default) or "true"
        self.config        = config
        e = config.experiment
        self.record        = e.record if record is None else record
        self.record_stride = e.record_stride
        self.ratemap_ndim          = e.ratemap_ndim
        self.ratemap_n_sub         = e.ratemap_n_sub
        self.ratemap_n_shuffle     = e.ratemap_n_shuffle
        self.ratemap_seed          = e.ratemap_seed
        self.ratemap_active_thresh = e.ratemap_active_thresh

        print(
            f"[{self.condition_label}] "
            f"neurons {self.qan.cans[0].S.shape[0]:,} "
            f"arena {e.env_size} m | grid_period {e.grid_spacing} m | "
            f"periods {e.env_size / e.grid_spacing:.2f}"
        )

    def generate_trajectory(self, turn_std: float = None, n_steps=None, seed=None):
        """Required hook: return a Trajectory (or a 3-tuple via Trajectory.__iter__).

        n_steps and seed default to config.experiment when omitted.
        """
        raise NotImplementedError("Should be implemented by subclass")

    def _walk_params(self, turn_std: float = None, n_steps=None, seed=None):
        return walk_params(
            self.config.experiment, self.config.network.dt,
            turn_std=turn_std, n_steps=n_steps, seed=seed,
        )

    def _settle_steps(self):
        s = getattr(self.config.network, "settle_steps", None)
        if s is not None:
            return int(s)
        tau_n = float(self.qan.cans[0].tau)
        dt = float(self.config.network.dt)
        return max(int(np.ceil(10.0 / ((dt / tau_n) * 0.2))), 100)

    def run(self, world_pos, v_body_seq, torus_gt, g_vec,
            n_true_seq=None, refresh_mask=None,
            segment_starts=None, plane_mode=None) -> ExperimentResult:
        bins     = self.config.experiment.ratemap_bins
        env_size = self.config.experiment.env_size
        ndim     = self.ratemap_ndim
        T        = len(world_pos)

        flat = world_to_flat_bins(world_pos, env_size, bins, ndim=ndim)

        kw = dict(self.integrator_kwargs)
        if plane_mode is not None:
            kw["plane_mode"] = plane_mode
        integrator = PathIntegrator(qan=self.qan, **kw)
        integrator.reset(torus_gt[0])
        settle = self._settle_steps()
        integrator.warmup(n_steps=settle)
        warmup_volume = integrator.backend.current_volume()

        if refresh_mask is None:
            refresh_mask = integrator.default_refresh_mask(
                T, timing=self.config.plane.refresh_timing,
                segment_starts=segment_starts,
            )

        # neuron subsample + shuffle lags (one RNG, sequential draws)
        N   = integrator.backend.S.shape[1]
        rng = np.random.default_rng(self.ratemap_seed)

        # ratemap_n_sub <= 0 keeps every cell. The shuffle buffer uses this
        # same column order; score_2d_from_map indexes both with one scored_idx.
        sub_idx = None
        if self.ratemap_n_sub > 0:
            sub_idx = np.sort(rng.choice(N, size=min(self.ratemap_n_sub, N),
                                         replace=False))

        lags = None
        n_shuffle = self.ratemap_n_shuffle
        if n_shuffle > 0:
            min_lag = max(1, int(T * 0.1))
            max_lag = max(min_lag + 1, int(T * 0.9))
            if min_lag < max_lag:
                lags = rng.integers(min_lag, max_lag, size=n_shuffle)
            else:
                n_shuffle = 0          # trajectory too short for shuffles

        analysis = getattr(self.config, "analysis", None)
        field_stride = getattr(analysis, "field_stride", 10)
        max_trace_bytes = getattr(analysis, "max_trace_bytes", 1024 ** 3)
        theta_hist = integrator.run(
            v_body_seq, g_vec,
            record=self.record,
            flat_indices=flat,
            ratemap_bins=bins,
            ratemap_ndim=ndim,
            sub_idx=sub_idx,
            n_shuffle=n_shuffle if lags is not None else 0,
            lags=lags,
            n_true_seq=n_true_seq,
            refresh_mask=refresh_mask,
            segment_starts=segment_starts,
            world_pos=world_pos if ndim == 3 else None,
            field_stride=field_stride,
            max_trace_bytes=max_trace_bytes,
        )

        if "kappa" in integrator.history and len(integrator.history["kappa"]):
            kappa = np.array(integrator.history["kappa"])
        else:
            kappa = np.array(integrator.history["z2"]) - np.array(integrator.history["z1"])
        n_hat_hist = np.array(integrator.history["n_hat"])
        n_held_hist = np.array(integrator.history["n_held"])
        n_sheet = int(integrator.backend.n)
        bump_sp = float(self.config.network.bump_spacing_cells)
        torus_err = torus_error_periods(
            theta_hist, torus_gt, n_sheet, bump_sp)
        score = tracking_score(theta_hist, torus_gt)

        active_mask = None
        if sub_idx is not None and integrator.ratemap_sums is not None and integrator.ratemap_counts is not None:
            sums_flat   = integrator.ratemap_sums.reshape(-1, integrator.ratemap_sums.shape[-1])  # (total_bins, n_sub)
            counts_flat = integrator.ratemap_counts.reshape(-1)                                    # (total_bins,)
            valid       = counts_flat > 0                                                          # unvisited bins → nan
            rate_flat   = np.full_like(sums_flat, np.nan)
            rate_flat[valid] = sums_flat[valid] / counts_flat[valid, np.newaxis]
            span        = np.nanmax(rate_flat, axis=0) - np.nanmin(rate_flat, axis=0)
            active_mask = span > self.ratemap_active_thresh * (np.nanmax(span) + 1e-12)

        self.last_integrator = integrator
        params = deepcopy(asdict(self.config))
        if plane_mode is not None:
            params["plane"]["plane_mode"] = plane_mode
        arm = plane_mode if plane_mode is not None else self.integrator_kwargs.get(
            "plane_mode", "")
        condition = self.condition_label if not arm else f"{self.condition_label}_{arm}"

        return ExperimentResult(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            ratemap_sums=integrator.ratemap_sums,
            ratemap_counts=integrator.ratemap_counts,
            torus_gt=torus_gt,
            theta_hist=theta_hist,
            n_hat_hist=n_hat_hist,
            n_held_hist=n_held_hist,
            gap_hist=kappa,
            S_tot_buffer=integrator.S_tot_buffer,
            vmf_snapshots=integrator.vmf_snapshots,
            norm_error=torus_err,
            mean_norm_error=float(score),
            condition=condition,
            params=params,
            sub_idx=sub_idx,
            shuf_sums=integrator.ratemap_shuf_sums,
            active_mask=active_mask,
            n_true_seq=n_true_seq,
            segment_starts=(
                None if segment_starts is None
                else np.asarray(segment_starts, dtype=int)),
            kappa_hist=kappa,
            warmup_volume=warmup_volume,
            n_gravity_flips=int(getattr(integrator, "n_gravity_flips", 0)),
            field_pos=getattr(integrator, "field_pos", None),
            field_act=getattr(integrator, "field_act", None),
            field_stride=int(getattr(integrator, "field_stride", None) or 1),
        )

    @staticmethod
    def _made_metric(decoded, ground_truth, world_pos=None):
        if world_pos is not None:
            steps = np.linalg.norm(np.diff(world_pos, axis=0), axis=1)  # metres
        else:
            steps = np.linalg.norm(
                wrapped_angle_diff(ground_truth[1:], ground_truth[:-1]), axis=1
            )
        traj_length = np.cumsum(steps)
        err = np.linalg.norm(wrapped_angle_diff(decoded, ground_truth), axis=1)
        norm_error = err[1:] / (traj_length + 1e-9)
        return np.concatenate([[0.0], norm_error])

    def run_experiment(self, g):
        traj = self.generate_trajectory()
        world_pos, v_body_seq, torus_gt = traj
        n_true = getattr(traj, "n_true_seq", None)
        starts = getattr(traj, "segment_starts", None)
        result = self.run(
            world_pos, v_body_seq, torus_gt, g,
            n_true_seq=n_true, segment_starts=starts,
        )
        if not result.condition:
            result.condition = self.condition_label
        if not result.params:
            result.params = asdict(self.config)
        if isinstance(traj, Trajectory):
            result.n_true_seq = traj.n_true_seq
            result.segment_id = traj.segment_id
            result.segment_starts = traj.segment_starts
            result.plane_id = traj.plane_id
            result.route_of = getattr(traj, "route_of", None)
            result.flight_of = getattr(traj, "flight_of", None)
            result.route_table = getattr(traj, "route_table", None)
            result.flight_table = getattr(traj, "flight_table", None)
        return result
