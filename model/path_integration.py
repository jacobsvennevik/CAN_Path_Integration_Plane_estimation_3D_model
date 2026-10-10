import warnings

import numpy as np
import torch

from model.plane_estimation import (
    predict,
    update,
    uniform_prior,
    sample_vmf,
)
from model.network.torch_backend import TorchBackend

Z_HAT = np.array([0.0, 0.0, 1.0])


def build_rotation_matrix(n_hat: np.ndarray) -> np.ndarray:
    """
    Returns R such that R @ n_hat = z_hat, via the closed-form
    vector-to-vector rotation (Rodrigues).
    """
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    z_hat = Z_HAT

    v = np.cross(n_hat, z_hat)   # rotation axis * sin(theta)
    c = float(np.dot(n_hat, z_hat))  # cos(theta)

    # already aligned -> identity (this is the common case on a flat floor)
    if c > 1.0 - 1e-10:
        return np.eye(3)
    # anti-aligned (n_hat points straight down) -> 180° flip
    if c < -1.0 + 1e-10:
        return np.diag([1.0, -1.0, -1.0])

    vx = np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0 ],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def frame(n_hat: np.ndarray) -> np.ndarray:
    """In-plane basis E (3, 2) for n_hat: E.T @ n = 0, E.T @ E = I."""
    return build_rotation_matrix(n_hat)[:2, :].T


def pi_star(v_alloc: np.ndarray, dim: int = None) -> np.ndarray:
    """Slice of an allocentric velocity.

    After R maps n̂ to ẑ the torus axes are the leading coordinates, so this
    returns the first `dim` components. The whole vector when dim is None.
    """
    v = np.asarray(v_alloc, dtype=float)
    return v if dim is None else v[:int(dim)]


KAPPA_W_DEFAULT = 1e4


def trace_nbytes(n_steps: int, n_cells: int, stride: int) -> int:
    """Bytes for a strided (position, activity) trace. Positions are 3 float32."""
    stride = int(stride)
    if stride < 1:
        raise ValueError(f"field_stride must be at least 1, got {stride}")
    n_steps = int(n_steps)
    n_samples = (n_steps + stride - 1) // stride
    bytes_per = 4 * int(max(int(n_cells), 1)) + 12
    return int(n_samples) * int(bytes_per)


def require_trace_budget(n_steps: int, n_cells: int, stride: int,
                         max_bytes: int) -> int:
    """Raise if the explicit stride would allocate more than `max_bytes`.

    The stride is not increased to squeeze the trace under the cap.
    """
    nbytes = trace_nbytes(n_steps, n_cells, stride)
    if nbytes > int(max_bytes):
        raise MemoryError(
            f"field trace is {nbytes / 1024**3:.2f} GB at stride {int(stride)} "
            f"({int(n_cells)} cells, {int(n_steps)} steps), above the "
            f"{int(max_bytes) / 1024**3:.2f} GB cap. The stride is not increased."
        )
    return nbytes


def step_filter(current, displacement, kappa, kappa_w=KAPPA_W_DEFAULT):
    """Single predict and update cycle."""
    predicted = predict(current, kappa_w)
    return update(predicted, displacement, kappa)


class PathIntegrator:
    """
    Path integrator coupling the vMF plane filter with the T^d QAN.
    """
    def __init__(self, qan, kappa_v=None, scale=1.0,
                 plane_mode="bayesian",
                 decode_chunk=4096, decode_radius=None, decode_seed_radius=None,
                 kappa_w=KAPPA_W_DEFAULT, kappa_sens=300.0,
                 tau_refresh=1, seed=0,
                 refresh_settle_steps=200, reset_prior_at_segment_start=False,
                 device=None):
        self.qan = qan
        self.kappa_v = None if kappa_v is None else float(kappa_v)
        self.kappa_w = float(kappa_w)  # process-noise concentration for predict()
        self.kappa_sens = float(kappa_sens)
        self.tau_refresh = int(tau_refresh)
        self.refresh_settle_steps = int(refresh_settle_steps)
        self.reset_prior_at_segment_start = bool(reset_prior_at_segment_start)
        self.scale = scale
        self.backend = TorchBackend(qan, device=device)
        self._vmf_state = uniform_prior() #starting belief for n̂
        self._n_held = None
        self._R = None
        self.n_gravity_flips = 0
        self._rng = np.random.default_rng(int(seed))
        self.plane_mode = plane_mode #whether to use the true plane mode or the vMF mode
        self._true_n_hat = Z_HAT.copy()      # gravity / flat floor
        self.decode_chunk = int(decode_chunk) #how many steps to buffer on-device before decoding in one go
        self.decode_radius = decode_radius
        self.decode_seed_radius = decode_seed_radius
        self.history = {
            "n_hat": np.zeros((0, 3)),
            "n_held": np.zeros((0, 3)),
            "kappa": np.zeros((0,)),
        }
        self.ratemap_sums = None
        self.ratemap_counts = None

    def _likelihood(self) -> float:
        return self.kappa_sens if self.kappa_v is None else self.kappa_v

    def warmup(self, n_steps: int = 100):
        """Let the lattice settle, then lock the tracker on.

        The tracker is re-seeded at the end rather than followed through warmup:
        while the pattern is still forming, the bump can move further in one step
        than the tracking window is wide, which loses it. Nothing has moved during
        warmup, so the position afterwards is still the seed position.
        """
        self.backend.form_lattice(self._theta_0, settle=n_steps, min_peakedness=0.0)
        self._seed_tracker()

    def _seed_tracker(self) -> np.ndarray:
        return self.backend.seed_tracker(
            self._theta_0, radius=self.decode_radius,
            seed_radius=self.decode_seed_radius)

    def _update_filter(self, v_body, n_ref):
        """Sensed measurement. True-plane runs do not call this."""
        d_norm = np.linalg.norm(v_body)
        if d_norm <= 1e-9:
            return
        z = sample_vmf(n_ref, self.kappa_sens, self._rng)
        self._vmf_state = step_filter(
            self._vmf_state, z, self._likelihood(), self.kappa_w)

    def default_refresh_mask(self, n_steps, timing="free", segment_starts=None):
        """Refresh schedule.

        ``aligned`` copies at each segment start, again ``refresh_settle_steps``
        later when that step is still inside the segment, then every
        ``tau_refresh`` steps after the settle copy until the next start.
        A segment shorter than the settle delay keeps only the start copy.
        Free-running ``t % tau == 0`` is unchanged. ``tau = 1`` is continuous
        projection and is not this schedule.
        """
        n_steps = int(n_steps)
        tau = max(int(self.tau_refresh), 1)
        if timing == "aligned":
            if segment_starts is None:
                warnings.warn(
                    "refresh_timing='aligned' needs segment_starts; "
                    "falling back to free-running mask",
                    UserWarning, stacklevel=2,
                )
            else:
                mask = np.zeros(n_steps, dtype=bool)
                starts = np.unique(np.asarray(segment_starts, dtype=int))
                starts = starts[(starts >= 0) & (starts < n_steps)]
                settle = max(int(self.refresh_settle_steps), 0)
                bounds = list(starts) + [n_steps]
                for a, b in zip(bounds[:-1], bounds[1:]):
                    mask[a] = True
                    t = a + settle
                    if t < b and t != a:
                        mask[t] = True
                    elif t >= b:
                        continue
                    t = a + settle + tau
                    while t < b:
                        mask[t] = True
                        t += tau
                return mask
        return np.arange(n_steps) % tau == 0

    def _advance(self, v_body, g_hat, n_true=None, refresh=True, reset_prior=False):
        """One step of the filter and the network. No decode, no history."""
        v_body = np.asarray(v_body, dtype=float)
        n_ref = n_true if n_true is not None else self._true_n_hat
        n_ref = np.asarray(n_ref, dtype=float)
        n_ref = n_ref / np.linalg.norm(n_ref)

        #plane mode either bayesian or true plane mode
        if self.plane_mode != "true":
            if reset_prior:
                self._vmf_state = uniform_prior()
            self._update_filter(v_body, n_ref)
            n_hat = np.asarray(self._vmf_state.mu, dtype=float).copy()
            if np.dot(n_hat, g_hat) > 0:
                n_hat = -n_hat
                self.n_gravity_flips += 1
        else:
            n_hat = n_ref

        #build rotation matrix and rotate velocity. R is kept until the held normal changes.
        if refresh or self._n_held is None:
            self._n_held = n_hat.copy()
            self._R = build_rotation_matrix(self._n_held)
        v_alloc = self._R @ v_body #allocentric velocity

        target_speed_rad = v_alloc * self.scale  # rad/step
        # backend.step expects rad per unit TIME (theta_dot_at's convention)
        # and one component per torus axis (π star after R maps n̂ -> ẑ)
        drive = pi_star(target_speed_rad, self.qan.manifold.dim) / self.qan.dt
        self.backend.step(drive)
        return n_hat, v_alloc, target_speed_rad

    def run(self, v_body_sequence: np.ndarray, g: np.ndarray,
            flat_indices: np.ndarray = None, ratemap_bins: int = 0,
            ratemap_ndim: int = 2, sub_idx: np.ndarray = None,
            n_shuffle: int = 0, lags: np.ndarray = None,
            null_idx: np.ndarray = None,
            n_true_seq: np.ndarray = None,
            refresh_mask: np.ndarray = None,
            segment_starts: np.ndarray = None,
            world_pos: np.ndarray = None,
            field_stride: int = None,
            max_trace_bytes: int = None) -> np.ndarray:
        """
        Run the full pipeline over a pre-computed velocity sequence.

        Decode is batched. History is written into arrays. The sheet mean is
        computed once per step and reused for the rate map and the shifts.
        """
        T = v_body_sequence.shape[0] #total timesteps
        torus_dim = self.qan.manifold.dim
        dev = self.backend.device
        N = self.backend.S.shape[1]
        theta_history = np.zeros((T, torus_dim)) #place to store decoded positions

        if refresh_mask is None:
            refresh_mask = self.default_refresh_mask(T)
        refresh_mask = np.asarray(refresh_mask, dtype=bool)
        reset_at = set()
        if self.reset_prior_at_segment_start and segment_starts is not None:
            reset_at = {int(s) for s in np.asarray(segment_starts).ravel()}
        if n_true_seq is None:
            n_true_seq = np.broadcast_to(self._true_n_hat, (T, 3))
        n_true_seq = np.asarray(n_true_seq, dtype=float)

        # history goes into flat arrays now, not growing lists
        _h_n_hat = np.empty((T, 3), dtype=np.float64)
        _h_n_held = np.empty((T, 3), dtype=np.float64)
        _h_kappa = np.empty(T, dtype=np.float64)

        # small on-device buffer, decode + dump to CPU once it fills (keeps memory bounded)
        chunk = min(self.decode_chunk, T)
        _S_chunk = torch.empty((chunk, N), dtype=torch.float32, device=dev)

        # neuron subsample → torch index tensor
        sub_t = (torch.tensor(sub_idx, dtype=torch.long, device=self.backend.device)
                 if sub_idx is not None else None)
        n_neurons = len(sub_idx) if sub_idx is not None else self.backend.S.shape[1]
        if null_idx is not None:
            null_t = torch.tensor(null_idx, dtype=torch.long, device=self.backend.device)
            n_null = int(null_t.numel())
        else:
            null_t = sub_t
            n_null = n_neurons

        # rate-map accumulator (2-D or 3-D)
        _acc = None
        if flat_indices is not None and ratemap_bins > 0:
            total_bins = ratemap_bins ** ratemap_ndim
            _acc = self.backend.allocate_ratemap(total_bins, sub_t)

        _shuf = None
        if n_shuffle > 0 and lags is not None and _acc is not None:
            _shuf = self.backend.allocate_shuffle_ratemap(total_bins, n_null, n_shuffle)

        # Strided (position, activity) trace for volumetric inter-field distance.
        # Only the scored subsample: the full sheet does not fit.
        self.field_pos = None
        self.field_act = None
        self.field_stride = None
        _field_pos = _field_act = None
        _field_write = 0
        _field_stride = 1
        if (world_pos is not None and sub_t is not None and ratemap_ndim == 3
                and _acc is not None):
            n_field = int(sub_t.numel())
            if field_stride is None or max_trace_bytes is None:
                raise ValueError("a field trace needs field_stride and max_trace_bytes")
            _field_stride = int(field_stride)
            require_trace_budget(T, n_field, _field_stride, max_trace_bytes)
            n_field_samp = (T + _field_stride - 1) // _field_stride
            _field_pos = np.empty((n_field_samp, 3), dtype=np.float32)
            _field_act = np.empty((n_field_samp, n_field), dtype=np.float32)

        # g is fixed for the whole run, so normalise it once not every step
        g = np.asarray(g, dtype=float)
        g_hat = g / np.linalg.norm(g)

        chunk_start = 0 #first step currently held in _S_chunk

        for t in range(T):
            v_body = np.asarray(v_body_sequence[t], dtype=float)
            n_hat, _, _ = self._advance(
                v_body, g_hat, n_true=n_true_seq[t], refresh=bool(refresh_mask[t]),
                reset_prior=(t in reset_at))
            activity = self.backend.sheet_mean()
            # stash the bump state on-device, no .cpu() here
            _S_chunk[t - chunk_start] = activity

            # write history straight into the arrays
            _h_n_hat[t] = n_hat
            _h_n_held[t] = self._n_held
            _h_kappa[t] = float(self._vmf_state.kappa)

            if _acc is not None:
                self.backend.record_ratemap(_acc, int(flat_indices[t]), sub_t, activity)
            if _shuf is not None:
                self.backend.record_shuffle_ratemap(
                    _shuf, flat_indices, t, lags, null_t, activity)

            filled = t - chunk_start + 1
            # chunk full (or last step) -> decode it all at once and dump to CPU.
            # The tracker keeps its state between calls, so chunks join up.
            if filled == chunk or t == T - 1:
                start = chunk_start
                theta_history[start:t + 1] = self.backend.track_batch(_S_chunk[:filled])
                if _field_act is not None:
                    times = np.arange(start, t + 1)
                    keep = (times % _field_stride) == 0
                    if np.any(keep):
                        rows = torch.as_tensor(
                            np.flatnonzero(keep), dtype=torch.long, device=dev)
                        act = _S_chunk.index_select(0, rows).index_select(1, sub_t)
                        n_keep = int(keep.sum())
                        _field_act[_field_write:_field_write + n_keep] = (
                            act.detach().cpu().numpy())
                        _field_pos[_field_write:_field_write + n_keep] = (
                            world_pos[times[keep]])
                        _field_write += n_keep
                chunk_start = t + 1

        self.history["n_hat"] = _h_n_hat
        self.history["n_held"] = _h_n_held
        self.history["kappa"] = _h_kappa

        if _acc is not None:
            self.ratemap_sums, self.ratemap_counts = \
                self.backend.ratemap_to_numpy(_acc, ratemap_bins, ndim=ratemap_ndim)
        else:
            self.ratemap_sums = self.ratemap_counts = None

        if _shuf is not None:
            shape = (n_shuffle,) + (ratemap_bins,) * ratemap_ndim + (-1,)
            self.ratemap_shuf_sums = _shuf.cpu().numpy().reshape(shape)
        else:
            self.ratemap_shuf_sums = None

        if _field_act is not None:
            self.field_pos = _field_pos[:_field_write]
            self.field_act = _field_act[:_field_write]
            self.field_stride = _field_stride

        return theta_history

    def reset(self, theta_0: np.ndarray):
        """Reset filter and CAN states without rebuilding the network."""
        self._vmf_state = uniform_prior()
        self.backend.reset(theta_0)
        self._theta_0 = np.asarray(theta_0, dtype=np.float64).copy()
        self._seed_tracker()
        self._n_held = None
        self._R = None
        self.n_gravity_flips = 0
        self.history = {
            "n_hat": np.zeros((0, 3)),
            "n_held": np.zeros((0, 3)),
            "kappa": np.zeros((0,)),
        }
