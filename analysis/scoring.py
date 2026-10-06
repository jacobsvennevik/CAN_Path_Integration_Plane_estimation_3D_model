"""
Grid-field scoring. Rate maps are accumulated during the run and scored from
those sums. There is no loader for a saved .npz activity buffer.
"""

import warnings
from dataclasses import dataclass
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, label, maximum_filter
from scipy.spatial.distance import pdist
from scipy.signal import correlate

from config import (
    AnalysisConfig, ExperimentConfig,
    world_to_normalized, world_to_flat_bins, world_to_flat_bins_3d,
)
from model.metrics import wrapped_angle_diff, unwrap_torus
from . import gongyu_scoring as gy


# Cube all scorers assume. Matches Gong & Yu [-1,1]^3.
LIM = ((-1, 1), (-1, 1), (-1, 1))

BINS         = ExperimentConfig.ratemap_bins
SMOOTH_SIGMA = AnalysisConfig.smooth_sigma


@dataclass
class ScoringInput:
    """The (x, a) pair every scorer consumes. Invariants checked in `validate`."""
    x: np.ndarray                        # positions, each axis within [-1, 1]
    a: np.ndarray                        # activity, each cell in [0, 1]
    cell_idx: np.ndarray | None = None   # which raw columns survived subsampling
    meta: dict | None = None             # free-form provenance

    @property
    def T(self) -> int:
        return self.x.shape[0]

    @property
    def n(self) -> int:
        return self.a.shape[1]

def activity_rate_map(si: ScoringInput, dims=(0, 1),
                      bins: int = BINS,
                      sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Mean activity per spatial bin. When the animal was at a positon, how active was the cell

    Works for both 2D (dims=(0,1)) and 3D (dims=(0,1,2)).

    Returns
    -------
    f : ndarray, shape (bins,)*len(dims) + (n,)
        Smoothed mean activity per bin per neuron.
    """
    ndim = len(dims)
    if ndim not in (2, 3):
        raise ValueError(f"dims must be length 2 or 3, got {ndim}")

    x   = si.x[:, list(dims)]
    lim = tuple(LIM[d] for d in dims)          # same range convention as occupancy()

    # visits per bin (unweighted) and summed activity per bin (weighted), per neuron
    counts, _ = np.histogramdd(x, bins=bins, range=lim)
    sums = np.stack(
        [np.histogramdd(x, bins=bins, range=lim, weights=si.a[:, c])[0]
         for c in range(si.n)],
        axis=-1,
    )

    # mean activity = summed activity / visits; empty bins -> 0
    f = sums / np.where(counts[..., None] > 0, counts[..., None], 1.0)

    # Smooth each neuron's map independently
    for c in range(si.n):
        f[..., c] = gaussian_filter(f[..., c], sigma=sigma)

    return f


def occupancy(si: ScoringInput, dims=(0, 1, 2),
              bins: int = BINS) -> np.ndarray:
    """Fraction of timesteps spent in each bin.

    dims controls whether a 2D or 3D occupancy map is returned.
    Must match the dims used in activity_rate_map.
    """
    lim = tuple(LIM[d] for d in dims)
    p, _ = np.histogramdd(si.x[:, list(dims)], bins=bins, range=lim,
                          density=False)
    return p / p.sum()


def occupancy_fraction(world_pos, env_size, bins, ndim=2):
    """Fraction of bins visited by a trajectory, from the same binning as the rate maps."""
    flat = world_to_flat_bins(world_pos, env_size, bins, ndim=ndim)
    counts = np.bincount(flat, minlength=bins ** ndim)
    return float((counts > 0).mean())


def autocorr2d(f: np.ndarray) -> np.ndarray:
    """Standardised 2-D autocorrelation. Slide the rate map over itself.

    Troughs (negative correlations) are kept. Nothing is thresholded to zero.

    Parameters
    ----------
    f  : (bins, bins, n)  smoothed rate map

    Returns
    -------
    ac : (2*bins-1, 2*bins-1, n), signed
    """
    if f.ndim == 2:
        f = f[..., None]
    n   = f.shape[0] * f.shape[1]
    #Spatial standard deviation of each cell’s map
    std = f.std(axis=(0, 1))
    #If a map is almost flat, set the standard deviation to 1.0 to not devide by zero
    std = np.where(std < 1e-10, 1.0, std)
    #Z-score: subtract the spatial mean, divide by that std.
    f_  = (f - f.mean(axis=(0, 1))) / std
    out = []
    #Loop over neurons.
    for i in range(f.shape[-1]):
        ac = correlate(f_[..., i], f_[..., i], mode="full") / n #slides the map over itself
        out.append(ac)
    return np.stack(out, axis=-1)


def _origin_blob_radius(ac2d: np.ndarray, thresh: float = 0.3) -> float:
    """Equivalent radius of the origin blob after thresholding.

    Grieves et al. (2021): threshold the autocorrelogram and take the
    connected component that contains the origin. Radius is sqrt(area / pi).
    """
    c = ac2d.shape[0] // 2
    if ac2d[c, c] <= thresh:
        return np.nan
    labels, _ = label(ac2d > thresh)
    n = int((labels == labels[c, c]).sum())
    return float(np.sqrt(n / np.pi))


def hex_gridness_2d(ac2d: np.ndarray, peak_thresh: float = 0.1,
                    blob_thresh: float = 0.3, hex_only: bool = False) -> tuple:
    """(hgs, sgs) hexagonal/square gridness for one 2-D autocorrelogram.

    The annulus is Grieves' mask on the six nearest peaks: pixels whose
    radius lies within r of the mean of those peak radii. r is the
    equivalent radius of the origin blob.

    Returns (nan, nan) when no ring is found. hex_only skips the square score.
    """
    d = ac2d.shape[0]
    c = d // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1] if d % 2 else np.mgrid[-c:c, -c:c]
    r = np.sqrt(xx ** 2 + yy ** 2)
    is_peak = (ac2d == maximum_filter(ac2d, size=3)) & (ac2d > peak_thresh) & (r > 2)
    radii = r[is_peak]
    if radii.size < 3:
        return np.nan, np.nan
    six = np.sort(radii)[:6]
    ring_r = float(np.mean(six))
    blob_r = _origin_blob_radius(ac2d, thresh=blob_thresh)
    if not np.isfinite(blob_r) or blob_r >= ring_r:
        return np.nan, np.nan
    lb = int(round(ring_r - blob_r))
    ub = min(int(round(ring_r + blob_r)), c)
    if ub <= lb or lb < 1:
        return np.nan, np.nan
    hgs, sgs = gy.gridness(ac2d, lb, ub, hex_only=hex_only)
    if hex_only:
        sgs = np.nan
    return hgs, sgs


# 24 headings × 30 pitches = 720 planes. Pitch steps of 3° land on the 60° cone.
GRIEVES_N_AZIMUTH = 24
GRIEVES_N_PITCH = 30
# Close-packed hexagonal planes sit at arccos(1/3) ≈ 70.53° from one another.
# 72° is that angle rounded; the rings are read at the exact angle.
GRIEVES_PITCH_HEX = float(np.degrees(np.arccos(1.0 / 3.0)))
# Angle between a close-packed hexagonal plane and a square plane: arccos(1/√3).
GRIEVES_PITCH_SQUARE = float(np.degrees(np.arccos(1.0 / np.sqrt(3.0))))
GRIEVES_PITCH_COLUMN = 60.0
# Headings for the hex-azimuth search. 5° puts 120° and 60° on the grid.
GRIEVES_HEX_AZ_STEP = 5.0


def _orientation_grid_deg():
    """Azimuth and pitch samples for the 720-plane sweep, in degrees."""
    az = np.arange(GRIEVES_N_AZIMUTH) * (360.0 / GRIEVES_N_AZIMUTH)
    pitch = np.arange(GRIEVES_N_PITCH) * (90.0 / GRIEVES_N_PITCH)
    return az, pitch


def _nanmedian(values) -> float:
    """Median of the finite entries. NaN when none of the slices had an annulus."""
    v = np.asarray(values, dtype=float).ravel()
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.nan
    return float(np.median(v))


def _plane_template(n: int) -> np.ndarray:
    """Points of one horizontal slice through the origin, in the cube's [-1, 1] frame."""
    x0 = np.linspace(-1.0, 1.0, n)
    plane = np.stack(np.meshgrid(x0, x0, indexing="ij"), axis=-1).reshape(-1, 2)
    return np.hstack((plane, np.zeros((plane.shape[0], 1))))


def _score_planes(ac: np.ndarray, azimuths_rad, altitudes_rad,
                  hex_only: bool = False,
                  interp=None, template=None) -> tuple:
    """HGS and SGS on each (azimuth, altitude) slice. NaN where no annulus exists.

    The annulus is the flat-arena one (`hex_gridness_2d`), so a slice that cannot
    define a ring is left unscored rather than entered as zero.
    """
    from scipy.interpolate import RegularGridInterpolator

    d = ac.shape[0]
    if interp is None:
        grid = np.linspace(-1.0, 1.0, d)
        interp = RegularGridInterpolator(
            (grid, grid, grid), np.ascontiguousarray(ac),
            bounds_error=False, fill_value=0.0,
        )
    if template is None:
        template = _plane_template(d)
    n_az = len(azimuths_rad)
    n_al = len(altitudes_rad)
    hgs = np.full((n_az, n_al), np.nan)
    sgs = np.full((n_az, n_al), np.nan)
    for i, az in enumerate(azimuths_rad):
        for j, al in enumerate(altitudes_rad):
            sl = np.nan_to_num(interp(gy.rot_x(template, azimuth=az, altitude=al)))
            h, s = hex_gridness_2d(sl.reshape(d, d), hex_only=hex_only)
            if np.isfinite(h):
                hgs[i, j] = h
            if np.isfinite(s):
                sgs[i, j] = s
    return hgs, sgs, interp, template


def _best_hex_azimuths(hgs_at_pitch: np.ndarray):
    """Three azimuth indices, 120° apart, whose HGS sum is largest.

    Returns None when no triplet has an annulus at all three headings.
    """
    n = len(hgs_at_pitch)
    step = n // 3
    best_idx, best_sum = None, -np.inf
    for i0 in range(step):
        idx = (i0 + np.arange(3) * step) % n
        vals = hgs_at_pitch[idx]
        if not np.isfinite(vals).all():
            continue
        total = float(vals.sum())
        if total > best_sum:
            best_sum = total
            best_idx = idx
    return best_idx


def grieves_chi_1cell(ac_cell: np.ndarray) -> np.ndarray:
    """(3,) structure scores (fcc, hcp, col) for one 3-D autocorrelogram.

    Each score is a difference of medians, χ = α − β, taken after the cube is
    rotated so the slice with the highest hexagonal gridness lies flat. Pitch is
    then the tilt away from that best plane. Slices with no annulus are left
    out of both medians.

    The 720-plane sweep is 24 azimuths by 30 pitches. It finds the best plane
    and supplies x_COL. Hexagonal headings are then the three azimuths 120°
    apart with the largest total HGS at pitch arccos(1/3) ≈ 70.53° (the 72°
    and 70.5° figures in the arrangement table). Square headings for FCC sit
    60° away from those. The square pitch is arccos(1/√3) ≈ 54.7°. Neither
    pitch lies on the 3° grid, so those slices are cut on their own.

    x_FCC compares square gridness at 54.7° on the offset headings with the
    same pitch on the hexagonal headings. x_HCP compares square gridness at
    54.7° and at 70.53° on the hexagonal headings. x_COL compares hexagonal
    gridness within 60° of the best plane with hexagonal gridness at the
    steeper pitches.
    """
    ac = np.asarray(ac_cell, dtype=float)
    if ac.ndim == 4:
        ac = ac[..., 0]
    az_deg, pitch_deg = _orientation_grid_deg()
    az = np.deg2rad(az_deg)
    pitch = np.deg2rad(pitch_deg)

    hgs_lab, _, _, _ = _score_planes(ac, az, pitch, hex_only=True)
    if not np.isfinite(hgs_lab).any():
        return np.full(3, np.nan)
    i_az, i_pitch = np.unravel_index(int(np.nanargmax(hgs_lab)), hgs_lab.shape)
    aligned = _rotate_cube_to_horizontal(ac[..., None], az[i_az], pitch[i_pitch])[..., 0]

    hgs, _, interp, template = _score_planes(aligned, az, pitch, hex_only=True)
    out = np.full(3, np.nan)

    near = pitch_deg <= GRIEVES_PITCH_COLUMN
    out[2] = _nanmedian(hgs[:, near]) - _nanmedian(hgs[:, ~near])

    az_fine_deg = np.arange(0.0, 360.0, GRIEVES_HEX_AZ_STEP)
    az_fine = np.deg2rad(az_fine_deg)
    hgs_hex, sgs_hex, _, _ = _score_planes(
        aligned, az_fine, [np.deg2rad(GRIEVES_PITCH_HEX)],
        interp=interp, template=template,
    )
    hex_idx = _best_hex_azimuths(hgs_hex[:, 0])
    if hex_idx is None:
        return out

    off_idx = (hex_idx + (len(az_fine_deg) // 6)) % len(az_fine_deg)
    headings = az_fine[np.concatenate([hex_idx, off_idx])]
    _, sgs_sq, _, _ = _score_planes(
        aligned, headings, [np.deg2rad(GRIEVES_PITCH_SQUARE)],
        interp=interp, template=template,
    )
    sgs_at_square_hex = sgs_sq[:3, 0]
    sgs_at_square_off = sgs_sq[3:, 0]
    sgs_at_hex_pitch = sgs_hex[hex_idx, 0]
    out[0] = _nanmedian(sgs_at_square_off) - _nanmedian(sgs_at_square_hex)
    out[1] = _nanmedian(sgs_at_square_hex) - _nanmedian(sgs_at_hex_pitch)
    return out


def autocorrelation_1cell(f_cell: np.ndarray) -> np.ndarray:
    """Standardised 3-D autocorrelation of a single cell's rate map.

    Parameters
    ----------
    f_cell : (b, b, b) one neuron's smoothed rate map

    Returns
    -------
    ac : (2b-1, 2b-1, 2b-1) signed autocorrelogram for that cell.
    """
    from scipy.signal import correlate
    nbins = f_cell.size
    std = f_cell.std()
    std = std if std > 1e-10 else 1.0
    f_ = (f_cell - f_cell.mean()) / std
    ac = correlate(f_, f_, mode="full") / nbins
    return ac


def chi_1cell(ac_cell: np.ndarray, align: bool = False) -> np.ndarray:
    """(3,) structure scores (fcc, hcp, col) for one cell's 3-D autocorrelogram.

    These are the Grieves differences in `grieves_chi_1cell`. `align` is accepted
    so older callers still run; the scores are always measured from the cell's
    own best plane, so the flag does not change the result.
    """
    del align
    return grieves_chi_1cell(ac_cell)


def _rotate_cube_to_horizontal(cube: np.ndarray, az: float, al: float) -> np.ndarray:
    """Resample an autocorrelation cube so the (az, al) slice lies in z = 0.

    `oblique_slice` reads that slice at coordinates rot_x(horizontal, az, al).
    Sampling the original cube at those coordinates puts the slice flat.
    Negating the angles is not the inverse: azimuth and altitude do not commute.
    """
    from scipy.interpolate import RegularGridInterpolator
    d = cube.shape[0]
    grid = np.linspace(-1, 1, d)
    interp = RegularGridInterpolator((grid, grid, grid), cube[..., 0],
                                     bounds_error=False, fill_value=0.0)
    gx, gy_, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([gx.ravel(), gy_.ravel(), gz.ravel()], axis=-1)
    pts_rot = gy.rot_x(pts, azimuth=az, altitude=al)
    out = interp(pts_rot).reshape(d, d, d)
    return out[..., None]


def mra_1cell(ac_cell: np.ndarray) -> np.ndarray:
    """(rmax,) modified radial autocorrelation (sum / sqrt(N(r))) for one cell."""
    rmax = ac_cell.shape[0] // 2
    return gy.autocorr_radial3d(ac_cell, rmax, method="mean_comp")


def global_order_1cell(ac_cell: np.ndarray, az_precision: int = 48,
                       al_precision: int = 24, radial_method: str = "max") -> float:
    """Orientation-invariant global-order score for one cell.
    """
    hgs_map, _ = gy.gridness_map(ac_cell[..., None], az_precision=az_precision,
                                 al_precision=al_precision, al_max=np.pi / 2,
                                 radial_method=radial_method, hex_only=True)
    return float(hgs_map.max())


def _spatial_info(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Skaggs spatial information (bits/sample), dimension-agnostic.

    """
    ndim = f.ndim - 1
    ax = tuple(range(ndim))
    p = p[..., None]
    lam_bar = (f * p).sum(axis=ax)                 # mean rate per cell
    f_ = f / np.where(lam_bar != 0, lam_bar, 1.0)  # lambda_i / lambda_bar
    f_ = np.where(f_ == 0, 1.0, f_)                # lim x->0 x log x = 0
    return (np.log2(f_) * p * f_).sum(axis=ax)


def _sparsity_idx(p: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Sparsity index (E[f]^2 / E[f^2]), does not care about dimension"""
    ndim = f.ndim - 1
    ax = tuple(range(ndim))
    p = p[..., None]
    e_f = (f * p).sum(axis=ax)
    e_f2 = (f ** 2 * p).sum(axis=ax)
    return e_f ** 2 / np.where(e_f2 != 0, e_f2, 1.0)


def info_scores(si: ScoringInput, p: np.ndarray, f: np.ndarray,
                n_shuffle: int = 50, seed: int = 0,
                bins: int = BINS, sigma: float = SMOOTH_SIGMA) -> dict:
    """Spatial information, sparsity, and their shuffle Z-scores.
 
    Null: permute each neuron's activity in time (breaks the position-activity
    link) and recompute the rate map. Matches G&Y's si_shuffle. 2-D or 3-D.
 
    p : occupancy map, must match f's spatial dims.
    f : rate map, (bins,)*ndim + (n,).
    """
    rng   = np.random.default_rng(seed)
    ndim  = f.ndim - 1             # spatial dimensions of the rate map
    dims  = tuple(range(ndim))

    sinfo = _spatial_info(p, f)
    sidx  = _sparsity_idx(p, f)

    sinfo_sf = np.zeros((n_shuffle, si.n))
    sidx_sf  = np.zeros((n_shuffle, si.n))

    for i in range(n_shuffle):
        # Permute each neuron's activity independently (temporal shuffle)
        a_sf = np.stack(
            [rng.permutation(si.a[:, c]) for c in range(si.n)], axis=1
        )
        si_sf = ScoringInput(x=si.x, a=a_sf, cell_idx=si.cell_idx)
        f_sf  = activity_rate_map(si_sf, dims=dims, bins=bins, sigma=sigma)
        sinfo_sf[i] = _spatial_info(p, f_sf)
        sidx_sf[i]  = _sparsity_idx(p, f_sf)

    sinfo_z = (sinfo - sinfo_sf.mean(0)) / np.where(
        sinfo_sf.std(0) > 1e-10, sinfo_sf.std(0), 1.0)
    sidx_z  = (sidx  - sidx_sf.mean(0))  / np.where(
        sidx_sf.std(0)  > 1e-10, sidx_sf.std(0),  1.0)

    return dict(sinfo=sinfo, sidx=sidx, sinfo_z=sinfo_z, sidx_z=sidx_z)


def field_kernel(grid_spacing: float, env_size: float) -> tuple:
    """MeanShift bandwidth and wall cutoff, in the normalised [-1, 1] cube.

    Bandwidth is one third of the grid spacing, the fraction Gong & Yu used
    relative to their lattice. For a 0.45 m spacing in a 2 m box that third is
    0.15; the default 0.48 m spacing gives 0.16. Centers closer to a wall than
    half a grid spacing are dropped, because the wall pulls them inward.
    """
    spacing_n = float(grid_spacing) / (float(env_size) / 2.0)
    bandwidth = spacing_n / 3.0
    ignore_range = max(0.0, 1.0 - 0.5 * spacing_n)
    return bandwidth, ignore_range


def field_trace_stride(n_steps: int, n_cells: int,
                       max_bytes: int = 256 * 1024 * 1024) -> int:
    """Time stride that keeps a (position, activity) trace under `max_bytes`.

    Stride 1 stores every step. Longer runs keep every k-th step, and the 90th
    percentile is taken on that downsampled trace.

    Experiment 2 does not use this. It passes an explicit stride and
    `require_trace_budget` raises if that stride does not fit.
    """
    n_steps = int(n_steps)
    bytes_per = 4 * int(max(n_cells, 1)) + 12
    max_samples = max(1, int(max_bytes) // bytes_per)
    return max(1, int(np.ceil(n_steps / max_samples)))


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


def inter_field_distances(si: ScoringInput | None = None, activity_pct: float = 90.0,
                          bandwidth: float | None = None, min_bin_freq: int = 25,
                          min_cluster_size: int = 30,
                          ignore_range: float | None = None,
                          positions: np.ndarray | None = None,
                          activity: np.ndarray | None = None,
                          grid_spacing: float | None = None,
                          env_size: float | None = None,
                          world_coords: bool = False) -> dict:
    """Distances between activity-field centers of each cell.

    A field center is the mean of one MeanShift cluster. The points in a cluster
    are the positions where that cell's activity was at or above its 90th
    percentile. This cut is only used here; rate maps and gridness use the
    continuous activity.

    Pass a ScoringInput, or `positions` (T, 3) with `activity` (T, n). World
    positions in metres are normalised when `world_coords` is set.
    """
    if si is not None:
        positions = si.x
        activity = si.a
        world_coords = False
        if si.meta:
            if env_size is None:
                env_size = si.meta.get("env_size")
            if grid_spacing is None:
                grid_spacing = si.meta.get("grid_spacing")
    if positions is None or activity is None:
        raise ValueError("inter_field_distances needs a ScoringInput or positions and activity")
    if grid_spacing is None:
        grid_spacing = ExperimentConfig.grid_spacing
    if env_size is None:
        env_size = ExperimentConfig.env_size
    if bandwidth is None or ignore_range is None:
        bw, ir = field_kernel(grid_spacing, env_size)
        if bandwidth is None:
            bandwidth = bw
        if ignore_range is None:
            ignore_range = ir
    if world_coords:
        positions = world_to_normalized(positions, env_size)

    out = {}
    n = activity.shape[1]
    for k in range(n):
        threshold  = np.percentile(activity[:, k], activity_pct)
        active_pos = positions[activity[:, k] >= threshold]
        if len(active_pos) < min_bin_freq:        # too few points to seed a bin
            out[k] = np.array([])
            continue
        try:
            _, _, centers = gy.ms_cluster(
                active_pos, bandwidth=bandwidth, min_bin_freq=min_bin_freq,
                min_cluster_size=min_cluster_size, ignore_range=ignore_range,
                plot=False)
        except ValueError:
            out[k] = np.array([])                 # MeanShift found no seeds
            continue
        out[k] = pdist(centers) if len(centers) >= 2 else np.array([])
    return out


def _rate_map_from_accumulator(sums: np.ndarray, counts: np.ndarray,
                               sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Smooth sums and counts, then divide.

    Works for any spatial dimensionality:
        2-D input : sums (b, b, n),   counts (b, b)
        3-D input : sums (b, b, b, n), counts (b, b, b)

    Returns f of the same shape as sums.
    """
    counts = np.asarray(counts, dtype=np.float64) # is how many times each spatial bin was visited.
    sums = np.asarray(sums, dtype=np.float64)
    raw = counts > 0 # true or false map of which bins were visited.
    counts_s = gaussian_filter(counts.astype(np.float32), sigma=sigma) #spread activity map.
    f = np.empty(sums.shape, dtype=np.float64) 
    
    #builds one rate map per cell and stores it in f.
    for c in range(sums.shape[-1]):
        sums_s = gaussian_filter(sums[..., c].astype(np.float32), sigma=sigma)
        rate = np.zeros(counts.shape, dtype=np.float64)
        np.divide(sums_s, counts_s, out=rate, where=counts_s > 0)
        if raw.any():
            mu = float(rate[raw].mean())
        else:
            mu = 0.0
        rate[~raw] = mu #fill in the zeros with the mean rate of the visited bins.
        f[..., c] = rate
    return f

def _score_one_cell(f_k, align, global_order, go_precision):
    """Score one neuron's rate map: chi, MRA, and (if asked) global order."""
    ac_k  = autocorrelation_1cell(f_k)
    chi_k = chi_1cell(ac_k, align=align)
    mra_k = mra_1cell(ac_k)
    go_k  = np.nan
    if global_order:
        go_k = global_order_1cell(ac_k, az_precision=go_precision[0],
                                  al_precision=go_precision[1])
    return chi_k, mra_k, go_k


def _stream_structure_scores(
    f: np.ndarray,
    bins: int,
    align: bool = False,
    global_order: bool = False,
    go_precision: tuple = (48, 24),
    n_jobs: int = 1,
) -> tuple:
    """Work out the chi / MRA / (if asked) global-order scores, one neuron at a time.

    score_3d_from_map leans on this.
    We only ever keep one neuron's big autocorrelation cube in memory at once, so
    this stays light even with lots of neurons.

    Returns
    -------
    chi_out    (3, n)   the fcc / hcp / col structure scores
    mra_out    (n, rmax)
    go_out     (n,) or None
    ring_found (n,) bool
    no_ring    (n,) bool
    n_failed   int
    """
    n    = f.shape[-1]
    rmax = (2 * bins - 1) // 2
    chi_out = np.full((3, n), np.nan)
    mra_out = np.zeros((n, rmax))
    go_out  = np.full(n, np.nan) if global_order else None

    if n_jobs == 1:
        # one neuron at a time, the original way
        for k in range(n):
            chi_k, mra_k, go_k = _score_one_cell(f[..., k], align,
                                                 global_order, go_precision)
            chi_out[:, k] = chi_k
            mra_out[k]    = mra_k
            if global_order:
                go_out[k] = go_k
    else:
        # spread neurons over cores; pin each worker to one thread so
        # scipy/BLAS don't fight over cores
        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_score_one_cell)(f[..., k].copy(), align,
                                         global_order, go_precision)
                for k in range(n)
            )
        for k, (chi_k, mra_k, go_k) in enumerate(results):
            chi_out[:, k] = chi_k
            mra_out[k]    = mra_k
            if global_order:
                go_out[k] = go_k

    # A missing annulus is NaN and stays out of the median. A measured zero is a
    # real score: an HCP cell is supposed to give x_FCC near zero. "No ring" means
    # none of the three differences could be computed.
    ring_found = np.isfinite(chi_out).any(axis=0)
    no_ring    = ~ring_found
    return chi_out, mra_out, go_out, ring_found, no_ring, int(no_ring.sum())


def _online_shuffle_zscores(
    shuf_sums: np.ndarray,
    scored_idx: np.ndarray,
    denom: np.ndarray,
    p: np.ndarray,
    sigma: float,
    sinfo: np.ndarray,
    sidx: np.ndarray,
) -> tuple:
    """Work out sinfo_z / sidx_z using the slide-in-time shuffle (RAM way only).

    This is a different test from the one in info_scores (which shuffles fully at
    random). Sliding each neuron forward in time keeps the neuron's own slow wiggle,
    which makes it a tougher test for CAN activity. So do not put these Z-scores
    head to head with the full-shuffle ones from the disk way.

    Parameters
    ----------
    shuf_sums  : (n_shuffle, *spatial, n_sub)  the shuffle maps we built earlier
    scored_idx : which neuron columns to score
    denom      : safe visit-count divider, shape (*spatial, 1)
    p          : fraction of time spent in each bin, shape (*spatial,)
    sigma      : how much to smooth, in bins
    sinfo, sidx: the real (un-shuffled) values, each (n_scored,)

    Returns
    -------
    (sinfo_z, sidx_z)  both (n_scored,) float64
        Rule of thumb: Z of 2.58 or more means 99% sure (Gong and Yu, Figure 4).
    """
    n_shuffle = shuf_sums.shape[0]
    n_scored  = len(scored_idx)
    sinfo_shuf = np.zeros((n_shuffle, n_scored), dtype=np.float32)
    sidx_shuf  = np.zeros((n_shuffle, n_scored), dtype=np.float32)

    for j in range(n_shuffle):
        f_j = shuf_sums[j][..., scored_idx] / denom
        for c in range(n_scored):
            f_j[..., c] = gaussian_filter(f_j[..., c].astype(np.float32), sigma=sigma)
        sinfo_shuf[j] = _spatial_info(p, f_j).astype(np.float32)
        sidx_shuf[j]  = _sparsity_idx(p, f_j).astype(np.float32)
        del f_j

    sinfo_z = (sinfo - sinfo_shuf.mean(0)) / (sinfo_shuf.std(0) + 1e-9)
    sidx_z  = (sidx  - sidx_shuf.mean(0))  / (sidx_shuf.std(0)  + 1e-9)
    return sinfo_z.astype(np.float64), sidx_z.astype(np.float64)


def random_hgs_floor(bins=40, n_maps=1000, sigma=SMOOTH_SIGMA, seed=0):
    """HGS of smoothed white-noise maps. Returns (mean, 95th percentile).

    The 95th percentile is the absolute grid-like floor. n_maps should be at
    least 1000 so that percentile rests on more than ten samples.
    """
    rng = np.random.default_rng(seed)
    hgs = np.empty(int(n_maps), dtype=float)
    for i in range(int(n_maps)):
        m = gaussian_filter(rng.random((bins, bins)), sigma=sigma)
        h, _ = hex_gridness_2d(autocorr2d(m)[..., 0])
        hgs[i] = h if np.isfinite(h) else np.nan
    return float(np.nanmean(hgs)), float(np.nanpercentile(hgs, 95))


def apply_hgs_floor(scores: dict, floor: float,
                    require_hgs_gt_sgs: bool = False) -> dict:
    """Mark grid_like against one HGS floor. Mutates and returns `scores`."""
    hgs = np.asarray(scores["hgs"], dtype=float)
    sgs = np.asarray(scores["sgs"], dtype=float)
    # finite HGS above the shared chance floor
    like = np.isfinite(hgs) & (hgs > floor)
    if require_hgs_gt_sgs:
        like = like & (hgs > sgs)
    scores["grid_like"] = like
    return scores


def summarize_2d(scores):
    """One-row summary of score_2d_from_map for a table."""
    hgs = np.asarray(scores["hgs"], dtype=float)
    sgs = np.asarray(scores["sgs"], dtype=float)
    finite = np.isfinite(hgs) & np.isfinite(sgs)
    n_fin = int(finite.sum())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_hgs = float(np.nanmedian(hgs))
        median_sgs = float(np.nanmedian(sgs))
    return dict(
        occupancy=float(scores["occupancy"]),
        ring_frac=float(scores["ring_frac"]),
        median_hgs=median_hgs,
        median_sgs=median_sgs,
        frac_hgs_gt_sgs=float(np.mean(hgs[finite] > sgs[finite])) if n_fin else float("nan"),
        frac_grid_like=float(np.mean(scores["grid_like"])),
        n_active=int(scores["n_active"]),
        hgs_shuf_p95=(
            float("nan") if scores.get("hgs_shuf_p95") is None
            else float(scores["hgs_shuf_p95"])
        ),
    )


def percentile_cells(hgs, percentiles=(10, 50, 90)):
    """Indices into a scored-cell array at fixed HGS percentiles (not top-N)."""
    hgs = np.asarray(hgs, dtype=float)
    order = np.argsort(np.where(np.isfinite(hgs), hgs, -np.inf))
    n = len(hgs)
    if n == 0:
        return []
    out = []
    for p in percentiles:
        out.append(int(order[int(np.clip(p / 100.0 * (n - 1), 0, n - 1))]))
    return out


def _segment_reset(integ, traj):
    """Steps that reset the plane prior, when that modelling choice is on."""
    if not getattr(integ, "reset_prior_at_segment_start", False):
        return set()
    starts = getattr(traj, "segment_starts", None)
    if starts is None:
        return set()
    return {int(s) for s in np.asarray(starts).ravel()}


def _label_fields(traj):
    """Route labels from the walk that was actually integrated."""
    return dict(
        route_of=getattr(traj, "route_of", None),
        flight_of=getattr(traj, "flight_of", None),
        route_table=getattr(traj, "route_table", None),
        flight_table=getattr(traj, "flight_table", None),
        segment_id=getattr(traj, "segment_id", None),
        segment_starts=getattr(traj, "segment_starts", None),
        plane_id=getattr(traj, "plane_id", None),
        condition=getattr(traj, "condition", None),
    )


def _hold_and_true(integ, exp, traj, T, refresh_mask, n_true_seq):
    """Refresh mask, true-normal schedule, and decode buffer for the scoring runners."""
    dim = integ.qan.manifold.dim
    #Check if refresh schedule is given, if not, build it from the config.
    if refresh_mask is None:
        refresh_mask = integ.default_refresh_mask(
            T, timing=exp.config.plane.refresh_timing,
            segment_starts=getattr(traj, "segment_starts", None),
        )
    if n_true_seq is None:
        n_true_seq = np.broadcast_to(integ._true_n_hat, (T, 3)) #broadcast a smaller array to the full length of the trajectory

    n_true_seq = np.asarray(n_true_seq, dtype=float)
    #buffer for decoded bump
    theta_hist = np.zeros((T, dim), dtype=np.float64)
    return refresh_mask, n_true_seq, theta_hist


def run_with_online_ratemap(exp, g, bins: int = 40, n_warmup: int = 100,
                            mode: str = "integrate",
                            trajectory=None,
                            n_shuffle: int = 0,
                            shuffle_min_lag_frac: float = 0.1,
                            seed: int = 0,
                            refresh_mask=None) -> dict:
    """Memory light 2-D run (just the XY floor).

    We add the activity up into a (bins², N) map on the device as we go, so the huge
    per-step buffer is never needed. One trip back to the CPU at the very end. If you
    want the full 3-D version use run_3d_online instead.

    Parameters
    ----------
    exp        : a BaseExperiment subclass
    g          : gravity vector
    trajectory : optional (world_pos, v_body_seq, torus_gt) tuple.
                 Hand this in to reuse a path you already have from
                 run_experiment(), so we do not run the network a second time
                 just to get the same path back.
    n_shuffle  : number of time-lag shuffle maps to build online (0 = disabled).
                 When > 0, shuf_sums is returned and can be passed to
                 score_2d_from_map for circular-shift HGS without
                 storing the full per-step activity buffer.
    shuffle_min_lag_frac : minimum lag as fraction of T. Upper bound is 0.9 T
                 (same window as experiments.base).
    seed       : RNG seed for lag sampling.
    refresh_mask : optional (T,) bool.
    """
    from model.path_integration import PathIntegrator

    cfg   = exp.config.experiment
    integ = PathIntegrator(qan=exp.qan, **exp.integrator_kwargs)
    bk    = integ.backend
    dev   = getattr(bk, "device", torch.device("cpu"))

    if trajectory is not None:
        traj = trajectory
    else:
        traj = exp.generate_trajectory()
    world_pos, v_body_seq, torus_gt = traj
    n_true_seq = getattr(traj, "n_true_seq", None)

    T, N = world_pos.shape[0], bk.S.shape[1]
    integ.reset(torus_gt[0])
    if mode == "integrate" and n_warmup:
        integ.warmup(n_warmup)

    # arena-anchored 2-D bin number for each step (XY only)
    flat_2d = world_to_flat_bins(world_pos, cfg.env_size, bins)     # (T,) int64

    sums   = torch.zeros((bins * bins, N), dtype=torch.float32, device=dev)
    counts = torch.zeros(bins * bins,      dtype=torch.float32, device=dev)

    # shuffle accumulators — only allocated when asked for
    shuf_sums_d = None
    rng = np.random.default_rng(seed)
    if n_shuffle > 0:
        min_lag = max(1, int(T * shuffle_min_lag_frac))
        max_lag = max(min_lag + 1, int(T * 0.9))  # [0.1T, 0.9T), same as base.py
        if min_lag >= max_lag:
            warnings.warn(
                f"run_with_online_ratemap: T={T} too short for n_shuffle={n_shuffle} "
                f"(min_lag={min_lag} >= max_lag={max_lag}). Disabling shuffle.",
                UserWarning, stacklevel=2,
            )
            n_shuffle = 0
        else:
            lags        = rng.integers(min_lag, max_lag, size=n_shuffle)
            shuf_sums_d = torch.zeros(
                (n_shuffle, bins * bins, N), dtype=torch.float32, device=dev)

    #get values for the integrete loop
    refresh_mask, n_true_seq, theta_hist = _hold_and_true(
        integ, exp, traj, T, refresh_mask, n_true_seq)
    reset_at = _segment_reset(integ, traj)
    #integrete loop
    for t in range(T):
        #run the QAN + plane filter
        if mode == "integrate":
            theta_hist[t] = integ.step(
                v_body_seq[t], g, n_true=n_true_seq[t],
                #Whether this step is a hold-refresh.
                refresh=bool(refresh_mask[t]),
                reset_prior=(t in reset_at))
        else:
            bk.reset(torus_gt[t])
            theta_hist[t] = torus_gt[t]
        s = bk.S.mean(dim=0).squeeze()          # (N,) on device, no per-step CPU copy
        b = int(flat_2d[t])
        sums[b]   += s
        counts[b] += 1.0
        if shuf_sums_d is not None:
            # activity at t goes into the bin visited at t+lag (wraps around)
            for j in range(n_shuffle):
                b_shuf = int(flat_2d[(t + lags[j]) % T])
                shuf_sums_d[j, b_shuf] += s

    shuf_sums = None
    if shuf_sums_d is not None:
        shuf_sums = shuf_sums_d.cpu().numpy().reshape(
            n_shuffle, bins, bins, N)

    return dict(
        sums=sums.cpu().numpy().reshape(bins, bins, N),
        counts=counts.cpu().numpy().reshape(bins, bins),
        shuf_sums=shuf_sums,
        bins=bins, world_pos=world_pos, torus_gt=torus_gt, theta_hist=theta_hist,
    )

def run_3d_online(
    exp,
    g_vec: np.ndarray,
    *,
    trajectory=None,
    bins: int = 25,
    n_sub: int = 300,
    seed: int = 0,
    n_warmup: int = 100,
    active_thresh: float = 1e-3,
    n_shuffle: int = 0,
    shuffle_min_lag_frac: float = 0.1,
    refresh_mask=None,
    field_stride: int | None = None,
    max_trace_bytes: int | None = None,
) -> dict:
    """RAM-only 3-D run. Never makes the big S_tot_buffer.

    We step the network one timestep at a time and add the activity into a
    (bins³, n_sub) tensor on the device. One trip back to the CPU at the end. If you
    ask for shuffle maps too, score_3d_from_map can then work out sinfo_z / sidx_z
    without ever needing the full per-step activity buffer.

    """
    from model.path_integration import PathIntegrator

    g_vec = np.asarray(g_vec, dtype=float)
    cfg   = exp.config.experiment

    # get the path: reuse one if handed in, otherwise make a fresh one
    if trajectory is not None:
        traj = trajectory
    else:
        traj = exp.generate_trajectory()
    world_pos, v_body_seq, torus_gt = traj
    n_true_seq = getattr(traj, "n_true_seq", None)
    T = world_pos.shape[0]

    # the thing that steps the network
    integ = PathIntegrator(qan=exp.qan, **exp.integrator_kwargs)
    bk    = integ.backend
    dev   = getattr(bk, "device", torch.device("cpu"))

    # pick which neurons to follow up front, keeps the accumulator small
    N        = bk.S.shape[1]
    rng      = np.random.default_rng(seed)
    sub_idx  = np.sort(rng.choice(N, size=min(n_sub, N), replace=False))
    sub_t    = torch.tensor(sub_idx, dtype=torch.long, device=dev)
    n_sub_actual = len(sub_idx)

    # the running totals, kept on the device
    sums_d   = torch.zeros((bins ** 3, n_sub_actual), dtype=torch.float32, device=dev)
    counts_d = torch.zeros(bins ** 3,                 dtype=torch.float32, device=dev)

    # extra totals for the shuffle, only if asked for
    shuf_sums_d = None
    lags        = None
    if n_shuffle > 0:
        min_lag = max(1, int(T * shuffle_min_lag_frac))
        if min_lag >= T:
            warnings.warn(
                f"run_3d_online: T={T} too short for n_shuffle={n_shuffle} "
                f"(min_lag={min_lag} ≥ T). Disabling shuffle.",
                UserWarning, stacklevel=2,
            )
            n_shuffle = 0
        else:
            lags        = rng.integers(
                min_lag, max(min_lag + 1, int(0.9 * T)), size=n_shuffle,
            )
            shuf_sums_d = torch.zeros(
                (n_shuffle, bins ** 3, n_sub_actual),
                dtype=torch.float32, device=dev,
            )

    # work out which 3-D box each step lands in, all at once
    flat3d = world_to_flat_bins_3d(world_pos, cfg.env_size, bins)   # (T,) int64

    # put the bump at the start and let things settle
    integ.reset(torus_gt[0])
    integ.warmup(n_warmup)

    # Strided activity trace. The stride is explicit: it is not coarsened to
    # stay under a byte cap. require_trace_budget raises instead.
    if field_stride is None:
        field_stride = AnalysisConfig.field_stride
    if max_trace_bytes is None:
        max_trace_bytes = AnalysisConfig.max_trace_bytes
    stride = int(field_stride)
    require_trace_budget(T, n_sub_actual, stride, max_trace_bytes)
    n_samp = (T + stride - 1) // stride
    field_pos = np.empty((n_samp, 3), dtype=np.float32)
    field_act_d = torch.empty((n_samp, n_sub_actual), dtype=torch.float32, device=dev)
    k_samp = 0

    # the main loop: step, then drop the activity into its box
    #get for the loop same as above
    refresh_mask, n_true_seq, theta_hist = _hold_and_true(
        integ, exp, traj, T, refresh_mask, n_true_seq)
    reset_at = _segment_reset(integ, traj)
    for t in range(T):
        theta_hist[t] = integ.step(
            v_body_seq[t], g_vec, n_true=n_true_seq[t],
            refresh=bool(refresh_mask[t]),
            reset_prior=(t in reset_at))

        s = bk.S.mean(dim=0).squeeze()     # (N,) on device
        b = int(flat3d[t])
        sums_d[b]   += s[sub_t]
        counts_d[b] += 1.0
        if t % stride == 0:
            field_act_d[k_samp] = s[sub_t]
            field_pos[k_samp] = world_pos[t]
            k_samp += 1

        if shuf_sums_d is not None:
            for j in range(n_shuffle):
                b_shuf = int(flat3d[(t + lags[j]) % T])
                shuf_sums_d[j, b_shuf] += s[sub_t]

    # one trip back to the CPU
    sums   = sums_d.cpu().numpy().reshape(bins, bins, bins, n_sub_actual)
    counts = counts_d.cpu().numpy().reshape(bins, bins, bins)

    shuf_sums = None
    if shuf_sums_d is not None:
        shuf_sums = shuf_sums_d.cpu().numpy().reshape(
            n_shuffle, bins, bins, bins, n_sub_actual
        )

    # mark the neurons that actually did something, and grab the filter's history
    span        = sums.reshape(-1, n_sub_actual).max(0) - sums.reshape(-1, n_sub_actual).min(0)
    active_mask = span > active_thresh * (span.max() + 1e-12)

    n_hat_hist = np.array(integ.history["n_hat"])
    n_held_hist = np.array(integ.history["n_held"])
    if "kappa" in integ.history and len(integ.history["kappa"]):
        kappa_hist = np.array(integ.history["kappa"])
    else:
        kappa_hist = np.array(integ.history["z2"]) - np.array(integ.history["z1"])
    gap_hist = kappa_hist  # alias

    return dict(
        sums=sums, counts=counts,
        sub_idx=sub_idx, active_mask=active_mask,
        world_pos=world_pos, torus_gt=torus_gt,
        theta_hist=theta_hist, bins=bins,
        n_hat_hist=n_hat_hist, n_held_hist=n_held_hist,
        gap_hist=gap_hist, kappa_hist=kappa_hist,
        shuf_sums=shuf_sums, n_shuffle=n_shuffle,
        field_pos=field_pos[:k_samp],
        field_act=field_act_d[:k_samp].cpu().numpy(),
        field_stride=stride,
        **_label_fields(traj),
    )


def _ifd_for_scored(positions, activity, scored_idx, grid_spacing, env_size):
    """Inter-field distances for the cells `score_3d_from_map` keeps.

    `activity` columns match the rate-map axis. Keys of the returned dict are
    indices into the scored-cell axis, the same axis as `chi`.
    """
    if positions is None or activity is None:
        return None
    activity = np.asarray(activity)
    if activity.ndim != 2 or activity.shape[1] <= int(np.max(scored_idx, initial=-1)):
        raise ValueError(
            f"activity cell axis {getattr(activity, 'shape', None)} does not cover "
            f"scored index {int(np.max(scored_idx))}"
        )
    return inter_field_distances(
        positions=np.asarray(positions),
        activity=activity[:, scored_idx],
        grid_spacing=grid_spacing,
        env_size=env_size,
        world_coords=True,
    )


def score_3d_from_map(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    shuf_sums: np.ndarray | None = None,
    sigma: float = SMOOTH_SIGMA,
    align: bool = False,
    global_order: bool = False,
    go_precision: tuple = (48, 24),
    occ_warn: float = 0.5,
    occ_error: float = 0.15,
    active_mask: np.ndarray | None = None,
    n_jobs: int = 1,
    positions: np.ndarray | None = None,
    activity: np.ndarray | None = None,
    grid_spacing: float | None = None,
    env_size: float | None = None,
) -> dict:
    """Score a 3-D rate map we already built. No .npz file, no ScoringInput needed.

    `positions` (T, 3) in metres and `activity` (T, n_cells) are the trace
    inter-field distance clusters. `n_cells` is the same axis as `sums`.
    Without them, `ifd` is None. Structure scores are the Grieves differences.

    Note (Gong and Yu, Figure 4): Z of 2.58 or more.
    """
    bins    = sums.shape[0]
    n_total = sums.shape[-1]

    # which neurons are we scoring
    scored_idx = np.where(active_mask)[0] if active_mask is not None \
                 else np.arange(n_total)

    if len(scored_idx) == 0:
        warnings.warn("score_3d_from_map: no active neurons to score.", UserWarning, stacklevel=2)
        return dict(chi=None, mra=None, ring_found=None, no_ring=None, n_failed=0,
                    sinfo=None, sidx=None, sinfo_z=None, sidx_z=None, ifd=None,
                    peak=None, occupancy=0.0, reliable=False,
                    scored_idx=scored_idx, bins=bins, aligned=align)

    # turn the totals into a smoothed rate map
    sums_sub = sums[..., scored_idx].copy()
    denom    = np.where(counts > 0, counts, 1.0)[..., None]
    f        = _rate_map_from_accumulator(sums_sub, counts, sigma=sigma)

    # how much of the box did we actually cover, and can we trust it
    p   = counts / (counts.sum() + 1e-30)
    occ = float((p > 0).mean())

    reliable = True
    if occ < occ_error:
        warnings.warn(
            f"score_3d_from_map: occupancy {occ:.1%} < {occ_error:.0%}. "
            f"Structural scores are NaN-filled (reliable=False). "
            f"Minimum T for occ ≥ {occ_error:.0%} at bins={bins}: "
            f"≳ {int(occ_error * bins**3)} steps.  "
            f"Try bins=12 for short exploratory runs.",
            UserWarning, stacklevel=2,
        )
        reliable = False
        n_scored = len(scored_idx)
        rmax     = (2 * bins - 1) // 2
        sinfo    = _spatial_info(p, f)
        sidx     = _sparsity_idx(p, f)
        sinfo_z, sidx_z = None, None
        if shuf_sums is not None:
            sinfo_z, sidx_z = _online_shuffle_zscores(
                shuf_sums, scored_idx, denom, p, sigma, sinfo, sidx)
        out = dict(
            chi=np.full((3, n_scored), np.nan),
            mra=np.full((n_scored, rmax), np.nan),
            ring_found=np.zeros(n_scored, bool), no_ring=np.zeros(n_scored, bool),
            n_failed=n_scored,
            sinfo=sinfo, sidx=sidx, sinfo_z=sinfo_z, sidx_z=sidx_z,
            ifd=_ifd_for_scored(positions, activity, scored_idx, grid_spacing, env_size),
            peak=f.max(axis=(0, 1, 2)), occupancy=occ, reliable=False,
            scored_idx=scored_idx, bins=bins, aligned=align,
        )
        return out

    elif occ < occ_warn:
        warnings.warn(
            f"score_3d_from_map: occupancy {occ:.1%} — map is sparse. "
            f"chi/MRA may reflect coverage more than tuning. "
            f"T ≳ {int(2 * bins**3)} steps needed for occ > {occ_warn:.0%}.",
            UserWarning, stacklevel=2,
        )

    # the spatial info numbers
    sinfo = _spatial_info(p, f)
    sidx  = _sparsity_idx(p, f)

    sinfo_z, sidx_z = None, None
    if shuf_sums is not None:
        sinfo_z, sidx_z = _online_shuffle_zscores(
            shuf_sums, scored_idx, denom, p, sigma, sinfo, sidx)

    # the structure scores, one neuron at a time
    chi_out, mra_out, go_out, ring_found, no_ring, n_failed = \
        _stream_structure_scores(f, bins, align=align,
                                 global_order=global_order, go_precision=go_precision, n_jobs=n_jobs)

    out = dict(
        chi=chi_out, mra=mra_out,
        ring_found=ring_found, no_ring=no_ring, n_failed=n_failed,
        sinfo=sinfo, sidx=sidx,
        sinfo_z=sinfo_z, sidx_z=sidx_z,     # explicit None when no shuf_sums
        ifd=_ifd_for_scored(positions, activity, scored_idx, grid_spacing, env_size),
        peak=f.max(axis=(0, 1, 2)),
        occupancy=occ, reliable=reliable,
        scored_idx=scored_idx, bins=bins, aligned=align,
    )
    if global_order:
        out["go"] = go_out
    return out

def run_and_score_3d(
    exp,
    g_vec: np.ndarray,
    *,
    bins: int = 25,
    n_sub: int = 300,
    seed: int = 0,
    n_warmup: int = 100,
    sigma: float = SMOOTH_SIGMA,
    global_order: bool = False,
    n_shuffle: int = 0,
    n_jobs: int = 1, 
) -> dict:
    """
    One-shot: run_3d_online then score_3d_from_map.

    Returns a merged dict with all keys from both functions, plus:
        grid_like  (n_scored,) bool
        grid_like_method  str  — which metric built grid_like

    grid_like is built from global_order scores ('go') when global_order=True:
    the maximum hexagonal gridness over orientations. Otherwise it is a cell
    whose best Grieves difference (FCC, HCP, or columnar) is positive.
    Inter-field distances are computed from the strided activity trace.
    """
    raw    = run_3d_online(
        exp, g_vec, bins=bins, n_sub=n_sub, seed=seed,
        n_warmup=n_warmup, n_shuffle=n_shuffle,
    )
    scores = score_3d_from_map(
        raw["sums"], raw["counts"],
        shuf_sums=raw["shuf_sums"],
        sigma=sigma, global_order=global_order,
        active_mask=raw["active_mask"],
        n_jobs=n_jobs,
        positions=raw["field_pos"],
        activity=raw["field_act"],
        grid_spacing=exp.config.experiment.grid_spacing,
        env_size=exp.config.experiment.env_size,
    )

    if not scores or scores.get("chi") is None:
        return {**raw, **scores, "grid_like": np.zeros(0, bool),
                "grid_like_method": "none (no scored cells)"}

    # Build grid_like from the most reliable available metric
    if global_order and scores.get("go") is not None:
        go = scores["go"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            grid_like = np.isfinite(go) & (go > 0)
        grid_like_method = "global_order (orientation-invariant)"
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            chi_max = np.nanmax(scores["chi"], axis=0)
        grid_like = scores.get("ring_found", np.zeros(0, bool)) & (chi_max > 0)
        grid_like_method = (
            "grieves chi (a positive FCC, HCP, or columnar difference). "
            "Pass global_order=True for the maximum hexagonal gridness."
        )
        n_pos = grid_like.sum()
        print(
            f"[run_and_score_3d] grid_like ({n_pos} / {len(grid_like)} cells) "
            f"uses the Grieves differences. Pass global_order=True for the "
            f"maximum hexagonal gridness over orientations."
        )

    return {**raw, **scores, "grid_like": grid_like,
            "grid_like_method": grid_like_method}
    
def hgs_shift_null(shuf_sums, counts, scored_idx, sigma=SMOOTH_SIGMA):
    """Hexagonal grid score of shuffled maps."""
    #setup
    scored_idx = np.asarray(scored_idx, dtype=int)
    n_shifts = int(shuf_sums.shape[0])
    n_scored = int(len(scored_idx))
    hgs_shuf = np.full((n_shifts, n_scored), np.nan)
    
    #loop thrpugh each shuffle and build the rate map, autocorrelation, and hexagonal grid score.
    for j in range(n_shifts):
        f_j = _rate_map_from_accumulator(
            shuf_sums[j][..., scored_idx], counts, sigma=sigma)
        ac_j = autocorr2d(f_j)
        for k in range(n_scored):
            hgs_shuf[j, k], _ = hex_gridness_2d(ac_j[..., k])
        del f_j, ac_j
    return hgs_shuf


def score_2d_from_map(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    sigma: float = SMOOTH_SIGMA,
    active_mask: np.ndarray | None = None,
    shuf_sums: np.ndarray | None = None,
    occ_warn: float = 0.3,
    hgs_thresh: float | None = None,
    require_hgs_gt_sgs: bool = False,
) -> dict:
    """Score a 2-D rate map already built by run_with_online_ratemap.

    Parameters
    ----------
    shuf_sums : (n_shuffle, bins, bins, n_cells) or None
        Circularly shifted activity sums from the online run (result.shuf_sums).
        The last axis is the same cell set as `sums` (the full sheet, or the
        same subsample). Occupancy is the original `counts`: t → (t+L) mod T
        is a bijection, so visits per bin are unchanged. Shifted HGS is pooled
        across cells and lags; because lags are shared, that pool is not
        n_cells × n_lags independent draws. An error bar on that floor would
        resample lags, not values.
    hgs_thresh : float or None
        If None, `grid_like` is left all-False. Apply a floor afterwards with
        `apply_hgs_floor` so several arms can share one null.
    """
    bins    = sums.shape[0]
    n_total = sums.shape[-1]
    if shuf_sums is not None and shuf_sums.shape[-1] != n_total:
        raise ValueError(
            f"shuffle cell axis {shuf_sums.shape[-1]} != rate-map cell axis {n_total}"
        )

    #Find neurons to score
    scored_idx = (np.where(active_mask)[0] if active_mask is not None
                  else np.arange(n_total))

    #Build the rate map
    f     = _rate_map_from_accumulator(sums[..., scored_idx], counts, sigma=sigma)
    #occupancy of the rate map
    p     = counts / (counts.sum() + 1e-30)
    occ   = float((p > 0).mean())

    if occ < occ_warn:
        warnings.warn(
            f"score_2d_from_map: occupancy {occ:.1%} — map is sparse. "
            f"HGS/ring results may reflect coverage rather than tuning.",
            UserWarning, stacklevel=2,
        )

    ac       = autocorr2d(f)          # (2*bins-1, 2*bins-1, n_scored)
    n_scored = len(scored_idx)
    
    hgs      = np.full(n_scored, np.nan)
    sgs      = np.full(n_scored, np.nan)
    for k in range(n_scored):
        hgs[k], sgs[k] = hex_gridness_2d(ac[..., k])

    ring_found = np.isfinite(hgs)
    #check if we actually get hexagonal pattern
    if hgs_thresh is None:
        grid_like = np.zeros(n_scored, dtype=bool)
    else:
        grid_like = ring_found & (hgs > hgs_thresh)
        if require_hgs_gt_sgs:
            grid_like = grid_like & (hgs > sgs)

    sinfo = _spatial_info(p, f)
    sidx  = _sparsity_idx(p, f)

    sinfo_z, sidx_z = None, None
    hgs_shuf = None
    hgs_shuf_p95 = None
    hgs_shuf_ring_frac = None
    
    # Score already-shifted maps, take the 95th percentile of HGS as the chance.
    if shuf_sums is not None:
        hgs_shuf = hgs_shift_null(shuf_sums, counts, scored_idx, sigma=sigma)
        finite = np.isfinite(hgs_shuf)  # True where a ring was found
        hgs_shuf_p95 = (
            float(np.nanpercentile(hgs_shuf, 95)) if finite.any()
            else float("nan")
        )  # chance floor
        hgs_shuf_ring_frac = float(finite.mean())  # how often a ring appeared

    return dict(
        #NOTE: not everything nessesarry probably
        f          = f,
        ac         = ac,
        p          = p,
        hgs        = hgs,
        sgs        = sgs,
        grid_like  = grid_like,
        ring_frac  = float(ring_found.mean()),
        sinfo      = sinfo,
        sidx       = sidx,
        sinfo_z    = sinfo_z,
        sidx_z     = sidx_z,
        hgs_shuf   = hgs_shuf,
        hgs_shuf_p95 = hgs_shuf_p95,
        hgs_shuf_ring_frac = hgs_shuf_ring_frac,
        peak       = f.max(axis=(0, 1)),
        occupancy  = occ,
        n_active   = n_scored,
        scored_idx = scored_idx,
        bins       = bins,
    )
    
def run_and_score_2d(
    exp,
    g_vec: np.ndarray,
    *,
    bins: int = 40,
    n_warmup: int = 100,
    sigma: float = SMOOTH_SIGMA,
    trajectory=None,
    refresh_mask=None,
) -> dict:
    """One-shot: run_with_online_ratemap then score_2d_from_map.

    The 2-D analogue of run_and_score_3d. Useful as a fast hexagonality
    sanity check on a flat-floor run without needing the full 3-D pipeline.
    """
    raw    = run_with_online_ratemap(exp, g_vec, bins=bins,
                                     n_warmup=n_warmup, trajectory=trajectory,
                                     refresh_mask=refresh_mask)
    scores = score_2d_from_map(raw["sums"], raw["counts"], sigma=sigma)
    return {**raw, **scores}


def check_lattice(sheet):
    """Is the settled (n, n) sheet a clean hexagon? passed = triad closes AND HGS > SGS.

    Triad: the three strongest wavevectors sum to exactly zero (they are integers;
    a residual of one bin is a defect). It misses squares, because rectification
    adds a k1 - k2 harmonic that closes the sum, so HGS > SGS is gated too.
    Field spacing = (2/sqrt(3)) * n / mean|k| of the three modes, in cells.
    """
    sheet = np.asarray(sheet, dtype=float)
    n = sheet.shape[0]
    power = np.abs(np.fft.fft2(sheet)) ** 2
    power[0, 0] = 0.0
    peak = (power == maximum_filter(power, size=3, mode="wrap")) & (power > 0)
    kk = np.where(np.arange(n) <= n // 2, np.arange(n), np.arange(n) - n)
    iy, ix = np.where(peak)
    k = np.stack([kk[iy], kk[ix]], axis=1).astype(float)
    half = (k[:, 0] > 0) | ((k[:, 0] == 0) & (k[:, 1] > 0))   # real sheet: F(k) = F(-k)*
    k3 = k[half][np.argsort(power[iy, ix][half])[::-1][:3]]

    triad = len(k3) == 3 and any(
        np.linalg.norm(k3[0] + a * k3[1] + b * k3[2]) < 0.5
        for a in (1, -1) for b in (1, -1))
    hgs, sgs = (float(v) for v in hex_gridness_2d(autocorr2d(sheet)[..., 0]))
    spacing = (2 / np.sqrt(3)) * n / np.linalg.norm(k3, axis=1).mean() if len(k3) else np.nan
    passed = bool(triad and hgs > sgs)
    return dict(
        passed=passed, triad_closes=bool(triad), hgs=hgs, sgs=sgs,
        modes=k3.tolist(), field_spacing_cells=float(spacing),
        summary=(f"lattice {'PASS' if passed else 'FAIL'}  triad={'yes' if triad else 'no'}  "
                 f"HGS={hgs:.3f} SGS={sgs:.3f}  field_spacing={spacing:.2f} cells"))


def tracking_score(decoded, truth):
    """Path-integration score: mean unwrap error / mean unwrap path length.

    Dimensionless. 0 is perfect; 1 means the error is as large as the walk.
    """
    decoded = np.asarray(decoded, dtype=float)
    truth = np.asarray(truth, dtype=float)
    dec_path = unwrap_torus(decoded)
    gt_path = unwrap_torus(truth)
    error = np.linalg.norm(dec_path - gt_path, axis=1)
    gt_step = wrapped_angle_diff(truth[1:], truth[:-1], period=2 * np.pi)
    path_length = np.concatenate(
        [[0.0], np.cumsum(np.linalg.norm(gt_step, axis=1))]
    )
    return float(error.mean() / (path_length.mean() + 1e-12))
