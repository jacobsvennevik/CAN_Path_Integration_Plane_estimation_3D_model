"""Rate maps, occupancy, and inter-field distances."""
import warnings
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform

from config import world_to_normalized, world_to_flat_bins, AnalysisConfig
from . import gongyu_scoring as gy

SMOOTH_SIGMA = AnalysisConfig.smooth_sigma

def visited_bins(world_pos, env_size, bins, ndim=2):
    """How many rate-map bins the trajectory enters, and how many bins the grid has."""
    flat = world_to_flat_bins(world_pos, env_size, bins, ndim=ndim)
    counts = np.bincount(flat, minlength=int(bins) ** int(ndim))
    return int((counts > 0).sum()), int(counts.size)


def occupancy_fraction(world_pos, env_size, bins, ndim=2):
    """Fraction of bins visited by a trajectory, from the same binning as the rate maps."""
    visited, total = visited_bins(world_pos, env_size, bins, ndim=ndim)
    return float(visited / total) if total else float("nan")


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


def local_spacing_cv(centers: np.ndarray, n_neighbours: int = 3) -> float:
    """Spread of each field's nearest-neighbour distances, as s.d. / mean.

    uses the distances from each field to its three nearest neighbours, and a low value means those
    distances agree. A cell needs at least four fields, so that three
    neighbours exist. An edge is counted from both ends when both fields list
    each other.
    """
    centers = np.asarray(centers, dtype=float)
    k = int(n_neighbours)
    if centers.ndim != 2 or len(centers) < k + 1:
        return float("nan")
    dist = squareform(pdist(centers))
    np.fill_diagonal(dist, np.inf)
    nearest = np.partition(dist, k - 1, axis=1)[:, :k]
    mean = float(nearest.mean())
    if mean <= 0.0:
        return float("nan")
    return float(nearest.std(ddof=1) / mean)


def _field_centers(activity_pct: float = 90.0,
                   bandwidth: float | None = None, min_bin_freq: int = 25,
                   min_cluster_size: int = 30,
                   ignore_range: float | None = None,
                   positions: np.ndarray | None = None,
                   activity: np.ndarray | None = None,
                   grid_spacing: float | None = None,
                   env_size: float | None = None,
                   world_coords: bool = False) -> dict:
    """MeanShift field centres for each cell, in the normalised cube.

    The points clustered are the positions where that cell's activity was at
    or above its 90th percentile. Rate maps do not use this cut. A centre is
    the mean of one surviving cluster.
    """
    if positions is None or activity is None:
        raise ValueError("field centres need positions and activity")
    if grid_spacing is None or env_size is None:
        raise ValueError("field centres need grid_spacing and env_size")
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
    empty = np.empty((0, positions.shape[1]))
    for k in range(n):
        threshold = np.percentile(activity[:, k], activity_pct)
        active_pos = positions[activity[:, k] >= threshold]
        if len(active_pos) < min_bin_freq:
            out[k] = empty
            continue
        try:
            _, _, centers = gy.ms_cluster(
                active_pos, bandwidth=bandwidth, min_bin_freq=min_bin_freq,
                min_cluster_size=min_cluster_size, ignore_range=ignore_range,
                plot=False)
        except ValueError:
            out[k] = empty
            continue
        out[k] = centers
    return out


def inter_field_distances(activity_pct: float = 90.0,
                          bandwidth: float | None = None, min_bin_freq: int = 25,
                          min_cluster_size: int = 30,
                          ignore_range: float | None = None,
                          positions: np.ndarray | None = None,
                          activity: np.ndarray | None = None,
                          grid_spacing: float | None = None,
                          env_size: float | None = None,
                          world_coords: bool = False) -> dict:
    """Finds the centers of the fields for each cell. Replace them with distance between the fields.
    """
    centers_of = _field_centers(
        activity_pct=activity_pct, bandwidth=bandwidth, min_bin_freq=min_bin_freq,
        min_cluster_size=min_cluster_size, ignore_range=ignore_range,
        positions=positions, activity=activity, grid_spacing=grid_spacing,
        env_size=env_size, world_coords=world_coords,
    )
    out = {}
    for k, centers in centers_of.items():
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


def _ifd_for_scored(positions, activity, scored_idx, grid_spacing, env_size):
    """Score (find the IFD) the distances between the fields for each cell.
    """
    if positions is None or activity is None:
        return None
    activity = np.asarray(activity)
    if activity.ndim != 2 or activity.shape[1] <= int(np.max(scored_idx, initial=-1)):
        raise ValueError(
            f"activity cell axis {getattr(activity, 'shape', None)} does not cover "
            f"scored index {int(np.max(scored_idx))}"
        )
    centers_of = _field_centers(
        positions=np.asarray(positions),
        activity=activity[:, scored_idx],
        grid_spacing=grid_spacing,
        env_size=env_size,
        world_coords=True,
    )
    n = activity[:, scored_idx].shape[1]
    ifd = {}
    n_fields = np.zeros(n, dtype=int)
    ifd_cv = np.full(n, np.nan)
    #loop thorugh each cell and find the scores
    for k, centers in centers_of.items():
        n_fields[k] = len(centers)
        #find the distance between the fields for each cell
        ifd[k] = pdist(centers) if len(centers) >= 2 else np.array([])
        #find the nearest-neighbour CV for each cell
        ifd_cv[k] = local_spacing_cv(centers)
    return dict(ifd=ifd, n_fields=n_fields, ifd_cv=ifd_cv)
