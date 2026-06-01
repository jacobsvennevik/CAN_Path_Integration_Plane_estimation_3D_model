import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


from scipy.ndimage import gaussian_filter


def standardize_ratemaps(sums, counts):
    """Turns the raw binned arrays into a rate map.
    """
    sums = np.asarray(sums, dtype=float)
    ratemap_counts = np.asarray(counts, dtype=float) #occupancy witch bins have been visited

    visited = ratemap_counts > 0
    rate = np.full_like(sums, np.nan)
    divisor = np.where(visited, ratemap_counts, 1.0)[..., None] 
    rate[visited] = (sums / divisor)[visited]
    return rate, ratemap_counts


def pick_active_cells(rate_maps, k=6):
    """Find the k cells that are the most active (highest peak rate)"""
    R = np.asarray(rate_maps)
    peak = np.array([np.nanmax(R[..., n]) for n in range(R.shape[-1])])
    peak = np.where(np.isfinite(peak), peak, -np.inf)
    return np.argsort(peak)[::-1][:k]

def rescale_positions_to_unit_cube(world_pos, env_size=None, isotropic=True):
    """Map world positions to [-1, 1] per axiss.

    """
    p = np.asarray(world_pos, dtype=float)
    xy = p[:, :2] if p.shape[1] >= 2 else p
    if env_size is not None and not isotropic:
        return xy / (env_size / 2.0)
    m = np.nanmax(np.abs(xy))
    return xy / (m if m > 0 else 1.0)


def _world_to_bin(xy_unit, bins):
    """maps world [−1, 1] positions to integer bin indices."""
    ij = np.floor((xy_unit + 1.0) / 2.0 * bins).astype(int)
    return np.clip(ij, 0, bins - 1)


def mark_high_activity(activity_trace, xy_unit, top_pct=20.0):
    """marks the positions where activity is high"""
    a = np.asarray(activity_trace, dtype=float)
    xy = np.asarray(xy_unit, dtype=float)
    thr = np.nanpercentile(a, 100.0 - top_pct)
    return xy[a >= thr]







