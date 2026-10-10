"""2-D autocorrelation and hexagonal gridness."""
import warnings
import numpy as np
from scipy.ndimage import label, maximum_filter
from scipy.signal import correlate

from config import AnalysisConfig
from . import gongyu_scoring as gy
from .rate_maps import SMOOTH_SIGMA, _rate_map_from_accumulator, _spatial_info, _sparsity_idx
from .structure3d import _online_shuffle_zscores

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


def _shift_info_z(shuf_sums, counts, scored_idx, p, sigma, sinfo, sidx):
    """_shift_info_z turns the time-shifted maps into z-scores for spatial information and sparsity. 
    Comparing the real scores (sinfo, sidx) with the same scores on the shuffles (shuf_sums).
    """
    scored_idx = np.asarray(scored_idx, dtype=int)
    
    #check atleast one cell is scored and index checkk
    per_cell = scored_idx.size and int(scored_idx.max()) < shuf_sums.shape[-1]
    if per_cell:
        #compuute z-scores for for spatial information and sparsity
        return _online_shuffle_zscores(
            shuf_sums, scored_idx, counts, p, sigma, sinfo, sidx)
        
    #if not, shuffled sets are pooled
    n_shifts = shuf_sums.shape[0]
    sinfo_n = np.empty((n_shifts, shuf_sums.shape[-1]), dtype=np.float64)
    sidx_n = np.empty_like(sinfo_n)
    
    #For each shift, the accumulated sums are turned into a smoothed rate
    for j in range(n_shifts):
        f_j = _rate_map_from_accumulator(shuf_sums[j], counts, sigma=sigma)
        sinfo_n[j] = _spatial_info(p, f_j)
        sidx_n[j] = _sparsity_idx(p, f_j)
    return (
        (sinfo - np.nanmean(sinfo_n)) / (np.nanstd(sinfo_n) + 1e-9),
        (sidx - np.nanmean(sidx_n)) / (np.nanstd(sidx_n) + 1e-9),
    )


def hgs_shift_null(shuf_sums, counts, scored_idx=None, sigma=SMOOTH_SIGMA):
    """Hexagonal grid score of shuffled maps. Returns HGS only.

    ``scored_idx`` selects columns when the shuffle has one column per cell
    of the real map. A separate null set (its own columns) is scored whole.
    """
    #setup
    if scored_idx is not None:
        scored_idx = np.asarray(scored_idx, dtype=int)
    fits = (scored_idx is not None and scored_idx.size
            and int(scored_idx.max()) < shuf_sums.shape[-1])
    if fits:
        block_idx = scored_idx
        n_scored = int(len(block_idx))
    else:
        block_idx = None
        n_scored = int(shuf_sums.shape[-1])
    n_shifts = int(shuf_sums.shape[0])
    hgs_shuf = np.full((n_shifts, n_scored), np.nan)

    #loop thrpugh each shuffle and build the rate map, autocorrelation, and hexagonal grid score.
    for j in range(n_shifts):
        piece = shuf_sums[j] if block_idx is None else shuf_sums[j][..., block_idx]
        f_j = _rate_map_from_accumulator(piece, counts, sigma=sigma)
        ac_j = autocorr2d(f_j)
        for k in range(n_scored):
            hgs_shuf[j, k], _ = hex_gridness_2d(ac_j[..., k])
        del f_j, ac_j
    return hgs_shuf


def apply_hgs_floor(scores: dict, floor: float,
                    require_hgs_gt_sgs: bool = False) -> dict:
    """Mark grid_like against one HGS floor. Mutates and returns `scores`."""
    #check if we actually get hexagonal pattern
    hgs = np.asarray(scores["hgs"], dtype=float)
    sgs = np.asarray(scores["sgs"], dtype=float)
    # finite HGS above the shared chance floor
    like = np.isfinite(hgs) & (hgs > floor)
    if require_hgs_gt_sgs:
        like = like & (hgs > sgs)
    scores["grid_like"] = like
    return scores


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


def score_2d_from_map(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    sigma: float = SMOOTH_SIGMA,
    active_mask: np.ndarray | None = None,
    shuf_sums: np.ndarray | None = None,
    occ_warn: float = 0.3,
) -> dict:
    """Score a 2-D rate map already built by run_with_online_ratemap.


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
        sinfo_z, sidx_z = _shift_info_z(
            shuf_sums, counts, scored_idx, p, sigma, sinfo, sidx)

    return dict(
        #NOTE: not everything nessesarry probably
        f          = f,
        ac         = ac,
        hgs        = hgs,
        sgs        = sgs,
        ring_frac  = float(ring_found.mean()),
        sinfo      = sinfo,
        sidx       = sidx,
        sinfo_z    = sinfo_z,
        sidx_z     = sidx_z,
        hgs_shuf   = hgs_shuf,
        hgs_shuf_p95 = hgs_shuf_p95,
        hgs_shuf_ring_frac = hgs_shuf_ring_frac,
        occupancy  = occ,
        n_active   = n_scored,
        scored_idx = scored_idx,
        bins       = bins,
    )
