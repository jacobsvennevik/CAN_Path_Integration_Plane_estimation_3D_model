"""
All of developed grid-field scoring code in one module, inheritance from G and Y in that file. 
"""

from dataclasses import dataclass
import numpy as np, torch

from config import ExperimentConfig

from . import gongyu_scoring as gy
from scipy.spatial.distance import pdist
from scipy.ndimage import gaussian_filter


# Cube all scorers assume. Matches Gong & Yu [-1,1]^3.
LIM = ((-1, 1), (-1, 1), (-1, 1))

# Raw .npz keys written by the harness when record=True.
RAW_POS_KEY = "world_pos"
RAW_ACT_KEY = "S_tot_buffer"

BINS = ExperimentConfig.ratemap_bins
AUTOCORR_TH = 0.1
SMOOTH_SIGMA = 1.75


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

def load_npz(npz_path: str) -> dict:
    """Read an .npz run file and return all arrays as a dict."""
    d = np.load(npz_path, mmap_mode="r")
    return {k: np.asarray(d[k]) for k in d.files}


def _normalise_activity(S: np.ndarray, mode: str = "divmax"):
    """Maps positive activity into [0, 1] per neuron.
    "divmax" preserves this is used in analysis following Skaggs spatial information.
    "minmax" stretches the full dynamic range
    """
    mn, mx = S.min(0), S.max(0)
    if mode == "minmax":
        a = (S - mn) / np.where(mx > mn, mx - mn, 1.0)
    elif mode == "divmax":
        a = S / np.where(mx > 0, mx, 1.0)
    else:
        raise ValueError(f"unknown norm mode {mode!r}; use 'divmax' or 'minmax'")
    a[np.isnan(a)] = 0.0
    np.clip(a, 0.0, 1.0, out=a)   # guard tiny float overshoot for validate()
    return a, mn, mx


def scoring_input_from_npz(npz_path: str, n_neuron: int = 20000,
                           seed: int = 0, norm: str = "divmax",
                           active_thresh: float = 1e-3) -> ScoringInput:
    """Runs the score tunrs a saved .npz file into something the scoring pipeline can run on. This is done by:
    1. Loading
    2. Rescaling positions
    3. Normalising activity pr neuron
    4. Filter only active neruons and then sumbsamples then

    Parameters
    ----------
    npz_path      : path to the saved run
    n_neuron      : how many active neurons to keep (cap; all taken if fewer exist)
    seed          : RNG seed for reproducible subsampling
    norm          : 'divmax' (default, preserves zero) or 'minmax' (see
                    `_normalise_activity`)
    active_thresh : a neuron counts as active if (max-min) > active_thresh * max(max)
    """
    d         = load_npz(npz_path)
    world_pos = d[RAW_POS_KEY]     # (T, 3)
    S         = d[RAW_ACT_KEY]     # (T, N), non-negative

    mn = S.min(0)    
    mx = S.max(0)
    
    #1. Subsample from acitve neurons only
    rng    = np.random.default_rng(seed)
    active = (mx - mn) > active_thresh * mx.max()
    active_idx = np.where(active)[0]
    n_take = min(n_neuron, len(active_idx))
    idx    = np.sort(rng.choice(active_idx, size=n_take, replace=False))  
    
    #2. Normalise activity to [0, 1] per neuron
    S_sub = np.asarray(S[:, idx])
    a, _, _ = _normalise_activity(S_sub, mode=norm)
    #2. Rescale positions to within [-1, 1]^3, preserving geometry
    lo, hi = world_pos.min(0), world_pos.max(0)
    center = (hi + lo) / 2.0
    half   = np.max((hi - lo) / 2.0)
    half   = half if half > 0 else 1.0
    x = (world_pos - center) / half


    return ScoringInput(
        x=x, a=a, cell_idx=idx,
        meta=dict(npz_path=npz_path, n_neuron=int(n_take), seed=int(seed),
                  norm=norm, n_active=int(active.sum()), n_total=int(S.shape[1])),
    )



def activity_rate_map(si: ScoringInput, dims=(0, 1),
                      bins: int = BINS,
                      sigma: float = SMOOTH_SIGMA) -> np.ndarray:
    """Mean activity per spatial bin.

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


def autocorr2d(f: np.ndarray, th: float = AUTOCORR_TH) -> np.ndarray:
    """Standardised 2-D autocorrelation.

    Parameters
    ----------
    f  : (bins, bins, n)  smoothed rate map
    th : values below this threshold are zeroed in the output

    Returns
    -------
    ac : (2*bins-1, 2*bins-1, n), non-negative
    """
    from scipy.signal import correlate
    if f.ndim == 2:
        f = f[..., None]
    n   = f.shape[0] * f.shape[1]
    std = f.std(axis=(0, 1))
    std = np.where(std < 1e-10, 1.0, std)   # guard: zero-std → skip normalisation
    f_  = (f - f.mean(axis=(0, 1))) / std
    out = []
    for i in range(f.shape[-1]):
        ac = correlate(f_[..., i], f_[..., i], mode="full") / n
        ac[ac < th] = 0.0
        out.append(ac)
    return np.stack(out, axis=-1)


def hex_gridness_2d(ac2d: np.ndarray, peak_thresh: float = 0.1) -> tuple:
    """(hgs, sgs) hexagonal/square gridness for one 2-D autocorrelogram.

    Finds the inner-ring radius directly from the 2-D autocorrelation peaks rather
    than from the radial-mean profile: the radial *mean* averages a real 6-peak
    ring together with its 6 troughs into a smooth monotonic decay, so
    `gy.peak` misses the ring and the score collapses to NaN even when the ring is
    plainly visible. Locating the peaks in 2-D avoids that failure mode.

    Returns (nan, nan) when no ring is found (genuinely no hexagonal structure).
    """
    from scipy.ndimage import maximum_filter
    d = ac2d.shape[0]
    c = d // 2
    yy, xx = np.mgrid[-c:c + 1, -c:c + 1] if d % 2 else np.mgrid[-c:c, -c:c]
    r = np.sqrt(xx ** 2 + yy ** 2)
    is_peak = (ac2d == maximum_filter(ac2d, size=3)) & (ac2d > peak_thresh) & (r > 2)
    radii = r[is_peak]
    if radii.size < 3:
        return np.nan, np.nan                       # no ring -> not a grid
    ring_r = np.median(np.sort(radii)[:6])          # six nearest peaks = inner ring
    lb, ub = int(round(0.5 * ring_r)), min(int(round(1.5 * ring_r)), c)
    if ub <= lb:
        return np.nan, np.nan
    return gy.gridness(ac2d, lb, ub)

def autocorrelation_1cell(f_cell: np.ndarray, th: float = AUTOCORR_TH) -> np.ndarray:
    """Standardised 3-D autocorrelation of a single cell's rate map.

    Parameters
    ----------
    f_cell : (b, b, b) one neuron's smoothed rate map

    Returns
    -------
    ac : (2b-1, 2b-1, 2b-1) non-negative autocorrelogram for that cell.
    """
    from scipy.signal import correlate
    nbins = f_cell.size
    std = f_cell.std()
    std = std if std > 1e-10 else 1.0
    f_ = (f_cell - f_cell.mean()) / std
    ac = correlate(f_, f_, mode="full") / nbins
    ac[ac < th] = 0.0
    return ac


# =============================================================================
# 4. STRUCTURE  (chi + MRA)
# =============================================================================

def chi_1cell(ac_cell: np.ndarray, align: bool = False) -> np.ndarray:
    """(3,) chi scores (fcc, hcp, col) for one cell's 3-D autocorrelogram.

    Parameters
    ----------
    ac_cell : (d, d, d) non-negative autocorrelation cube for one cell.
    align   : if True, find the best hexagonal plane (gy.best_plane) and rotate the
              cube before scoring.

    """
    cube = ac_cell[..., None]                  # (d, d, d, 1) as chi_score expects
    if align:
        try:
            az, al, _ = gy.best_plane(cube)
            cube = _rotate_cube_to_horizontal(cube, az[0], al[0])
            cube[cube < 0] = 0.0               # rotation interpolation can dip <0
        except (AssertionError, ValueError):
            return np.full(3, np.nan)
    try:
        fcc, hcp, col = gy.chi_score(cube)
        return np.array([fcc[0], hcp[0], col[0]])
    except AssertionError:
        return np.full(3, np.nan)              # peak() found no clean ring


def _rotate_cube_to_horizontal(cube: np.ndarray, az: float, al: float) -> np.ndarray:
    """Resample an autocorrelation cube so the (az, al) plane becomes horizontal.

    Uses the same rotation convention as gy.rot_x / gy.oblique_slice, so the
    aligned cube is consistent with chi_score's fixed plane indices.
    """
    from scipy.interpolate import RegularGridInterpolator
    d = cube.shape[0]
    grid = np.linspace(-1, 1, d)
    interp = RegularGridInterpolator((grid, grid, grid), cube[..., 0],
                                     bounds_error=False, fill_value=0.0)
    gx, gy_, gz = np.meshgrid(grid, grid, grid, indexing="ij")
    pts = np.stack([gx.ravel(), gy_.ravel(), gz.ravel()], axis=-1)
    # inverse rotation: sample the original cube at rotated coordinates
    pts_rot = gy.rot_x(pts, azimuth=-az, altitude=-al)
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
                                 radial_method=radial_method)
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


def inter_field_distances(si: ScoringInput, activity_pct: float = 90.0,
                          bandwidth: float = 0.25, min_bin_freq: int = 25,
                          min_cluster_size: int = 30,
                          ignore_range: float = 0.95) -> dict:
    """Find the distance between two firing fields.
    """
    out = {}
    for k in range(si.n):
        threshold  = np.percentile(si.a[:, k], activity_pct)
        active_pos = si.x[si.a[:, k] >= threshold]
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



def score_2d(npz_path: str, n_neuron: int = 5000, seed: int = 0,
             bins: int = BINS, sigma: float = SMOOTH_SIGMA,
             norm: str = "divmax") -> dict:
    """End-to-end 2-D baseline: project to xy, score hexagonality.

    Sucess: autocorrelation shows a 6-fold ring for most of the included neurons.
    
    NOT IN USE ANYMORE
    """
    si = scoring_input_from_npz(npz_path, n_neuron=n_neuron, seed=seed, norm=norm)
    assert si.x[:, 2].std() < 1e-3, \
        f"score_2d expects a flat-floor run; z std = {si.x[:, 2].std():.3g}"

    f  = activity_rate_map(si, dims=(0, 1), bins=bins, sigma=sigma)  # (b,b,n)
    ac = autocorr2d(f)                                                # (2b-1,2b-1,n)
    p  = occupancy(si, dims=(0, 1), bins=bins)                        # (b,b)

    # per-cell hexagonal gridness: the actual baseline pass/fail number, not just
    # an autocorrelogram to eyeball. SGS should sit low for a real hexagonal grid.
    hgs = np.full(si.n, np.nan)
    sgs = np.full(si.n, np.nan)
    for k in range(si.n):
        hgs[k], sgs[k] = hex_gridness_2d(ac[..., k])

    return dict(
        si        = si,
        f         = f,
        ac        = ac,
        p         = p,
        hgs       = hgs,                    # hexagonal gridness per cell (grid if > ~0.3)
        sgs       = sgs,                    # square gridness (should be low for hex)
        ring_frac = float(np.isfinite(hgs).mean()),   # fraction with a detectable ring
        peak      = f.max(axis=(0, 1)),     # peak activity per neuron
        occupancy = float((p > 0).mean()),
    )


def score_run(npz_path: str, n_neuron: int = 300, seed: int = 0,
              bins: int = BINS, sigma: float = SMOOTH_SIGMA,
              align: bool = False, norm: str = "divmax",
              global_order: bool = False, go_precision: tuple = (48, 24),
              occ_warn: float = 0.5) -> dict:
    """Full 3-D scoring pipeline (memory-safe, streams one cell at a time).

    Returns chi structure scores, MRA, spatial information, sparsity, and
    inter-field distances per neuron.
    """
    si  = scoring_input_from_npz(npz_path, n_neuron=n_neuron, seed=seed, norm=norm)
    f   = activity_rate_map(si, dims=(0, 1, 2), bins=bins, sigma=sigma)  # (b,b,b,n)
    p   = occupancy(si, dims=(0, 1, 2), bins=bins)                       # (b,b,b)
    occ = float((p > 0).mean())

    if occ < occ_warn:
        import warnings
        warnings.warn(
            f"occupancy {occ:.1%}: map is sparse — chi/MRA reflect coverage "
            "more than tuning. Lengthen the trajectory."
        )
    if align:
        import warnings
        warnings.warn(
            "align=True is miscalibrated against chi_score's fixed template and "
            "can collapse a cleanly-ordered cell to ~0 (see chi_1cell). Prefer "
            "align=False + global_order=True for an orientation-invariant readout."
        )
    if n_neuron > 5000:
        import warnings
        warnings.warn(
            f"n_neuron > 5000: scoring this many cells will be slow."
        )

    # --- structure scores: stream one cell at a time, never hold the full cube
    rmax = (2 * bins - 1) // 2
    chi_out = np.full((3, si.n), np.nan)
    mra_out = np.zeros((si.n, rmax))
    go_out  = np.full(si.n, np.nan) if global_order else None
    for k in range(si.n):
        ac_k = autocorrelation_1cell(f[..., k])     # (2b-1, 2b-1, 2b-1)
        chi_out[:, k] = chi_1cell(ac_k, align=align)
        mra_out[k]    = mra_1cell(ac_k)
        if global_order:
            go_out[k] = global_order_1cell(ac_k, az_precision=go_precision[0],
                                           al_precision=go_precision[1])
        del ac_k                                     # free before next cell

    # distinguish "no ring detected" (chi forced to exactly 0) from a measured-low
    # gridness and from an outright failure (NaN). chi-at-0 is NOT the same evidence
    # as a measured-low score — see the H1 caveat in the review.
    chi_max  = np.nanmax(chi_out, axis=0)
    ring_found = np.isfinite(chi_max) & (np.abs(chi_max) > 1e-9)
    no_ring  = np.isfinite(chi_max) & (np.abs(chi_max) <= 1e-9)

    info = info_scores(si, p, f, bins=bins, sigma=sigma, seed=seed)
    ifd  = inter_field_distances(si)

    out = dict(
        idx        = si.cell_idx,
        chi        = chi_out,
        mra        = mra_out,
        ring_found = ring_found,                 # True = a ring was detected
        no_ring    = no_ring,                    # True = chi==0 because no ring (not "measured low")
        n_failed   = int(np.isnan(chi_max).sum()),
        **info,
        ifd        = ifd,
        peak       = f.max(axis=(0, 1, 2)),
        occupancy  = occ,
        aligned    = align,
    )
    if global_order:
        out["go"] = go_out                       # orientation-invariant order score
    return out

def run_with_online_ratemap(exp, g, bins=40, n_warmup=100, mode="integrate"):
    """
    Mmeory light run, so that the hughe buffers is not used and stride might not be needed.
    """
    integ, bk, cfg = exp.integrator, exp.integrator.backend, exp.config.experiment
    dev, half = bk.device, cfg.env_size / 2.0

    world_pos, v_body_seq, torus_gt = exp.generate_trajectory() 
    T, N = world_pos.shape[0], bk.S.shape[1]

    integ.reset(torus_gt[0])                       # bump at start + fresh filter + clear history
    if mode == "integrate" and n_warmup:
        integ.warmup(n_warmup)

    # flat spatial-bin index for every step
    xy   = world_pos[:, :2] / half                 # -> ~[-1, 1]
    ij   = np.clip(np.floor((xy + 1.0) * 0.5 * bins).astype(np.int64), 0, bins - 1)
    flat = (ij[:, 0] * bins + ij[:, 1]).tolist()

    sums   = torch.zeros((bins * bins, N), dtype=torch.float32, device=dev)   # the accumulator
    counts = torch.zeros(bins * bins,      dtype=torch.float32, device=dev)

    theta_hist = np.zeros((T, 3), dtype=np.float32)
    for t in range(T):
        if mode == "integrate":
            theta_hist[t] = integ.step(v_body_seq[t], g)     # filter + drive + decode (real path)
        else:                                                # oracle
            bk.reset(torus_gt[t]); theta_hist[t] = torus_gt[t]
        s = bk.S.mean(dim=0).squeeze()                       # (N,) on device — no per-step CPU copy
        b = flat[t]
        sums[b] += s
        counts[b] += 1.0

    return dict(                                             # one CPU transfer, at the end
        sums   = sums.cpu().numpy().reshape(bins, bins, N),
        counts = counts.cpu().numpy().reshape(bins, bins),
        bins=bins, world_pos=world_pos, torus_gt=torus_gt, theta_hist=theta_hist,
    )


def score_2d_from_map(sums, counts, sigma=1.75, n_neuron=5000, seed=0, active_thresh=1e-3):
    """Score an accumulated rate map. Same autocorr + gridness as score_2d, no .npz needed."""
    from analysis.scoring import autocorr2d, hex_gridness_2d
    N = sums.shape[-1]

    # drop dead/flat neurons from the summed map, then subsample (mirrors score_2d)
    flat = sums.reshape(-1, N)
    span = flat.max(0) - flat.min(0)
    active_idx = np.where(span > active_thresh * (span.max() + 1e-12))[0]
    take = min(n_neuron, len(active_idx))
    idx  = np.sort(np.random.default_rng(seed).choice(active_idx, size=take, replace=False)) \
           if take else np.array([], int)

    denom = np.where(counts > 0, counts, 1.0)[..., None]
    f = sums[..., idx] / denom                               # (bins, bins, take) — small after subsample
    for c in range(f.shape[-1]):
        f[..., c] = gaussian_filter(f[..., c], sigma=sigma)

    ac  = autocorr2d(f)                                      # gridness is scale-invariant (it z-scores),
    hgs = np.full(f.shape[-1], np.nan)                       # so no per-neuron divmax needed here
    sgs = np.full(f.shape[-1], np.nan)
    for k in range(f.shape[-1]):
        hgs[k], sgs[k] = hex_gridness_2d(ac[..., k])

    grid_like = np.isfinite(hgs) & (hgs > 0.3) & (sgs < hgs)
    return dict(f=f, ac=ac, hgs=hgs, sgs=sgs, idx=idx,
                ring_frac=float(np.isfinite(hgs).mean()) if len(hgs) else 0.0,
                grid_like=grid_like, occupancy=float((counts > 0).mean()),
                n_active=int(len(active_idx)))