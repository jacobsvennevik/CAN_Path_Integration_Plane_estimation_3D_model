"""3-D structure scores (Grieves χ) and shuffle z-scores."""
import warnings
import numpy as np

from . import gongyu_scoring as gy
from .rate_maps import (
    SMOOTH_SIGMA, _rate_map_from_accumulator, _spatial_info, _sparsity_idx,
    _ifd_for_scored,
)

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
    from .gridness2d import hex_gridness_2d

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


def _score_one_cell(f_k):
    """Score one neuron's rate map: Grieves χ and MRA."""
    ac_k = autocorrelation_1cell(f_k)
    return grieves_chi_1cell(ac_k), mra_1cell(ac_k)


def _stream_structure_scores(f: np.ndarray, bins: int, n_jobs: int = 1) -> tuple:
    """χ and MRA, one neuron at a time.

    Returns chi (3, n), mra (n, rmax), ring_found (n,).
    """
    n = f.shape[-1]
    rmax = (2 * bins - 1) // 2
    chi_out = np.full((3, n), np.nan)
    mra_out = np.zeros((n, rmax))

    if n_jobs == 1:
        # one neuron at a time, the original way
        for k in range(n):
            chi_k, mra_k = _score_one_cell(f[..., k])
            chi_out[:, k] = chi_k
            mra_out[k] = mra_k
    else:
        # spread neurons over cores; pin each worker to one thread so
        # scipy/BLAS don't fight over cores
        from joblib import Parallel, delayed, parallel_backend
        with parallel_backend("loky", inner_max_num_threads=1):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_score_one_cell)(f[..., k].copy()) for k in range(n)
            )
        for k, (chi_k, mra_k) in enumerate(results):
            chi_out[:, k] = chi_k
            mra_out[k] = mra_k

    # A missing annulus is NaN and stays out of the median. A measured zero is a
    # real score: an HCP cell is supposed to give x_FCC near zero. "No ring" means
    # none of the three differences could be computed.
    ring_found = np.isfinite(chi_out).any(axis=0)
    return chi_out, mra_out, ring_found


def _online_shuffle_zscores(
    shuf_sums: np.ndarray,
    scored_idx: np.ndarray,
    counts: np.ndarray,
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
        # Same smoothing and empty-bin fill as the real rate map.
        f_j = _rate_map_from_accumulator(
            shuf_sums[j][..., scored_idx], counts, sigma=sigma)
        sinfo_shuf[j] = _spatial_info(p, f_j).astype(np.float32)
        sidx_shuf[j]  = _sparsity_idx(p, f_j).astype(np.float32)
        del f_j

    sinfo_z = (sinfo - sinfo_shuf.mean(0)) / (sinfo_shuf.std(0) + 1e-9)
    sidx_z  = (sidx  - sidx_shuf.mean(0))  / (sidx_shuf.std(0)  + 1e-9)
    return sinfo_z.astype(np.float64), sidx_z.astype(np.float64)


def score_3d_from_map(
    sums: np.ndarray,
    counts: np.ndarray,
    *,
    shuf_sums: np.ndarray | None = None,
    sigma: float = SMOOTH_SIGMA,
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
        return dict(chi=None, mra=None, ring_found=None,
                    sinfo=None, sidx=None, sinfo_z=None, sidx_z=None, ifd=None,
                    occupancy=0.0, reliable=False,
                    scored_idx=scored_idx, bins=bins)

    # turn the totals into a smoothed rate map
    sums_sub = sums[..., scored_idx].copy()
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
                shuf_sums, scored_idx, counts, p, sigma, sinfo, sidx)
        out = dict(
            chi=np.full((3, n_scored), np.nan),
            mra=np.full((n_scored, rmax), np.nan),
            ring_found=np.zeros(n_scored, bool),
            sinfo=sinfo, sidx=sidx, sinfo_z=sinfo_z, sidx_z=sidx_z,
            ifd=_ifd_for_scored(positions, activity, scored_idx, grid_spacing, env_size),
            occupancy=occ, reliable=False,
            scored_idx=scored_idx, bins=bins,
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
            shuf_sums, scored_idx, counts, p, sigma, sinfo, sidx)

    # the structure scores, one neuron at a time
    chi_out, mra_out, ring_found = _stream_structure_scores(f, bins, n_jobs=n_jobs)

    return dict(
        chi=chi_out, mra=mra_out,
        ring_found=ring_found,
        sinfo=sinfo, sidx=sidx,
        sinfo_z=sinfo_z, sidx_z=sidx_z,
        ifd=_ifd_for_scored(positions, activity, scored_idx, grid_spacing, env_size),
        occupancy=occ, reliable=reliable,
        scored_idx=scored_idx, bins=bins,
    )
