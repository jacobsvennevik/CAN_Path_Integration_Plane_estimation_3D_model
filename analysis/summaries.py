"""One-row summaries of a scored run."""
import numpy as np

INFO_Z_CUTOFF = 1.96


def _finite(values):
    if values is None:
        return np.array([])
    a = np.asarray(values, dtype=float).ravel()
    return a[np.isfinite(a)]


def _median(values):
    a = _finite(values)
    return float(np.median(a)) if a.size else float("nan")


def _fraction(values, keep):
    a = _finite(values)
    return float(np.mean(keep(a))) if a.size else float("nan")


def selectivity_row(scores, cutoff=INFO_Z_CUTOFF):
    """Median per-cell z, and the share past the shuffle line."""
    sinfo_z = scores.get("sinfo_z")
    sidx_z = scores.get("sidx_z")
    return dict(
        median_sinfo_z=_median(sinfo_z),
        frac_sinfo_z_above=_fraction(sinfo_z, lambda a: a > cutoff),
        median_sidx_z=_median(sidx_z),
        frac_sidx_z_below=_fraction(sidx_z, lambda a: a < -cutoff),
        n_info=int(_finite(sinfo_z).size),
    )


def summarize_2d(scores):
    """One-row summary of score_2d_from_map. `grid_like` comes from apply_hgs_floor."""
    hgs = np.asarray(scores["hgs"], dtype=float)
    sgs = np.asarray(scores["sgs"], dtype=float)
    finite = np.isfinite(hgs) & np.isfinite(sgs)
    n_fin = int(finite.sum())
    median_hgs = _median(hgs)
    median_sgs = _median(sgs)
    if "grid_like" in scores:
        frac_grid_like = float(np.mean(scores["grid_like"]))
    else:
        frac_grid_like = 0.0
    return dict(
        occupancy=float(scores["occupancy"]),
        ring_frac=float(scores["ring_frac"]),
        median_hgs=median_hgs,
        median_sgs=median_sgs,
        frac_hgs_gt_sgs=float(np.mean(hgs[finite] > sgs[finite])) if n_fin else float("nan"),
        frac_grid_like=frac_grid_like,
        n_active=int(scores["n_active"]),
        **selectivity_row(scores),
        hgs_shuf_p95=(
            float("nan") if scores.get("hgs_shuf_p95") is None
            else float(scores["hgs_shuf_p95"])
        ),
        hgs_shuf_ring_frac=(
            float("nan") if scores.get("hgs_shuf_ring_frac") is None
            else float(scores["hgs_shuf_ring_frac"])
        ),
    )


def pooled_ifd(scores):
    """All pairwise field distances in one score dict, in the normalised cube."""
    parts = []
    for values in (scores.get("ifd") or {}).values():
        dist = np.asarray(values, dtype=float).ravel()
        dist = dist[np.isfinite(dist)]
        if dist.size:
            parts.append(dist)
    if not parts:
        return np.empty(0)
    return np.concatenate(parts)


def pooled_ifd_cv(scores):
    """Per-cell nearest-neighbour spacing CV stored on one score dict."""
    cv = scores.get("ifd_cv")
    if cv is None:
        return np.empty(0)
    values = np.asarray(cv, dtype=float).ravel()
    return values[np.isfinite(values)]


def mra_curve(scores):
    """Stored MRA, shape (cells, radii), or None when this dict has no curve."""
    mra = scores.get("mra")
    if mra is None:
        return None
    mra = np.asarray(mra, dtype=float)
    if mra.ndim != 2 or mra.shape[1] == 0:
        return None
    return mra


def _mra_bin_m(bins, env_size):
    """Width of one autocorrelation shell in metres."""
    return float(env_size) / float(bins)


def mra_radii_m(n_radii, bins, env_size):
    """Outer edge of each MRA shell, in metres. Index 0 is the centre."""
    return np.arange(int(n_radii)) * _mra_bin_m(bins, env_size)


def _mra_spacing_index(n_radii, bins, grid_spacing, env_size):
    """Column whose shell contains the grid spacing."""
    r_grid = float(grid_spacing) / _mra_bin_m(bins, env_size)
    return int(np.clip(np.ceil(r_grid), 0, int(n_radii) - 1))


def summarize_3d(scores, grid_spacing=None, env_size=None):
    """One-row summary of score_3d_from_map, one flat row of medians and fractions.
    Missing values become NaN instead of crashing the row.
    """
    chi = scores.get("chi")
    chi = np.full((3, 0), np.nan) if chi is None else np.asarray(chi, dtype=float)
    ifd = scores.get("ifd") or {}
    n_fields = scores.get("n_fields")
    if n_fields is None:
        median_n_fields = float("nan")
        frac_local_ifd = float("nan")
    else:
        n_fields = np.asarray(n_fields)
        median_n_fields = float(np.median(n_fields)) if n_fields.size else float("nan")
        frac_local_ifd = float(np.mean(n_fields >= 4)) if n_fields.size else float("nan")
    mra = None
    if grid_spacing is not None and env_size is not None:
        mra = mra_curve(scores)
    mra_at = float("nan")
    if mra is not None:
        r = _mra_spacing_index(mra.shape[1], scores["bins"], grid_spacing, env_size)
        mra_at = _median(mra[:, r])
    ring = scores.get("ring_found")
    return dict(
        occupancy=float(scores["occupancy"]),
        reliable=bool(scores["reliable"]),
        n_active=int(np.asarray(scores["scored_idx"]).size),
        ring_frac=float(np.mean(ring)) if ring is not None and len(ring) else float("nan"),
        median_sinfo=_median(scores.get("sinfo")),
        median_sidx=_median(scores.get("sidx")),
        **selectivity_row(scores),
        median_chi_fcc=_median(chi[0]) if chi.shape[0] else float("nan"),
        median_chi_hcp=_median(chi[1]) if chi.shape[0] > 1 else float("nan"),
        median_chi_col=_median(chi[2]) if chi.shape[0] > 2 else float("nan"),
        median_ifd=_median(pooled_ifd(scores)),
        median_ifd_cv=_median(pooled_ifd_cv(scores)),
        median_n_fields=median_n_fields,
        frac_local_ifd=frac_local_ifd,
        n_ifd_cells=int(sum(len(v) > 0 for v in ifd.values())),
        mra_at_spacing=mra_at,
    )


def summarize_qy(fitted):
    return dict(
        median_grid_fit=float(fitted["median_grid_fit"]),
        frac_significant=float(fitted["frac_significant"]),
        n_silent=int(fitted["n_silent"]),
    )
