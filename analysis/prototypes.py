"""
prototypes.py — scorer sanity check + prototype reference scores.

Builds the four prototype lattices from Gong & Yu (2021). Namely, FCC, HCP, columnar
(COL) and random (RND). Scores each with the vendored `chi_score`, and checks
that every structured prototype wins its own category while RND scores low on
all three.
"""
import numpy as np

from . import gongyu_scoring as gy        # when imported as part of the package
         # when the module is on sys.path / in a notebook

STRUCT_TYPES = ("fcc", "hcp", "col", "rnd")
_CHI_NAMES = ("fcc", "hcp", "col")            # the order chi_score returns

# --- prototype build constants (named, not magic; match scoring.BINS / AUTOCORR_TH) ---
PROTO_R       = 0.7                 # lattice spacing used by the G&Y notebook
PROTO_ROTZ    = 8 * np.pi / 180     # CALIBRATED frame; chi_score's plane indices assume it
PROTO_SCALE   = 0.1                 # per-point Gaussian jitter when building the density
PROTO_N_JIT   = 500                 # jitter replicates per lattice point
PROTO_N_RND   = 100                 # points in the random (RND) baseline cloud
PROTO_BINS    = 40                  # histogram bins per axis (== scoring.BINS)
PROTO_TH      = 0.1                 # autocorrelation threshold (== scoring.AUTOCORR_TH)


def wrap(struct_type, r=PROTO_R, rotz=PROTO_ROTZ, n_jit=PROTO_N_JIT,
         scale=PROTO_SCALE, bins=PROTO_BINS, th=PROTO_TH, n_rnd=PROTO_N_RND,
         seed=None):
    """Build one prototype's 3-D autocorrelation, moslty like G and Y
    """
    rng = np.random.default_rng(seed)
    if struct_type == "rnd":
        points = rng.uniform(-1, 1, size=(n_rnd, 3))
    else:
        points = gy.hexagonal_structure(struct_type, r, rotz)
    pts = rng.normal(loc=points, scale=scale, size=(n_jit, *points.shape))
    pts = pts.reshape(-1, 3)
    pts = pts[(np.abs(pts) < 1).all(axis=1)]                 # keep points in (-1,1)^3
    f = np.histogramdd(pts, bins=bins, range=((-1, 1),) * 3, density=False)[0]
    return gy.autocorr(f, th), f


def _as_float(z) -> float:
    """chi_score returns length-n arrays, squeeze to scalar."""
    return float(np.ravel(z)[0])


def _score_prototype(struct_type, seed, wrap_kwargs=None, chi_kwargs=None):
    """ Build the prototype's [chi_fcc, chi_hcp, chi_col] autocorrelogram
    """
    ac, _ = wrap(struct_type, seed=seed, **(wrap_kwargs or {}))
    return [_as_float(c) for c in gy.chi_score(ac, **(chi_kwargs or {}))]


def reference_scores(n_repeats: int = 30, seed: int = 0,
                     wrap_kwargs: dict | None = None,
                     chi_kwargs: dict | None = None) -> dict:
    """Mean prototype chi values for plotting (Gong & Yu's `std` baseline).
    """
    diag = {t: [] for t in _CHI_NAMES}
    rnd_max = []
    for rep in range(n_repeats):
        base = seed + 1000 * rep
        for i, t in enumerate(_CHI_NAMES):
            diag[t].append(
                _score_prototype(t, base + i, wrap_kwargs, chi_kwargs)[_CHI_NAMES.index(t)])
        rnd_max.append(max(_score_prototype("rnd", base + 99, wrap_kwargs, chi_kwargs)))
    out = {t: float(np.mean(diag[t])) for t in _CHI_NAMES}
    out["rnd_max"] = float(np.mean(rnd_max))
    return out


def global_order_reference(n_repeats: int = 5, seed: int = 0,
                           az_precision: int = 48, al_precision: int = 24,
                           radial_method: str = "max",
                           wrap_kwargs: dict | None = None) -> dict:
    """Build the best (prototype) lines, the ceiling and the worst (random cloud) floor references.

    """
    def _best_plane_hgs(struct_type, seed):
        ac, _ = wrap(struct_type, seed=seed, **(wrap_kwargs or {}))
        # `ac` is already (d, d, d, 1) from gy.autocorr
        hgs_map, _ = gy.gridness_map(ac, az_precision=az_precision,
                                     al_precision=al_precision, al_max=np.pi / 2,
                                     radial_method=radial_method)
        return float(hgs_map.max())

    struct, rnd = [], []
    for rep in range(n_repeats):
        base = seed + 1000 * rep
        struct += [_best_plane_hgs(t, base + i) for i, t in enumerate(_CHI_NAMES)]
        rnd.append(_best_plane_hgs("rnd", base + 99))
    return {"struct": float(np.mean(struct)), "rnd": float(np.mean(rnd))}
