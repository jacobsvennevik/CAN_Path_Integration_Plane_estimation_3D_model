"""Measure bump speed vs commanded speed, then set velocity_gain so they match.
Possibly only useful in the three dimensional case.
"""

import numpy as np


def dirs_for(d: int) -> dict:
    """Torus-axis directions of length `d` (not world axes)."""
    names = ("x", "y", "z")
    out = {}
    for i in range(d):
        e = np.zeros(d, dtype=float)
        e[i] = 1.0
        label = names[i] if i < 3 else str(i + 1)
        out[f"+{label}"] = e.copy()
        out[f"-{label}"] = -e
        out[f"+θ{i+1}"] = e.copy()
        out[f"-θ{i+1}"] = -e.copy()
    ones = np.ones(d, dtype=float)
    out["all"] = ones
    out["111"] = ones
    if d >= 2:
        e = np.zeros(d, dtype=float)
        e[0] = 1.0
        e[1] = 1.0
        out["110"] = e
    return out


# 3-D aliases kept so existing notebooks that import DIRS still resolve.
DIRS = {k: tuple(v) for k, v in dirs_for(3).items()}
CAL_DIRS = ("+x", "111")


def _unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("zero direction vector")
    return v / n


def _peakedness(backend) -> float:
    """max / mean activity. Near 1 means the lattice is gone."""
    s = backend.S.mean(dim=0)
    m = float(s.mean())
    return float(s.max()) / m if m > 1e-12 else float("nan")


def _resolve_direction(backend, direction):
    if isinstance(direction, str):
        table = dirs_for(backend.d)
        if direction not in table:
            raise KeyError(f"unknown direction {direction!r} for d={backend.d}")
        direction = table[direction]
    return _unit(direction)


def measure(backend, theta_0, direction, speed, n_meas=None, burn=None,
            periods=10.0):
    """Drive at constant speed, fit the bump velocity, put the state back."""
    dt = float(backend.qan.dt)
    tau = float(backend.tau)
    u = _resolve_direction(backend, direction)
    theta_0 = np.asarray(theta_0, dtype=np.float64)
    shp = float(backend.n)
    extent = 2.0 * np.pi

    period_cells = float(backend.bump_period_cells())
    period_coord = period_cells * (extent / shp)
    if burn is None:
        burn = int(np.ceil(50.0 * tau / dt))
    if n_meas is None:
        n_meas = int(np.ceil(float(periods) * period_coord / (speed * dt)))

    S0 = backend.S.clone()
    pk_before = _peakedness(backend)

    T = int(burn) + int(n_meas)
    traj = (theta_0 + u * speed * np.arange(T)[:, None] * dt) % extent
    backend.drive(traj, display_stride=T + 1, snapshot_stride=T + 1)
    pos = backend.pos_com_unwrapped

    seg = pos[burn:]
    t = np.arange(len(seg), dtype=float) * dt
    A = np.vstack([t, np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, seg, rcond=None)
    v_hat = coef[0]
    ripple = float(np.sqrt(((seg - A @ coef) ** 2).sum(axis=1).mean()))

    d_cells = np.diff(pos, axis=0) * shp / extent
    steps = np.linalg.norm(d_cells, axis=1)
    max_step = float(steps.max()) if steps.size else 0.0

    pk_after = _peakedness(backend)
    period_after = float(backend.bump_period_cells())
    backend.S = S0.clone()

    mag = float(np.linalg.norm(v_hat))
    par = float(v_hat @ u)
    angle = (float(np.degrees(np.arccos(np.clip(par / mag, -1.0, 1.0))))
             if mag > 1e-12 else float("nan"))

    lattice_ok = (np.isfinite(pk_after) and pk_after >= 0.8 * pk_before
                  and abs(period_after - period_cells) <= 0.05 * period_cells)
    ok = bool(np.all(np.isfinite(v_hat))
              and max_step <= 0.5 * period_cells
              and lattice_ok)

    return dict(
        speed=float(speed),
        g_par=par / speed,
        g_mag=mag / speed,
        angle_deg=angle,
        v_hat=v_hat,
        ripple_rad=ripple,
        max_step_cells=max_step,
        pk_before=pk_before, pk_after=pk_after,
        period_cells=period_cells, period_after=period_after,
        ok=ok,
    )


def fit_gain(backend, theta_0, speed, tol=0.02, max_iter=2, dirs=None,
             **kw):
    """Measure on +x and 111, then scale velocity_gain so g ≈ 1.

    At general d this is axis-0 and the all-ones direction.
    Returns (velocity_gain, g). Restores the old gain if the lattice died.
    """
    if dirs is None:
        dirs = CAL_DIRS if backend.d == 3 else ("+θ1", "all")
    vg0 = float(backend.qan.velocity_gain)
    g = float("nan")
    table = dirs_for(backend.d)
    for _ in range(int(max_iter)):
        rows = [measure(backend, theta_0, table[d] if isinstance(d, str) else d,
                        speed, **kw) for d in dirs]
        if not all(r["ok"] for r in rows):
            backend.qan.velocity_gain = vg0
            bad = next(r for r in rows if not r["ok"])
            return float("nan"), float(bad["g_par"])
        g = float(np.mean([r["g_par"] for r in rows]))
        if not np.isfinite(g) or g < 0.1:
            backend.qan.velocity_gain = vg0
            return float("nan"), g
        if abs(g - 1.0) < tol:
            return float(backend.qan.velocity_gain), g
        backend.qan.velocity_gain /= g
    return float("nan"), g
