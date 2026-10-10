"""Torus geometry: wrapped distances, unwrap, field spacing."""
import numpy as np


def wrapped_angle_diff(a, b, period=2 * np.pi):
    """
    Since theta_i lives on the circle (is periodic) we need to calculate the angular distance.
    """
    return (a - b + period / 2) % period - period / 2


def unwrap_torus(theta, period=2 * np.pi):
    """Unwrap a (T, d) torus path by accumulating wrapped step differences."""
    theta = np.asarray(theta, dtype=float)
    if theta.shape[0] == 0:
        return theta.copy()
    step = wrapped_angle_diff(theta[1:], theta[:-1], period=period)
    return theta[0] + np.vstack(
        [np.zeros((1, theta.shape[1])), np.cumsum(step, axis=0)]
    )


def field_spacing_rad(n, bump_spacing_cells):
    """One hexagonal field spacing, in radians on the torus."""
    return float(bump_spacing_cells) * 2.0 * np.pi / float(n)


def torus_error_periods(decoded, truth, n, bump_spacing_cells):
    """Wrapped Euclidean error as a fraction of one field spacing."""
    err = np.linalg.norm(
        wrapped_angle_diff(np.asarray(decoded), np.asarray(truth)), axis=1
    )
    return err / field_spacing_rad(n, bump_spacing_cells)


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
