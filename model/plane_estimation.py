import numpy as np
from scipy.special import ive

"""
Recursive von Mises-Fisher filter on S^2, following Kurz et al. (2016),
See references.

"""

D = 3 


def A_d(kappa, d=D):
    """ Computes the length of the mean resultant vector."""
    if kappa <= 0.0:
        return 0.0
    if not np.isfinite(kappa):
        return 1.0
    return float(ive(d / 2.0, kappa) / ive(d / 2.0 - 1.0, kappa))


def A_d_inverse(r, d=D, iters=3, kappa_max=1e6):
    """ Computes the inverse of the length of the mean resultant vector. Meaning 
    what kappa would produce a mean resultant vector of length r.
    """
    r = float(np.clip(r, 0.0, 1.0))
    if r <= 0.0:
        return 0.0
    if r >= 1.0:
        return kappa_max
    # Initial guess for kappa
    kappa = r * (d - r ** 2) / (1.0 - r ** 2)
    for _ in range(iters): #Newton method on top of that guess.
        a = A_d(kappa, d)
        kappa = kappa - (a - r) / (1.0 - a ** 2 - (d - 1.0) / kappa * a)
    return float(min(max(kappa, 0.0), kappa_max))


class VonMisesFisherDistribution:
    """ The von Mises-Fisher distribution on the sphere. Holds the filter state."""
    def __init__(self, mu, kappa):
        mu = np.asarray(mu, dtype=float)
        self.mu = mu / np.linalg.norm(mu)
        self.kappa = float(kappa)

    def __repr__(self):
        return (f"VonMisesFisherDistribution(\n"
                f"  mu={np.round(self.mu, 4)},\n"
                f"  kappa={self.kappa:.4f}\n)")


def predict(estimate, kappa_w, d=D):
    """Identity system model: keep the direction, lower the confidence.
    This is the filter's one approximation (moment matching)."""
    return VonMisesFisherDistribution(
        estimate.mu, A_d_inverse(A_d(estimate.kappa, d) * A_d(kappa_w, d), d))


def update(prediction, measurement, kappa_v):
    """
    This is the Bayesian update step.
    Corrects the prediction using a new measurement.
    """
    measurement = np.asarray(measurement, dtype=float)
    measurement = measurement / np.linalg.norm(measurement)
    #compute the new its exact because both exponents are linear in x, so they simply add.
    theta = prediction.kappa * prediction.mu + kappa_v * measurement
    kappa = float(np.linalg.norm(theta))
    mu = theta / kappa if kappa > 0.0 else prediction.mu
    return VonMisesFisherDistribution(mu, kappa)


def run_vmf_filter(initial_estimate, measurements, kappa_v, kappa_w=1e4):
    """Runs the recursive (bayesian) vMF filter.
    
    Predict then update, repeat. 
    """
    estimates = []
    current = initial_estimate
    for z in measurements:
        #forget first, then learn.
        current = update(predict(current, kappa_w), z, kappa_v)
        estimates.append(current)
    return estimates


def uniform_prior():
    """Returns a uniform prior, which is a vMF pointing north with kappa = 0. 
        Just a starign state.
    """
    return VonMisesFisherDistribution(np.array([0.0, 0.0, 1.0]), 0.0)


def sample_vmf(mu, kappa, rng):
    """One draw from vMF(mu, kappa) on S^2 (closed-form inverse CDF of the polar cosine)."""
    from model.path_integration import build_rotation_matrix
    mu = np.asarray(mu, dtype=float) 
    mu = mu / np.linalg.norm(mu) #make mu a unit vector
    kappa = float(kappa) #kappa is the concentration parameter
    u = float(rng.random()) #uniform random number between 0 and 1
    phi = 2.0 * np.pi * float(rng.random()) #direction aorund mu
    if kappa < 1e-12: #catch to not divide by zero
        w = 2.0 * u - 1.0 
    else:
        #calculates how far off is this noisy normal from the true one
        # w says how much of the sample points along μ,
        w = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa 
    s = np.sqrt(max(1.0 - w * w, 0.0)) # the sideways part of the sample points
    return build_rotation_matrix(mu).T @ np.array([s * np.cos(phi), s * np.sin(phi), w]) #builds the vector and then rotates it to face μ.


def held_normal_stats(n_held, n_true, refresh_mask):
    """Gives the RMS angular error, mean angular rate, refresh count, error time-course.
    how wrong was the normal the network actually used, and how much did that normal move around.
    """
    n_held = np.asarray(n_held, dtype=float)
    n_true = np.asarray(n_true, dtype=float)
    dots = np.einsum("ij,ij->i", n_held, n_true) #takes, at each time step, the dot product of the held and true normal.
    err_deg = np.rad2deg(np.arccos(np.clip(dots, -1.0, 1.0))) #converts the error of the held normal at each step in degrees.
    
    #compares each step with the one before it to get the angular rate
    if len(n_held) > 1:
        dots_h = np.einsum("ij,ij->i", n_held[1:], n_held[:-1])
        rate = np.rad2deg(np.arccos(np.clip(dots_h, -1.0, 1.0)))
        mean_rate = float(np.mean(rate))
    else:
        mean_rate = 0.0
    return dict(
        rms_err_deg=float(np.sqrt(np.mean(err_deg ** 2))),
        mean_rate_deg=mean_rate,
        refresh_count=int(np.asarray(refresh_mask).astype(bool).sum()),
        err_deg=err_deg,
    )


