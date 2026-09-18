import numpy as np
from scipy.special import iv, ive
import matplotlib.pyplot as plt

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


def c_d(kappa, d=D):
    """ Computes the normalization constant for the von Mises-Fisher distribution."""
    return kappa ** (d / 2.0 - 1.0) / ((2.0 * np.pi) ** (d / 2.0) * iv(d / 2.0 - 1.0, kappa))


class VonMisesFisherDistribution:
    """ The von Mises-Fisher distribution on the sphere. Holds the filter state."""
    def __init__(self, mu, kappa):
        mu = np.asarray(mu, dtype=float)
        self.mu = mu / np.linalg.norm(mu)
        self.kappa = float(kappa)

    @property
    def mode(self): 
        """ Returns the mode of the distribution, the most likely direction."""
        return self.mu

    def pdf(self, x):
        """ Evaluates the density at a point, diganostic"""
        x = np.asarray(x, dtype=float)
        return c_d(self.kappa) * np.exp(self.kappa * (x @ self.mu))

    def __repr__(self):
        return (f"VonMisesFisherDistribution(\n"
                f"  mu={np.round(self.mu, 4)},\n"
                f"  kappa={self.kappa:.4f}\n)")


def deterministic_sampling(mu, kappa, d=D):
    """Algorithm 1, from kurz (2016) paper. 
    Samples point whose vector average has exactly length A_d(κ)"""
    # Initialize the sample matrix S
    S = np.zeros((d, 2 * d - 1))
    S[0, 0] = 1.0
    # compute the target length the samples should integrete to
    m1 = A_d(kappa, d)
    #Solves for the one tilt angle a that makes the points own resultant length equal m1.
    alpha = np.arccos(((2 * d - 1) * m1 - 1.0) / (2 * d - 2))
    for i in range(1, d):
        S[0, 2 * i - 1] = np.cos(alpha)
        S[0, 2 * i] = np.cos(alpha)
        S[i, 2 * i - 1] = np.sin(alpha)
        S[i, 2 * i] = -np.sin(alpha)
    # Rotate the samples to the desired direction.
    M = np.zeros((d, d))
    M[:, 0] = np.asarray(mu, dtype=float)
    Q, R = np.linalg.qr(M)
    if R[0, 0] < 0:
        # sign check ensures you land on μ rather than −μ.
        Q = -Q
    return Q @ S


def parameter_estimation(S, weights=None, d=D, tol=1e-12):
    """ the inverse of sampling. Given a cloud of points fit the vMF."""
    S = np.asarray(S, dtype=float)
    n = S.shape[1]
    w = np.full(n, 1.0 / n) if weights is None else np.asarray(weights, dtype=float)
    # compute the mean of the samples
    m = S @ w
    r = float(np.linalg.norm(m))
    if r < tol:
        return None, 0.0
    return m / r, A_d_inverse(r, d)


def predict(estimate, kappa_w, a=None, d=D):
    """the forgetting step. Advances the belief one step and lets confidence decay. This is the one approximation in the filter"""
    #Step 1: scatter 5 representative points
    S = deterministic_sampling(estimate.mu, estimate.kappa, d)
    #Step 2: push each through the system function a_k.
    if a is not None:
        S = np.column_stack([a(S[:, i]) for i in range(S.shape[1])])
    #Step 3: re-fit. A uniform state predicts to a uniform state, keeping the placeholder direction mu.
    mu, kappa = parameter_estimation(S, d=d)
    if mu is None:
        #Return a uniform state if mu is none
        return VonMisesFisherDistribution(estimate.mu, 0.0)
    #Step 4: return the new estimate.
    return VonMisesFisherDistribution(mu, A_d_inverse(A_d(kappa, d) * A_d(kappa_w, d), d))


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


def run_vmf_filter(initial_estimate, measurements, kappa_v, kappa_w=100.0):
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


def init_from_normal_guess(n_hat_guess, kappa=0.1):
    """
    Weak prior centered on the guessed plane normal.
    """
    return VonMisesFisherDistribution(n_hat_guess, kappa)


def angular_error_s2(est_mode, n_true):
    """ Use for visualisation and checking in experiemnts """
    dot = np.clip(np.dot(est_mode, n_true), -1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def vmf_pdf_on_sphere(estimate, n_points=100):
    """
    Visualisation helper
    """
    theta = np.linspace(0, np.pi, n_points)
    phi = np.linspace(0, 2 * np.pi, n_points)
    T, P = np.meshgrid(theta, phi)

    X = np.sin(T) * np.cos(P)
    Y = np.sin(T) * np.sin(P)
    Z = np.cos(T)

    pts = np.stack([X, Y, Z], axis=-1)
    PDF = c_d(estimate.kappa) * np.exp(estimate.kappa * (pts @ estimate.mu))

    return X, Y, Z, PDF


