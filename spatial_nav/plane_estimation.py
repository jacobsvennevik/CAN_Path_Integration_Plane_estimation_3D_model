import numpy as np
from scipy.integrate import dblquad
from scipy.optimize import fsolve
from scipy.linalg import eigh
import matplotlib.pyplot as plt
from scipy.special import hyp1f1
import numpy as np

"""
Implementasion of a recurrent Bingham distribution filter using Kurz et al. 2014 
see references. 

"""

class BinghamDistribution:
    """
    Represents a single 2D Bingham distribution.
    
    M : (3,3) orthogonal matrix
        Defines the orientation of the distribution in space.
        The last column is the mode, the most probable plane normal.
        Columns must be perpendicular unit vectors.
    
    Z : (3,3) diagonal matrix
        Defines the concentration, how certain the distribution is.
        Z = diag(z1, z2, 0) where z1 <= z2 <= 0 always.
        z1 : concentration along m1 
        z2 : concentration along m2 
        The third is fixed at 0
        The more negative it is the more certain the estimation is, 
        and the smaller spread of the probability distribution. And opposite
        the closer to 0 the more uncertain.
        
        
    """
    def __init__(self, M, Z):
        M = np.array(M, dtype=float) 
        Z = np.array(Z, dtype=float)

        self.M = M
        self.Z = Z
        self.z1 = Z[0, 0]
        self.z2 = Z[1, 1]


    @property
    def mode(self):
        return self.M[:, -1] #Returns the mode, most probable plane normal direction


    def pdf(self, x):
        """
        The probability density function of the Bingham function. Taken from the paper. 
        Uses M, Z and F 
        Outputs: a probability density
        """
        x = np.asarray(x, dtype=float) 
        A = self.M @ self.Z @ self.M.T
        F = compute_F(self.z1, self.z2) #normalisation constant
        return np.exp(x @ A @ x) / F

    #TODO: also more debugging can remove this later
    def __repr__(self):
        return (f"BinghamDistribution(\n"
                f"  M={self.M},\n"
                f"  Z={self.Z},\n"
                f"  mode={self.mode}\n"
                f"  z1={self.z1:.4f}, z2={self.z2:.4f}\n)")
        
def _integrand(phi, t, z1, z2):
    """
    Integrand for the Bingham normalisation constant on S².
    """
    return np.exp(z1 * (1 - t**2) * np.cos(phi)**2
                + z2 * (1 - t**2) * np.sin(phi)**2)

def compute_F(z1, z2):
    """
    Computes the normalisation constant.
    with Z = diag(z1, z2, 0).
    
    Used to analyse results, during experiemnts we will use lookup tables.
    Beacuse of the computasional overhead of computing confluent hypergeometric function.
    
    
    Uses z1, z2 - concetration parameters
    """
    #Calculates the double integral, with limits -1,1 for the outer integral over t
    # And the inner integral over phi
    val, _ = dblquad(_integrand, -1, 1, 0, 2*np.pi, args=(z1, z2))
    return val



def multiply_bingham(b1, b2):
    """
    The bayesian update opperation. Since Bingham probability PDF´s is closed under multiplication,
    this return an exact with no approximation. 
    
    FORMULA:
    Equation (12)
    """
    #From equation (12), multiplying two Bingham densities gives:
    C = (b1.M @ b1.Z @ b1.M.T + 
         b2.M @ b2.Z @ b2.M.T)
    #Recover M and Z from C
    eigenvalues, eigenvectors = eigh(C) 
    M_new = eigenvectors
    #Enforce the Z convention (last entry of Z is 0):
    D = eigenvalues - eigenvalues[-1]
    Z_new = np.diag(D)

    return BinghamDistribution(M_new, Z_new)


def predict(estimate,  alpha):
    """
    Does an approximation of isotropic diffusion on S^2, meaning spread the distribution outward uniformly.
    This is a departue and simplification of the kurz paper
    """
    #Add a slight uncertinity to the plane normal estimation
    Z_pred = alpha * estimate.Z
    # Last diagonal entry has to be 0
    Z_pred[2, 2] = 0.0
    #M left unchanged, new more uncertain Z
    return BinghamDistribution(estimate.M.copy(), Z_pred)


def update(prediction, measurement, kappa):
    """
    This is the Bayesian update step.
    Corrects the prediction using a new measurement.
    """
    A_prior = prediction.M @ prediction.Z @ prediction.M.T
    A_likelihood = -kappa * np.outer(measurement, measurement)
    #Bayes rule, + becasue it is applied to distributions inside an exponent
    A_posterior = A_prior + A_likelihood
    #recover the new M and Z by eigendecomposing A_posterior.
    eigenvalues, eigenvectors = eigh(A_posterior)
    #the largest eigenvalue maps to z = 0. 
    z = eigenvalues - eigenvalues[-1]
    return BinghamDistribution(eigenvectors, np.diag(z))


def run_bingham_filter(initial_estimate, measurements, kappa, alpha=0.999):
    """
    This is the complete reccursive Bingham distribution filter. 
    
    Predict then update, repeat. 
    """
    estimates = []
    current = initial_estimate
    #loop over each displacement vector
    for d_t in measurements:
        predicted = predict(current, alpha)      # mode fixed, Z deflated
        current = update(predicted, d_t, kappa)  # accumulate displacement evidence
        estimates.append(current)
 
    return estimates
 


def uniform_prior():
    """Uniform distribution on S². Maximum uncertainty about the plane normal."""
    return BinghamDistribution(np.eye(3), np.zeros((3, 3)))

def plane_axes(n):
    """
    Returns two orthonormal vectors e1, e2 that span the plane
    perpendicular to unit vector n. As well as unit vector n
    """
    n   = np.asarray(n, dtype=float)
    n  /= np.linalg.norm(n)
    
    ## Gram-Schmidt: build two axes perpendicular to n
    arb = np.array([1., 0., 0.]) if abs(n[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = arb - (arb @ n) * n
    e1 /= np.linalg.norm(e1)
    e2  = np.cross(n, e1)
    return e1, e2, n



def init_from_normal_guess(n_hat_guess, z1=-0.1, z2=-0.1):
    """
    Weak prior centred on a guessed plane normal.

    Builds an orthonormal frame with n_hat_guess as the mode.
    z1, z2 small negative -> weak prior. More negative -> more certain.
    """

    e1, e2, n_normalised = plane_axes(n_hat_guess)

    M = np.column_stack([e1, e2, n_normalised])  # last column is the mode
    Z = np.diag([z1, z2, 0.0])
    return BinghamDistribution(M, Z)