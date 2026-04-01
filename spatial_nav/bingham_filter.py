import numpy as np
from scipy.special import i0, i1
from scipy.optimize import fsolve
from scipy.linalg import eigh
import matplotlib.pyplot as plt

"""
Implementasion of a recurrent Bingham distribution filter using Kurz et al. 2014 
see references. 

"""

#TODO: add the equations from the paper

class BinghamDistribution:
    """
    Represents a single 2D Bingham distribution.
    
    M : (2,2) orthogonal matrix
        Defines the orientation of the distribution in space.
        The last column is the mode, the most probable direction.
        Columns must be perpendicular unit vectors.
    
    Z : (2,2) diagonal matrix
        Defines the concentration, how certain the distribution is.
        Z = diag(z1, 0) where z1 <= 0 always.
        z1 close to 0   -> very uncertain, spread over the whole circle
        z1 very negative -> very certain, sharply peaked at the mode
        
        
    """
    def __init__(self, M, Z):
        M = np.array(M, dtype=float) 
        Z = np.array(Z, dtype=float)

        # TODO: remove asserts after debugging
        assert np.allclose(M @ M.T, np.eye(2), atol=1e-6)
        assert abs(Z[0, 1]) < 1e-10 and abs(Z[1, 0]) < 1e-10
        assert Z[0, 0] <= Z[1, 1] + 1e-10
        assert abs(Z[1, 1]) < 1e-10

        self.M = M
        self.Z = Z
        self.z1 = Z[0, 0] # The only free parameter in d=2

    @property
    def mode(self):
        return self.M[:, -1] #Returns the mode, most probable direction of the distribution

    @property
    def mode_angle(self):
        m = self.mode
        angle = np.arctan2(m[1], m[0])
        return angle % np.pi

    
    def pdf(self, theta):
        """
        The probability density function of the Bingham function. Taken from the paper. 
        Uses M, Z and F 
        Outputs: a probability density
        """
        x = np.array([np.cos(theta), np.sin(theta)]) #x is the point on the unit sphere
        exponent = x @ self.M @ self.Z @ self.M.T @ x #
        F = compute_F(self.z1) #The normalisation constant
        return np.exp(exponent) / F

    #TODO: also more debugging can remove this later
    def __repr__(self):
        return (f"BinghamDistribution(\n"
                f"  M={self.M},\n"
                f"  Z={self.Z},\n"
                f"  mode_angle={np.degrees(self.mode_angle):.1f} deg\n"
                f"  z1={self.z1:.3f}\n)")


def compute_F(z1):
    """
    Computes the normalisation constant. 
    
    Formula: Lemma 6, equation (77)
    Derived:
    This is a direct transcription. The paper derives it 
    by showing the d=2 Bingham is a von Mises in disguise, then substituting the von Mises 
    normalization constant.
    
    
    Uses z1 - concetration parameter
    """
    return 2 * np.pi * i0(z1 / 2) * np.exp(z1 / 2)


def compute_dF_dz1(z1):
    """
    Computes the change of the F in respect to z1
    
    Formula: Lemma 6, equation 77
    """
    return np.pi * np.exp(z1 / 2) * (i1(z1 / 2) + i0(z1 / 2))


def compute_dF_dz2(z1, z2=0.0):
    """
    Computes the changes of F in respect to z2
    
    Formula: Lemma 6, equation 78
    """
    return (np.pi * np.exp((z1 + z2) / 2) * (i0((z1 - z2) / 2) - i1((z1 - z2) / 2)))


def bingham_to_covariance(bingham):
    """
    Converts a Bingham distribution to its covariance matrix S = E[x xᵀ].
    
    Goes from Bingham parameters (M, Z) -> covariance matrix S.
    Used in the prediction step before composition.
    
    FORMULA:
    Equation (16) and Equation (15)
    
    """
    F = compute_F(bingham.z1)
    dF1 = compute_dF_dz1(bingham.z1) 
    dF2 = compute_dF_dz2(bingham.z1)

    omega1 = dF1 / F #  equation (16)
    omega2 = dF2 / F

    Omega = np.diag([omega1, omega2])
    S = bingham.M @ Omega @ bingham.M.T # equation (15)
    return S


def covariance_to_bingham(S):
    """
    Parameter estimation. 
    
    Uses the covariance matrix S = E[x xᵀ] to find the Bingham distribution that produced it.
    
    Goes from covariance matrix S -> Bingham parameters (M, Z).
    Used in the prediction step before composition.
    
    FORMULA:
    Equation (13) and Equation (15)
    """
    eigenvalues, eigenvectors = eigh(S) # Eigendecomposition equation (13) 

    omega1 = eigenvalues[0]
    M = eigenvectors

    def equation(z1_candidate):
        F = compute_F(z1_candidate[0])
        dF1 = compute_dF_dz1(z1_candidate[0])
        return dF1 - omega1 * F

    z1_solution = fsolve(equation, x0=[-1.0], full_output=False)[0] #Equation 14
    z1_solution = min(z1_solution, 0.0)

    Z = np.diag([z1_solution, 0.0])
    return BinghamDistribution(M, Z)


def compose_covariances_2d(A, B):
    """
    Computes the covariance of the composed Bingham distribution A ⊕ B,
    given the covariances of A and B separately.
    
    Used in the prediction step because Bingham distributions are not closed
    under composition so we compose in covariance space instead.
    """
    a11, a12, a22 = A[0,0], A[0,1], A[1,1]
    b11, b12, b22 = B[0,0], B[0,1], B[1,1]

    c11 = a11*b11 - 2*a12*b12 + a22*b22 #Equation 28
    c12 = a11*b12 - a12*b22 + a12*b11 - a22*b12 #Equation 29
    c22 = a11*b22 + 2*a12*b12 + a22*b11 #Equation 30

    return np.array([[c11, c12],
                     [c12, c22]]) # The trace of C always equals 1


def compose_complex(x, y):
    """
    Defines the compose opperator ⊕ for d=2.
    Composition is defined on the unit circle as complex multiplication. 
    Treating the 2D unit vector [x₁, x₂]ᵀ as the complex number x₁ + ix₂. 
    
    FORMULA:
    Equation (24)
    """
    x1, x2 = x[0], x[1]
    y1, y2 = y[0], y[1]
    return np.array([x1*y1 - x2*y2, 
                     x1*y2 + x2*y1]) 


def compose_matrix_columns(z_vec, M):
    """
    Helper for the update step. The update step requires composing 
    a measurement vector with each column of the noise orientation matrix M.
    Rather than writing loop in update, pulled out here.
    """
    result = np.zeros_like(M)
    for col in range(M.shape[1]):
        result[:, col] = compose_complex(z_vec, M[:, col])
    return result


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


def predict(estimate, sys_noise):
    """
    Algorithm 1 from Kurz et al. (2014). This is the Bayesian prediction step.
    """
    A = bingham_to_covariance(estimate) # (M,Z) -> covariance matrix
    B = bingham_to_covariance(sys_noise) # (M,Z) -> covariance matrix
    C = compose_covariances_2d(A, B) # covariance of the two distributsions. 
    return covariance_to_bingham(C) # fit new Bingham to C  (eq 13-14)


def update(prediction, meas_noise, measurement):
    """
    Algorithm 2 from Kurz et al. (2014). This is the Bayesian update step.
    Corrects the prediction using a new measurement.
    """
    D = np.diag([1.0, -1.0])
    DM = D @ meas_noise.M
    M_likelihood = compose_matrix_columns(measurement, DM)
    likelihood = BinghamDistribution(M_likelihood, meas_noise.Z) #Build the rotated likelihood as a Bingham distribution 
    return multiply_bingham(likelihood, prediction)


def run_bingham_filter(initial_estimate, sys_noise, meas_noise,
                       measurements, true_states=None):
    """
    This is the complete reccursive Bingham distribution filter. 
    
    Predict then update, repeat. 
    """
    estimates = []
    current = initial_estimate

    for k, z_k in enumerate(measurements):
        predicted = predict(current, sys_noise) #predicts
        current = update(predicted, meas_noise, z_k) #updates
        estimates.append(current)

    return estimates


