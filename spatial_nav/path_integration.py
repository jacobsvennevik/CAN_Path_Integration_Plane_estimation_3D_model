import numpy as np
import copy
from typing import Optional

from spatial_nav.plane_estimation import (
    BinghamDistribution,
    predict,
    update,
    uniform_prior,
)
from scipy.spatial.transform import Rotation as Rot
from spatial_nav.CAN_IMP.torch_backend import TorchBackend


def build_rotation_matrix(n_hat: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Builds the rotation matricies based in n_hat so that we can later rotate the velocity.
    """
    n_hat = np.asarray(n_hat, dtype=float) 
    n_hat = n_hat / np.linalg.norm(n_hat) #make sure the estimated plane normal is unit length
    # find R such that R @ n_hat = z_hat
    z_hat = np.array([[0.0, 0.0, 1.0]])
    R, _ = Rot.align_vectors(z_hat, n_hat.reshape(1, 3))
    return R.as_matrix()

def compute_pi_star_scale(env_size: float, torus_period: float = 2 * np.pi) -> float:
    """
    Compute the scale of pi_star
    """
    return torus_period / env_size

def pi_star(v_alloc: np.ndarray) -> np.ndarray:
    """
    Applies the pushforward π★ (differential of π) to an allocentric velocity vector.
    It is the indetity matrix so a trivial computasion, in our case but kept for consistency vis a vi the MADE framework (Claudi et, al. 2025)
    """
    return np.asarray(v_alloc, dtype=float)

def step_filter(current, displacement, kappa, alpha):
    """Single predict and update cycle. """
    predicted = predict(current, alpha)
    return update(predicted, displacement, kappa)


class PathIntegrator:
    """
    Path integrator coupling the Bingham plane filter with the
    T³ QAN.
    """
    def __init__(self, qan, kappa=10.0, alpha=0.999, dt=1.0, scale=1.0, initial_estimate=None,):
        self.qan = qan 
        self.kappa = kappa #likelihood consentration for the Bingham update.
        self.alpha = alpha #predict deflation factor
        self.dt = dt #timestep size
        self.scale = scale
        self.backend = TorchBackend(qan, dt=dt) 
        self._bingham_state = initial_estimate or uniform_prior() #starting belief for n̂
        self._theta = np.zeros(qan.manifold.dim) #decoded position
        self._n_hat_corrected = None  # gravity-disambiguated, stored on self
        self.history = {
            "n_hat": [], #MAP plane normal at each step
            "z1": [], "z2": [], #concentration parameters
            "v_body": [], #body velocity
            "v_alloc": [], #allocentric velocity after rotation
            "target_speed_rad ": [], #push-forward velocity fed to CANs
            "theta": [] #decoded CAN position
            } 
        # Optional recording buffers
        # S_tot_buffer: stays on-device (no per-step CPU transfer)
        self.S_tot_buffer      = None 
        self.bingham_snapshots = None   
        
    def warmup(self, n_steps: int = 100):
        zero_v = np.zeros(3)
        for _ in range(n_steps):
            self.backend.step(zero_v)
            self._bingham_state = predict(self._bingham_state, self.alpha)
        self._theta = self._decode_from_torch()

    def step(self, v_body: np.ndarray, g: np.ndarray) -> np.ndarray:
        """
        Advance the integrator by one timestep.

        """
        v_body = np.asarray(v_body, dtype=float)
        g = np.asarray(g, dtype=float) #gravity vector
        
        displacement = v_body * self.dt
        d_norm = np.linalg.norm(displacement)
        
        if d_norm > 1e-9:
            #we only want direction, not magnitude
            displacement_unit = displacement / d_norm
            # run the bingham filter
            self._bingham_state = step_filter(self._bingham_state, displacement_unit, self.kappa, self.alpha)


        # Extract MAP estimate and disambiguate with gravity
        g_hat = g / np.linalg.norm(g)
        n_hat = self._bingham_state.M[:, -1]
        if np.dot(n_hat, g_hat) > 0:
            n_hat = -n_hat
            
        self._n_hat_corrected = n_hat

        #build rotation matrix and rotate velocity 
        R = build_rotation_matrix(n_hat, g)
        v_alloc = R @ v_body #allocentric velocity

        # push-forward the roated velocity into the Jacobian matricies
        target_speed_rad  = pi_star(v_alloc) * self.scale

        # drive each QAN
        self.backend.step(target_speed_rad )

        # decode current position
        self._theta = self._decode_from_torch()

        # Record into history
        self.history["n_hat"].append(n_hat.copy())
        self.history["z1"].append(self._bingham_state.z1)
        self.history["z2"].append(self._bingham_state.z2)
        self.history["v_body"].append(v_body.copy())
        self.history["v_alloc"].append(v_alloc.copy())
        self.history["target_speed_rad "].append(target_speed_rad .copy())
        self.history["theta"].append(self._theta.copy())

        return self._theta


    def run(self, v_body_sequence : np.ndarray, g : np.ndarray, record: bool = False,
            ) -> np.ndarray:
        """
        Run the full pipeline over a pre-computed velocity sequence.

        """

        T = v_body_sequence.shape[0] #total timesteps
        theta_history = np.zeros((T, self.qan.manifold.dim)) #place to store decoded positions
        
        if record:
            _buf  = self.backend.allocate_state_buffer(T)
            _bing = []
        else:
            _buf  = None
            _bing = None

        for t in range(T):
            theta_history[t] = self.step(v_body_sequence[t], g)
            if record:
                self.backend.record_state_to_buffer(_buf, t)
                _bing.append(copy.deepcopy(self._bingham_state))

        if record:
            self.S_tot_buffer      = self.backend.buffer_to_numpy(_buf)
            self.bingham_snapshots = _bing
        else:
            self.S_tot_buffer      = None
            self.bingham_snapshots = None

        return theta_history


    def concentration_eigenvalue_gap(self) -> float:
        """
        Diagnostics for uncertinity of the filter.
        Large gap, filter is confident.
        Small gap, still uncertain between two candidate axes.
        """
        return self._bingham_state.z2 - self._bingham_state.z1

    def reset(self, theta_0: np.ndarray, initial_estimate: Optional[BinghamDistribution] = None):
        """
        Reset filter and CAN states without rebuilding the full object.
        For running multiple trials with the same QAN hyperparameters.
        """
        if initial_estimate is None:
            self._bingham_state = uniform_prior()
        else:
            self._bingham_state = initial_estimate

        self.backend.reset(theta_0) 

        self._theta = self._decode_from_torch()  

        for key in self.history:
            self.history[key] = []
        
        self.S_tot_buffer      = None
        self.bingham_snapshots = None

            
    def _decode_from_torch(self) -> np.ndarray: 
        """Decode current bump position from the Torch backend.""" 
        S_tot = self.backend.S.mean(dim=0) 
        return self.backend.decode_position_com(S_tot).detach().cpu().numpy()