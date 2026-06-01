import numpy as np
import copy
from typing import Optional

from plane_estimation import (
    BinghamDistribution,
    predict,
    update,
    uniform_prior,
)
from scipy.spatial.transform import Rotation as Rot
from network.torch_backend import TorchBackend



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

def build_rotation_matrix(n_hat: np.ndarray, g: np.ndarray) -> np.ndarray:
    """
    Returns R such that R @ n_hat = z_hat, via the closed-form
    vector-to-vector rotation (Rodrigues). 
    """
    n_hat = np.asarray(n_hat, dtype=float)
    n_hat = n_hat / np.linalg.norm(n_hat)
    z_hat = np.array([0.0, 0.0, 1.0])

    v = np.cross(n_hat, z_hat)   # rotation axis * sin(theta)
    c = float(np.dot(n_hat, z_hat))  # cos(theta)

    # already aligned -> identity (this is the common case on a flat floor)
    if c > 1.0 - 1e-10:
        return np.eye(3)
    # anti-aligned (n_hat points straight down) -> 180° flip
    if c < -1.0 + 1e-10:
        return np.diag([1.0, -1.0, -1.0])

    vx = np.array([
        [0.0,  -v[2],  v[1]],
        [v[2],  0.0,  -v[0]],
        [-v[1], v[0],  0.0 ],
    ])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))

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
    """Single predict and update cycle."""
    predicted = predict(current, alpha)
    return update(predicted, displacement, kappa)


class PathIntegrator:
    """
    Path integrator coupling the Bingham plane filter with the
    T³ QAN.
    """
    def __init__(self, qan, kappa=10.0, alpha=0.999, scale=1.0, initial_estimate=None, record_stride=10,plane_mode="bayesian"):
        self.qan = qan 
        self.kappa = kappa #likelihood consentration for the Bingham update.
        self.alpha = alpha #predict deflation factor
        self.scale = scale
        self.backend = TorchBackend(qan) 
        self._bingham_state = initial_estimate or uniform_prior() #starting belief for n̂
        self._theta = np.zeros(qan.manifold.dim) #decoded position
        self._n_hat_corrected = None  # gravity-disambiguated, stored on self
        self.plane_mode = plane_mode #whether to use the true plane mode or the Bingham mode
        if true_n_hat is None:
            true_n_hat = np.array([0.0, 0.0, 1.0])      # gravity / flat floor
        true_n_hat = np.asarray(true_n_hat, dtype=float)
        self._true_n_hat = true_n_hat / np.linalg.norm(true_n_hat)
        self.history = {
            "n_hat": [], #MAP plane normal at each step
            "z1": [], "z2": [], #concentration parameters
            "v_body": [], #body velocity
            "v_alloc": [], #allocentric velocity after rotation
            "target_speed_rad": [], #push-forward velocity fed to CANs
            "theta": [] #decoded CAN position
            } 
        # Optional recording buffers
        # S_tot_buffer: stays on-device (no per-step CPU transfer)
        self.S_tot_buffer      = None 
        self.bingham_snapshots = None
        self.record_stride = record_stride #TODO not in use anymore
        self.ratemap_sums   = None   # set by run(..., ratemap_bins=N) when > 0
        self.ratemap_counts = None   
        
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
        
        d_norm = np.linalg.norm(v_body)
        
        if d_norm > 1e-9:
            #we only want direction, not magnitude
            v_body_t_unit = v_body / d_norm
            # run the bingham filter
            self._bingham_state = step_filter(self._bingham_state, v_body_t_unit, self.kappa, self.alpha)

        #plane mode either bayesian or true
        if self.plane_mode == "true":
            n_hat = self._true_n_hat
            self._bingham_state = step_filter(self._bingham_state, v_body_t_unit, self.kappa, self.alpha)

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
        v_phase = self.qan.manifold.metric.to_phase(pi_star(v_alloc))
        target_speed_rad = v_phase * self.scale

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
        self.history["target_speed_rad"].append(target_speed_rad .copy())
        self.history["theta"].append(self._theta.copy())

        return self._theta


    def run(self, v_body_sequence : np.ndarray, g : np.ndarray, record: bool = False, 
            flat_indices: np.ndarray = None,   ratemap_bins: int = 0) -> np.ndarray:
        """
        Run the full pipeline over a pre-computed velocity sequence.

        """

        T = v_body_sequence.shape[0] #total timesteps
        theta_history = np.zeros((T, self.qan.manifold.dim)) #place to store decoded positions
        
        if record:
            _buf  = self.backend.allocate_state_buffer(T, stride=self.record_stride)
            _bing = []
        else:
            _buf  = None
            _bing = None
            
        _acc = None
        if flat_indices is not None and ratemap_bins > 0:
            _acc = self.backend.allocate_ratemap(ratemap_bins)

        for t in range(T):
            theta_history[t] = self.step(v_body_sequence[t], g)
            if record and (t % self.record_stride== 0):
                self.backend.record_state_to_buffer(_buf, t, stride=self.record_stride)
                _bing.append(copy.deepcopy(self._bingham_state))
            if _acc is not None:              # ADD ── one line in the hot loop
                self.backend.record_ratemap(_acc, int(flat_indices[t]))

        if record:
            self.S_tot_buffer      = self.backend.buffer_to_numpy(_buf)
            self.bingham_snapshots = _bing
        else:
            self.S_tot_buffer      = None
            self.bingham_snapshots = None
        
        if _acc is not None:
            self.ratemap_sums, self.ratemap_counts = \
            self.backend.ratemap_to_numpy(_acc, ratemap_bins)
        else:
            self.ratemap_sums = self.ratemap_counts = None
            return theta_history

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