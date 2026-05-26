import pickle
from dataclasses import dataclass, field

import numpy as np

from spatial_nav.CAN_IMP.metrics import wrapped_angle_diff
from spatial_nav.path_integration import PathIntegrator

@dataclass
class ExperimentResult:
    # Trajectory
    world_pos:        np.ndarray   # (T, 3) physical positions
    v_body_seq:       np.ndarray   # (T, 3) velocities fed to integrator
    torus_gt:         np.ndarray   # (T, 3) ground-truth torus coordinates
    theta_hist:       np.ndarray   # (T, 3) decoded torus positions

    # Filter diagnostics
    n_hat_hist:       np.ndarray   # (T, 3) MAP plane normal over time
    gap_hist:         np.ndarray   # (T,)   z2 - z1 eigenvalue gap

    # CAN states (only if record=True), heavy
    S_tot_buffer:     np.ndarray   # (T, N) population vectors, or None
    bingham_snapshots: list        # list[BinghamDistribution], or None

    # Metrics — computed once, stored
    norm_error:       np.ndarray   # (T,) MADE normalised error
    mean_norm_error:  float        # scalar summary

    # Metadata
    condition:        str          # "arena_2d" | "volumetric_3d" | "plane_switching"
    params:           dict         # all hyperparameters used


class BaseExperiment:
    def __init__(self, qan, integrator_kwargs, record=True):
        self.qan = qan
        self.integrator_kwargs = integrator_kwargs  # kappa, alpha, dt, scale
        self.record = record

    def run(self, world_pos, v_body_seq, torus_gt, g_vec) -> ExperimentResult:
        """
        Standard run loop. Called by each condition.
        """
        integrator = PathIntegrator(qan=self.qan, **self.integrator_kwargs)
        integrator.reset(torus_gt[0])
        #Let the CAN stabalize a little bit, nad the filter converge 
        integrator.warmup(n_steps=100)
        
        theta_hist = integrator.run(v_body_seq, g_vec, record=self.record)

        gap = np.array(integrator.history["z2"]) - np.array(integrator.history["z1"])
        n_hat_hist = np.array(integrator.history["n_hat"])

        norm_error = self._made_metric(theta_hist, torus_gt, world_pos=world_pos)
        
        self.last_integrator = integrator 

        return ExperimentResult(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            torus_gt=torus_gt,
            theta_hist=theta_hist,
            n_hat_hist=n_hat_hist,
            gap_hist=gap,
            S_tot_buffer=integrator.S_tot_buffer,
            bingham_snapshots=integrator.bingham_snapshots,
            norm_error=norm_error,
            mean_norm_error=float(norm_error[1:].mean()),
            condition="",      # filled by subclass
            params={},         # filled by subclass
        )

    def save(self, result: ExperimentResult, path: str):
        """
        Save to .npz. S_tot_buffer and bingham_snapshots are optional,
        skips them if None to keep files manageable.
        """
        arrays = {
            "world_pos":    result.world_pos,
            "v_body_seq":   result.v_body_seq,
            "torus_gt":     result.torus_gt,
            "theta_hist":   result.theta_hist,
            "n_hat_hist":   result.n_hat_hist,
            "gap_hist":     result.gap_hist,
            "norm_error":   result.norm_error,
        }
        if result.S_tot_buffer is not None:
            arrays["S_tot_buffer"] = result.S_tot_buffer
        np.savez(path, **arrays)
        
        if result.bingham_snapshots is not None:
            bing_path = path.replace(".npz", "_bingham.pkl")
            with open(bing_path, "wb") as f:
                pickle.dump(result.bingham_snapshots, f)
        # bingham_snapshots are Python objects. Save separately with pickle if needed

    @staticmethod
    def _made_metric(decoded, ground_truth, world_pos=None):
        #If physical positions are available, compute step lengths in metres.
        if world_pos is not None:
            #Euclidean length of each displacement vector
            steps = np.linalg.norm(np.diff(world_pos, axis=0), axis=1)  # metres
        #Fallback when no physical positions are available
        else:
            steps = np.linalg.norm(
                wrapped_angle_diff(ground_truth[1:], ground_truth[:-1]), axis=1
            )
        traj_length = np.cumsum(steps)
        err = np.linalg.norm(wrapped_angle_diff(decoded, ground_truth), axis=1)
        norm_error = err[1:] / (traj_length + 1e-9)
        return np.concatenate([[0.0], norm_error])
    
@dataclass
class BaseConfig:
    env_size: float = 2.0
    n_steps:  int   = 3000
    seed:     int   = 0
    kappa:    float = 10.0
    alpha:    float = 0.999
    dt:       float = 1.0
    target_speed_rad : float = 0.01 #how fast does the simulated animal walk on the torus
    speed:    float = field(init=False)   # derived in __post_init__

    def __post_init__(self):
        self.speed = self.target_speed_rad  * self.env_size / (2 * np.pi)