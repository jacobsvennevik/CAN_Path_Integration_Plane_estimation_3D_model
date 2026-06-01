import os
import json
import pickle
from dataclasses import dataclass

import numpy as np

from metrics import wrapped_angle_diff
from path_integration import PathIntegrator
from config import world_to_flat_bins


@dataclass
class ExperimentResult:
    world_pos:         np.ndarray
    v_body_seq:        np.ndarray
    torus_gt:          np.ndarray
    theta_hist:        np.ndarray
    n_hat_hist:        np.ndarray
    gap_hist:          np.ndarray
    S_tot_buffer:      np.ndarray   # or None
    bingham_snapshots: list         # or None
    norm_error:        np.ndarray
    mean_norm_error:   float
    condition:         str
    params:            dict
    ratemap_sums:      np.ndarray = None   
    ratemap_counts:    np.ndarray = None


class BaseExperiment:
    def __init__(self, qan, integrator_kwargs, record=True, record_stride=1):
        self.qan = qan
        self.integrator_kwargs = integrator_kwargs  # kappa, alpha, scale
        self.record = record
        self.record_stride = record_stride
        

    def run(self, world_pos, v_body_seq, torus_gt, g_vec) -> ExperimentResult:
        bins = self.config.experiment.ratemap_bins
        flat = world_to_flat_bins(world_pos, self.config.experiment.env_size, bins)
        
        integrator = PathIntegrator(qan=self.qan, **self.integrator_kwargs)
        integrator.reset(torus_gt[0])
        integrator.warmup(n_steps=100)   # let CAN stabilize, filter converge

        theta_hist = integrator.run(
            v_body_seq, g_vec, 
            record=self.record,
            flat_indices=flat,         
            ratemap_bins=bins,     
            )

        gap = np.array(integrator.history["z2"]) - np.array(integrator.history["z1"])
        n_hat_hist = np.array(integrator.history["n_hat"])
        norm_error = self._made_metric(theta_hist, torus_gt, world_pos=world_pos)

        self.last_integrator = integrator

        return ExperimentResult(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            ratemap_sums=integrator.ratemap_sums,       
            ratemap_counts=integrator.ratemap_counts,
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
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stride = (len(result.world_pos) // result.S_tot_buffer.shape[0]
              if hasattr(result, "S_tot_buffer") and result.S_tot_buffer is not None
              else 1)
        arrays = {
            "world_pos":     result.world_pos[::stride],   # strided to match S_tot_buffer
            "torus_gt":      result.torus_gt[::stride],
            "v_body_seq":    result.v_body_seq,
            "theta_hist":    result.theta_hist,
            "n_hat_hist":    result.n_hat_hist,
            "gap_hist":      result.gap_hist,
            "norm_error":    result.norm_error,
            "record_stride": np.int32(stride),
        }
        if result.S_tot_buffer is not None:
            arrays["S_tot_buffer"] = result.S_tot_buffer   # only added when recording
        np.savez(path, **arrays)

        # persist the full config so the run is reproducible
        with open(path.replace(".npz", "_config.json"), "w") as f:
            json.dump({"condition": result.condition, "params": result.params},
                      f, indent=2)

        if result.bingham_snapshots is not None:
            with open(path.replace(".npz", "_bingham.pkl"), "wb") as f:
                pickle.dump(result.bingham_snapshots, f)

    @staticmethod
    def _made_metric(decoded, ground_truth, world_pos=None):
        if world_pos is not None:
            steps = np.linalg.norm(np.diff(world_pos, axis=0), axis=1)  # metres
        else:
            steps = np.linalg.norm(
                wrapped_angle_diff(ground_truth[1:], ground_truth[:-1]), axis=1
            )
        traj_length = np.cumsum(steps)
        err = np.linalg.norm(wrapped_angle_diff(decoded, ground_truth), axis=1)
        norm_error = err[1:] / (traj_length + 1e-9)
        return np.concatenate([[0.0], norm_error])
    
    