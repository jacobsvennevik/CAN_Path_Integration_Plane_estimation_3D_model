from dataclasses import dataclass

import numpy as np
from config import ExperimentConfig
from experiments.base import BaseExperiment, Trajectory, world_to_torus_gt


@dataclass
class Arena3DConfig(ExperimentConfig):
    ratemap_ndim: int = 3
    ratemap_n_sub: int = 300
    ratemap_n_shuffle: int = 20

    @property
    def run_name(self) -> str:
        return f"arena3d_T{self.n_steps}_seed{self.seed}_envSize{self.env_size:}_gridSpacing{self.grid_spacing}"


class Arena3DExperiment(BaseExperiment):
    condition_label = "arena_3d"

    def __init__(self, config, record=None, plane_mode=None):
        assert config.experiment.ratemap_ndim == 3
        super().__init__(config, record=record, plane_mode=plane_mode)
        # plane_mode: flip the filter on/off when given; otherwise PlaneConfig.plane_mode
        

    def generate_trajectory(self, turn_std: float = None, n_steps=None, seed=None):
        """Independent 3D reflecting walk. n_steps and seed default to config.

        Random walk in physical 3D space.
        Returns world_pos (sequence of positions), velocity_body_seq (sequence of speeds),
        torus_gt (sequence of positions on the torus manifold).
        Basicly turns the physical wlak in the box into ground truth for the torus manifold.
        """
        w = self._walk_params(turn_std, n_steps, seed)

        world_pos  = np.zeros((w.n_steps, 3))
        v_body_seq = np.zeros((w.n_steps, 3))
 
        direction = w.rng.normal(size=3)
        direction /= np.linalg.norm(direction)            # random initial unit heading
 
        for t in range(1, w.n_steps):
            direction = direction + w.rng.normal(0, w.step_turn_std, size=3)
            direction /= np.linalg.norm(direction)        # diffuse heading on the sphere
            v = w.world_speed * direction
            new_pos = world_pos[t - 1] + v
            for dim in range(3):                          # reflect in x, y AND z
                if new_pos[dim] > w.limit or new_pos[dim] < -w.limit:
                    direction[dim] = -direction[dim]      # specular bounce
                    v = w.world_speed * direction
                    new_pos = world_pos[t - 1] + v
            world_pos[t]  = new_pos
            v_body_seq[t] = v

        d = self.qan.manifold.dim
        n_true = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (w.n_steps, 3))
        return Trajectory(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            torus_gt=world_to_torus_gt(
                world_pos[:, :d],#keep as many axis as the the manifold has
                w.scale),
            n_true_seq=n_true,
        )
    
    