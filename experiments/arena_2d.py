from dataclasses import dataclass

import numpy as np
from config import ExperimentConfig
from experiments.base import (
    BaseExperiment, Trajectory, planar_walk, world_to_torus_gt,
)


@dataclass
class Arena2DConfig(ExperimentConfig):
    ratemap_n_sub: int = 300
    ratemap_n_shuffle: int = 50

    @property
    def run_name(self) -> str:
        return f"arena2d_T{self.n_steps}_seed{self.seed}_envSize{self.env_size:}_gridSpacing{self.grid_spacing}"


class Arena2DExperiment(BaseExperiment):

    condition_label = "arena_2d"

    def generate_trajectory(self, turn_std: float = None, n_steps=None, seed=None):
        """
        Random walk in physical 2D space.
        Returns world_pos (sequence of positions), velocity_body_seq (sequence of speeds),
        torus_gt (sequence of positions on the torus manifold).
        Basicly turns the physical wlak in the box into ground truth for the torus manifold.

        Flat-floor walk: walk computed in 2-D plane then embeeded with basis E into 3-D space.
        """
        w = self._walk_params(turn_std, n_steps, seed)
        uv, duv = planar_walk(w.rng, w.n_steps, w.world_speed, w.step_turn_std, w.limit)
        E = np.eye(3)[:, :2]
        world_pos = uv @ E.T
        v_body_seq = duv @ E.T

        d = self.qan.manifold.dim
        n_true = np.broadcast_to(np.array([0.0, 0.0, 1.0]), (w.n_steps, 3))
        return Trajectory(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            torus_gt=world_to_torus_gt(
                world_pos[:, :d], #keep as many axis as the the manifold has
                w.scale),
            n_true_seq=n_true,
        )
