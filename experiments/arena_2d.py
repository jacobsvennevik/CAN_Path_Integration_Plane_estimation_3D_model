from dataclasses import dataclass

import numpy as np
from config import ExperimentConfig
from experiments.base import (
    BaseExperiment, Trajectory, movement_plane_normals, planar_walk,
    torus_gt_from_velocity,
)


@dataclass
class Arena2DConfig(ExperimentConfig):
    ratemap_n_sub: int = 300
    ratemap_n_shuffle: int = 50


class Arena2DExperiment(BaseExperiment):

    condition_label = "arena_2d"
    ratemap_ndim = 2

    def generate_trajectory(self, turn_std: float = None, n_steps=None, seed=None):
        """
        Random walk in physical 2D space.
        Returns world_pos (sequence of positions), velocity_body_seq (sequence of speeds),
        torus_gt (sequence of positions on the torus manifold).
        Basicly turns the physical wlak in the box into ground truth for the torus manifold.

        Flat-floor walk: walk computed in 2-D plane then embedded with basis E into 3-D space.
        """
        w = self._walk_params(turn_std, n_steps, seed)
        uv, duv = planar_walk(w.rng, w.n_steps, w.world_speed, w.step_turn_std, w.limit)
        E = np.eye(3)[:, :2]
        world_pos = uv @ E.T
        v_body_seq = duv @ E.T

        d = self.qan.manifold.dim
        n_true = movement_plane_normals(v_body_seq)
        return Trajectory(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            torus_gt=torus_gt_from_velocity(v_body_seq, n_true, w.scale, d),
            n_true_seq=n_true,
        )
