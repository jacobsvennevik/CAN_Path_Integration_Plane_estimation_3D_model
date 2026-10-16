from dataclasses import dataclass

import numpy as np
from config import ExperimentConfig
from experiments.base import (
    BaseExperiment, Trajectory, movement_plane_normals, torus_gt_from_velocity,
)


@dataclass
class Arena3DConfig(ExperimentConfig):
    ratemap_n_sub: int = 300
    ratemap_n_shuffle: int = 20


class Arena3DExperiment(BaseExperiment):
    condition_label = "arena_3d"
    ratemap_ndim = 3

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
        #save the times we hit a wall
        wall_hits = np.zeros((w.n_steps, 3), dtype=np.int8)

        direction = w.rng.normal(size=3)
        direction /= np.linalg.norm(direction)            # random initial unit heading
 
        for t in range(1, w.n_steps):
            direction = direction + w.rng.normal(0, w.step_turn_std, size=3)
            direction /= np.linalg.norm(direction)        # diffuse heading on the sphere
            v = w.world_speed * direction
            new_pos = world_pos[t - 1] + v
            for dim in range(3):                          # reflect in x, y AND z
                if new_pos[dim] > w.limit or new_pos[dim] < -w.limit:
                    wall_hits[t, dim] = 1 if new_pos[dim] > w.limit else -1
                    direction[dim] = -direction[dim]      # specular bounce
                    v = w.world_speed * direction
                    new_pos = world_pos[t - 1] + v
            world_pos[t]  = new_pos
            v_body_seq[t] = v

        d = self.qan.manifold.dim
        # Plane of this step and the next, normal against gravity. Not the horizontal floor.
        n_true = movement_plane_normals(v_body_seq)
        return Trajectory(
            world_pos=world_pos,
            v_body_seq=v_body_seq,
            torus_gt=torus_gt_from_velocity(v_body_seq, n_true, w.scale, d),
            n_true_seq=n_true,
            wall_hits=wall_hits,
        )
    
    