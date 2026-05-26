from dataclasses import dataclass

import numpy as np

from spatial_nav.experiments.base import BaseExperiment, ExperimentResult
from spatial_nav.path_integration import compute_pi_star_scale
from spatial_nav.experiments.base import BaseConfig

@dataclass
class Arena2DConfig(BaseConfig):
    pass



class Arena2DExperiment(BaseExperiment):

    def __init__(self, qan, config: Arena2DConfig, record=True):
        self.scale = compute_pi_star_scale(config.env_size) #Scale of arena in the real world manifold to the torus manifold, meters to radians
        integrator_kwargs = dict( 
            kappa=config.kappa, #Bingham filter concentration 
            alpha=config.alpha, #diffusion decay applied each step to the Bingham prior, 
            scale=self.scale,
        )
        super().__init__(qan, integrator_kwargs, record)
        self.config = config


    def generate_trajectory(self):
        """
        Random walk in physical 2D space.
        Returns world_pos, v_body_seq, torus_gt, scale.
        """
        rng = np.random.default_rng(self.config.seed)
        cfg = self.config
        scale = self.scale

        #Pre-allocate two arrays of zeroes in 3-dimensions
        world_pos  = np.zeros((cfg.n_steps, 3))
        v_body_seq = np.zeros((cfg.n_steps, 3))
        #persistent random walk
        heading = rng.uniform(0, 2 * np.pi) #random heading
        # Reflect at boundaries ±(env_size/2)
        limit = cfg.env_size / 2
        for t in range(1, cfg.n_steps):
            heading += rng.normal(0, 0.1)          # small turn per step, gaussian
            v = cfg.speed * np.array([np.cos(heading), np.sin(heading), 0.0]) #velocity heading at constant speed
            #update world positon
            new_pos = world_pos[t - 1] + v
            for dim in range(2):  # only x, y for 2D arena
                if new_pos[dim] > limit or new_pos[dim] < -limit:
                    heading = np.pi - heading if dim == 0 else -heading  # reflect
                    v = cfg.speed * np.array([np.cos(heading), np.sin(heading), 0.0]) #Recompute the velocity vector using the reflected heading
                    new_pos = world_pos[t - 1] + v 
            world_pos[t] = new_pos
            v_body_seq[t] = v

        # Switch meters in the real world to radians on the torus, to get the ground truth
        torus_gt = np.zeros((cfg.n_steps, 3))
        torus_gt[:, 0] = (np.pi + world_pos[:, 0] * scale) % (2 * np.pi)
        torus_gt[:, 1] = (np.pi + world_pos[:, 1] * scale) % (2 * np.pi)
        torus_gt[:, 2] = 0.0   # flat arena, θ₃ stays the same

        return world_pos, v_body_seq, torus_gt

    def run_experiment(self, g):
        world_pos, v_body_seq, torus_gt = self.generate_trajectory()
        result = self.run(world_pos, v_body_seq, torus_gt, g)
        result.condition = "arena_2d"
        result.params = {
            "env_size": self.config.env_size,
            "speed":    self.config.speed,
            "n_steps":  self.config.n_steps,
            "kappa":    self.config.kappa,
            "alpha":    self.config.alpha,
            "scale":    self.scale,
        }
        return result