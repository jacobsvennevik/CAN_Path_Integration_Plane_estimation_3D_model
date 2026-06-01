import numpy as np
from dataclasses import asdict
from experiments.arena_3d import BaseExperiment


class Arena3DExperiment(BaseExperiment):
    condition_label = "arena_3d"
    
    def __init__(self, config, record=True, plane_mode="true"):
        super().__init__(config, record)                  # QAN + integrator_kwargs from Arena2D
        self.integrator_kwargs["plane_mode"] = plane_mode  # flip the filter on/off
        

    def generate_trajectory(self, turn_std: float = 0.1):
        cfg   = self.config.experiment
        rng   = np.random.default_rng(cfg.seed)
        scale = cfg.scale
        limit = cfg.env_size / 2
 
        world_pos  = np.zeros((cfg.n_steps, 3))
        v_body_seq = np.zeros((cfg.n_steps, 3))
 
        direction = rng.normal(size=3)
        direction /= np.linalg.norm(direction)            # random initial unit heading
 
        for t in range(1, cfg.n_steps):
            direction = direction + rng.normal(0, turn_std, size=3)
            direction /= np.linalg.norm(direction)        # diffuse heading on the sphere
            v = cfg.speed * direction
            new_pos = world_pos[t - 1] + v
            for dim in range(3):                          # reflect in x, y AND z
                if new_pos[dim] > limit or new_pos[dim] < -limit:
                    direction[dim] = -direction[dim]      # specular bounce
                    v = cfg.speed * direction
                    new_pos = world_pos[t - 1] + v
            world_pos[t]  = new_pos
            v_body_seq[t] = v
 
        metric = self.qan.manifold.metric
        torus_gt = np.zeros((cfg.n_steps, 3))
        phased = metric.to_phase(world_pos[:, :2] * scale)            # hex shear, x & y only
        torus_gt[:, 0] = (np.pi + phased[:, 0]) % (2 * np.pi)
        torus_gt[:, 1] = (np.pi + phased[:, 1]) % (2 * np.pi)
        torus_gt[:, 2] = (np.pi + world_pos[:, 2] * scale) % (2 * np.pi)  # z is columnar not periodic TODO: watch out for this
        return world_pos, v_body_seq, torus_gt