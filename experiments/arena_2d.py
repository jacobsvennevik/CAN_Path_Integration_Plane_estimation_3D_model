import numpy as np
from dataclasses import asdict, dataclass
from config import RunConfig, ExperimentConfig
from experiments.base import BaseExperiment
from network.QAN3D import Torus3DQAN

class Arena2DConfig(ExperimentConfig):
    @property
    def run_name(self) -> str:
        return f"arena2d_T{self.n_steps}_kap{self.kappa}_seed{self.seed}"
    

class Arena2DExperiment(BaseExperiment):

    condition_label = "arena_2d"
        
    def generate_trajectory(self):
        """
        Random walk in physical 2D space.
        Returns world_pos, v_body_seq, torus_gt, scale.
        """
        cfg   = self.config.experiment          
        rng   = np.random.default_rng(cfg.seed) 
        scale = cfg.scale                        


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

        
        metric = self.qan.manifold.metric
        # Switch meters in the real world to radians on the torus, to get the ground truth
        torus_gt = np.zeros((cfg.n_steps, 3))
        phased = metric.to_phase(world_pos[:, :2] * scale)    # (T, 2)
        torus_gt[:, 0] = (np.pi + phased[:, 0]) % (2 * np.pi)
        torus_gt[:, 1] = (np.pi + phased[:, 1]) % (2 * np.pi)
        torus_gt[:, 2] = 0.0   # flat arena, θ₃ stays the same

        return world_pos, v_body_seq, torus_gt
    
    def generate_trajectory_(self, rows_per_bin=1):
        """
        Not in use
        """
        cfg   = self.config.experiment
        scale = cfg.scale
        T     = cfg.n_steps
        lim   = cfg.env_size / 2.0
        speed = cfg.speed                                  # metres/step = the cap

        # serpentine waypoints; row spacing <= one bin width so no bin is skipped
        row_dy = (cfg.env_size / cfg.ratemap_bins) / rows_per_bin
        rows   = np.arange(-lim, lim + 1e-9, row_dy)
        up = []
        for i, y in enumerate(rows):
            up += ([[-lim, y], [lim, y]] if i % 2 == 0 else [[lim, y], [-lim, y]])
        up    = np.array(up)
        cycle = np.vstack([up, up[::-1][1:]])              # up then retrace down -> closed loop

        # walk the loop at constant speed; modulo repeats it with no teleport
        s_wp = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(cycle, axis=0), axis=1))])
        s    = (np.arange(T) * speed) % s_wp[-1]
        world_pos       = np.zeros((T, 3))
        world_pos[:, 0] = np.interp(s, s_wp, cycle[:, 0])
        world_pos[:, 1] = np.interp(s, s_wp, cycle[:, 1])

        v_body_seq         = np.zeros((T, 3))
        v_body_seq[1:, :2] = np.diff(world_pos[:, :2], axis=0)

        metric   = self.qan.manifold.metric
        torus_gt = np.zeros((T, 3))
        phased   = metric.to_phase(world_pos[:, :2] * scale)
        torus_gt[:, 0] = (np.pi + phased[:, 0]) % (2 * np.pi)
        torus_gt[:, 1] = (np.pi + phased[:, 1]) % (2 * np.pi)
        torus_gt[:, 2] = 0.0
        return world_pos, v_body_seq, torus_gt
    
