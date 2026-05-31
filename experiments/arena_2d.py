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
    def __init__(self, config: RunConfig, record=True):
        # QAN is built HERE, from config.network — not in the notebook
        net = config.network
        qan = Torus3DQAN(
            spacing=net.spacing,
            alpha=net.kernel_alpha,        # rename boundary: QAN still says 'alpha'
            sigma=net.sigma,
            b=net.b,
            offset_magnitude=net.offset_magnitude,
            build_connectivity=net.build_connectivity,
        )
        integrator_kwargs = dict(
            kappa=config.experiment.kappa,
            alpha=config.experiment.bingham_decay,   # Bingham, not network
            scale=config.experiment.scale, 
        )
        super().__init__(qan, integrator_kwargs, record,
                 record_stride=config.experiment.record_stride)
        self.config = config


    def generate_trajectory(self):
        """
        Random walk in physical 2D space.
        Returns world_pos, v_body_seq, torus_gt, scale.
        """
        cfg   = self.config.experiment          # was: self.config
        rng   = np.random.default_rng(cfg.seed) # was: self.config.seed
        scale = cfg.scale                        # was: self.scale (no longer exists)


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

    def run_experiment(self, g):
        world_pos, v_body_seq, torus_gt = self.generate_trajectory()
        result = self.run(world_pos, v_body_seq, torus_gt, g)
        result.condition = "arena_2d"
        result.params = asdict(self.config)
        return result