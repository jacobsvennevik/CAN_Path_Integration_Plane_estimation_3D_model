from dataclasses import dataclass, field
import numpy as np

@dataclass
class NetworkConfig:
    """Everything Torus3DQAN needs."""
    spacing:          float = 0.3    # radians between neighboring neurons on the torus (resolution)
    kernel_alpha:    float = 1.5    # kernel amplitude
    sigma:            float = 1.4    # kernel width
    b:                float = 0.83   #Positive global exitasion to the whole network
    velocity_gains:   float = 20 #how fast the animal is moving into how hard to push the bump. I
    offset_magnitude: float = 0.25 #kernel offset: the ±shift applied to each CAN's connectivity

@dataclass
class ExperimentConfig:
    """Environment + Bingham filter."""
    env_size:         float = 2.0    # metres: size of the box the animal walks in (walk boundaries)
    n_steps:          int   = 3000 #Defult timesteps
    seed:             int   = 0
    kappa:            float = 10.0 #Bingham filter measurement strength:
    bingham_decay:    float = 0.999  #decay of certinity of the bingham filter
    grid_spacing:     float = 0.3    # metres per full 2π wrap = the torus period
    target_speed_rad: float = 0.01   # desired bump speed, rad/step (the thing held fixed)
    speed:            float = field(init=False)
    scale:            float = field(init=False)

    def __post_init__(self):
        self.scale = (2 * np.pi) / self.grid_spacing      # the metres→radians conversion from the world manifold to the tours manifold and visa versa
        self.speed = self.target_speed_rad / self.scale   #the physical walk speed of the animal in the arena
    def m_to_rad(self, x_m):   return x_m * self.scale
    def rad_to_m(self, x_rad): return x_rad / self.scale

@dataclass
class RunConfig:
    network:    NetworkConfig    = field(default_factory=NetworkConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    # no spacing/grid_spacing assert: different unit systems, no reason to match