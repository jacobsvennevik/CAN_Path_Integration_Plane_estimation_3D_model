from dataclasses import dataclass, field
import numpy as np

@dataclass
class NetworkConfig:
    """Everything Torus3DQAN needs."""
    spacing:          float = 0.1    # radians between neighboring neurons on the torus (resolution)
    # shap of the bump
    kernel_alpha:    float = 0.498687    # kernel amplitude
    sigma:            float = 1.347771   # kernel width
    
    #movement of the bump
    b:                float = 0.83   #Positive global exitasion to the whole network
    offset_magnitude: float = 0.349579 #kernel offset: the ±shift applied to each CAN's connectivity
    
    #flag related to building dense numoy matricies or skipping that
    build_connectivity: bool  = True 
@dataclass
class ExperimentConfig:
    """Environment + Bingham filter."""
    env_size:         float = 2.0    # metres: size of the box the animal walks in (walk boundaries)
    n_steps:          int   = 3000 #Defult timesteps
    seed:             int   = 0
    kappa:            float = 10.0 #Bingham filter measurement strength:
    bingham_decay:    float = 0.999  #decay of certinity of the bingham filter
    grid_spacing:     float = 0.48    # metres per full 2π wrap = the torus period
    target_speed_rad: float = 0.01   # desired bump speed, rad/step (the thing held fixed)
    record_stride: int = 20 #How many recordings
    ratemap_bins: int = 40
    speed:            float = field(init=False)
    scale:            float = field(init=False)
    
    def __post_init__(self):
        self.scale = (2 * np.pi) / self.grid_spacing      # the metres→radians conversion from the world manifold to the tours manifold and visa versa
        self.speed = self.target_speed_rad / self.scale   #the physical walk speed of the animal in the arena
    def m_to_rad(self, x_m):   return x_m * self.scale
    def rad_to_m(self, x_rad): return x_rad / self.scale


class AnalysisConfig:
    """Offline scoring parameters — single source for the scoring pipeline."""
    bins:            int   = 40     # histogram bins/axis for rate map + autocorrelogram
    smooth_sigma:    float = 1.75   # gaussian_filter sigma, in BINS (see note below)
    autocorr_th:     float = 0.1    # autocorrelation zeroing threshold
    n_neuron:        int   = 300    # default cells subsampled by score_run
    go_az_precision: int   = 48     # global-order azimuth samples
    go_al_precision: int   = 24     # global-order altitude samples

@dataclass
class RunConfig:
    network:    NetworkConfig    = field(default_factory=NetworkConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    # no spacing/grid_spacing assert: different unit systems, no reason to match
    
def world_to_flat_bins(world_pos, env_size, bins):
    half = env_size / 2.0
    xy = world_pos[:, :2] / half
    ij = np.clip(np.floor((xy + 1.0) * 0.5 * bins).astype(np.int64), 0, bins - 1)
    return ij[:, 0] * bins + ij[:, 1]