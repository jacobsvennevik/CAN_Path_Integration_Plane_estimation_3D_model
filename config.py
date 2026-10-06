from dataclasses import dataclass, field, fields
import json
import numpy as np


@dataclass
class NetworkConfig:
    """
    Sets up the network for the experiment: Network + Movement + Integration also a flag realted to memory

    Defaults are a provisional live network, NOTE: might need to change
    """
    spacing:            float = 0.1  # the spacing of neurons on the manifold
    dim:                int   = 2  # T^d the dimension of the torus manifold
    # provisional live network, not the step-2 selection
    lambda_net:         float = 1.28
    ratio:              float = 1.05  # B&F gamma/beta
    alpha:              float = 1.806  # kernel scaling factor; realised gain ~1.7 at this point

    #movement of the bump
    b:                  float = 1.0 #Positive global exitasion to the whole network
    offset_magnitude:   float = 0.196 # How much the neron sheet is shifted

    #integration
    dt:                 float = 0.25 #forward-Euler step size, same units as tau
    tau:                float = 5.0  # neural time constant; dt/tau ≤ 0.125 at tau=2  NOTE: tau is 5.0 now, not 2
    velocity_gain:      float = 2.626  # bump speed vs commanded speed; Optuna searches this in the 2-D sweep

    # Field spacing on the sheet (cells), not Fourier wavelength.
    bump_spacing_cells: float = 19.8

    # Optional undriven settle. None keeps the formula in BaseExperiment.run.
    # Frozen networks (trial 69) set this explicitly; do not default to 4000.
    settle_steps:       int | None = None

    #flag related to building dense numoy matricies or skipping that
    build_connectivity: bool  = False


@dataclass
class PlaneConfig:
    """Estimator readout: vMF filter, hold, and which normal drives E."""
    kappa:           float = 10.0  # likelihood concentration for the vMF update
    kappa_w:         float = 1e4
    kappa_sens:      float = 300.0
    measurement:     str   = "sensed"   # "sensed", motion could be implemented later
    plane_mode:      str   = "bayesian"  # "bayesian" (default) or "true"
    tau_refresh:     int   = 1
    refresh_timing:  str   = "free"     # harness
    refresh_settle_steps: int = 200
    reset_prior_at_segment_start: bool = False
    seed:            int   = 0


@dataclass
class ExperimentConfig:
    """Sets up the enviorment for the experiment: Environment
    """
    env_size:         float = 2.0    # metres: size of the box the animal walks in (walk boundaries)
    n_steps:          int   = 3000 #Defult timesteps
    seed:             int   = 0
    grid_spacing:     float = 0.48    # metres per bump spacing 
    target_speed_rad_per_time: float = 0.002  # desired bump speed

    omega_std:        float = 0.03
    #NOTE: How many of this is needed, look at this one more time then
    record_stride:    int   = 20 #How many recordings
    record:           bool  = True
    ratemap_bins:     int   = 40
    ratemap_ndim:     int   = 2
    ratemap_n_sub:    int   = 0
    ratemap_n_shuffle: int  = 0
    ratemap_seed:     int   = 0
    ratemap_active_thresh: float = 1e-3
    scale:            float = field(init=False)

    def __post_init__(self):
        # Fallback until RunConfig overwrites
        self.scale = (2 * np.pi) / self.grid_spacing      # the metres→radians conversion from the world manifold to the tours manifold and visa versa  NOTE: fallback only; RunConfig overwrites with G

    def m_to_rad(self, x_m):   return x_m * self.scale
    def rad_to_m(self, x_rad): return x_rad / self.scale


@dataclass
class AnalysisConfig:
    """Offline scoring parameters, read by everything under analysis/."""
    bins:            int   = 40     # histogram bins/axis for rate map + autocorrelogram
    smooth_sigma:    float = 1.75   # gaussian_filter sigma, in BINS (see note below)
    n_neuron:        int   = 300    # default cells subsampled for a volumetric score
    go_az_precision: int   = 48     # global-order azimuth samples
    go_al_precision: int   = 24     # global-order altitude samples

    # These three numbers limit the saved trace of network activity.
    field_stride:       int = 10
    n_traced_cells:     int = 600
    max_trace_bytes:    int = 1024 ** 3


@dataclass
class RunConfig:
    network:    NetworkConfig    = field(default_factory=NetworkConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    plane:      PlaneConfig      = field(default_factory=PlaneConfig)
    analysis:   AnalysisConfig   = field(default_factory=AnalysisConfig)
    # no spacing/grid_spacing assert: different unit systems, no reason to match

    def __post_init__(self):
        n = int(np.ceil(2 * np.pi / self.network.spacing))
        self.experiment.scale = float(
            2 * np.pi * self.network.bump_spacing_cells
            / (n * self.experiment.grid_spacing)
        )





def world_to_normalized(world_pos, env_size):
    """Map physical position (metres, within ±env_size/2) to the [-1,1] cube
    that every scorer assumes. Arena-anchored: the ruler is the wall, not the
    trajectory, so grid SPACING stays physically meaningful across runs."""
    half = env_size / 2.0
    return np.clip(world_pos / half, -1.0, 1.0)


def world_to_flat_bins(world_pos, env_size, bins, ndim=2):
    """Answers which flat bin does each position fall in.

    Only the leading `ndim` columns are used, which is how the 2-D arena drops z
    from its (T, 3) positions.
    """
    x = world_to_normalized(world_pos[:, :ndim], env_size)  # shared normalize; index-clip below handles bounds
    idx = np.clip(np.floor((x + 1.0) * 0.5 * bins).astype(np.int64), 0, bins - 1)
    flat = idx[:, 0]
    for d in range(1, ndim):
        flat = flat * bins + idx[:, d]
    return flat


def world_to_flat_bins_3d(world_pos, env_size, bins):
    """The 3-D case, for call sites that ask for it by name."""
    return world_to_flat_bins(world_pos, env_size, bins, ndim=3)
