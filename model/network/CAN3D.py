from made.can import CAN
from dataclasses import dataclass, field
import numpy as np

def torus_grid(n: int, d: int = 3) -> np.ndarray:
    """Coordinates of every neuron on an n^d."""
    n = int(n)
    d = int(d)
    return (np.indices((n,) * d).reshape(d, -1).T) * (2 * np.pi / n)


def kernel_field_on_grid(kernel, metric, n: int, offset=None,
                         grid: np.ndarray = None, d: int = None) -> np.ndarray:
    """Connection strength from one neuron to every neuron, as an n^d volume.

    Measures the torus distance from every lattice point to the offset, then
    passes those distances through the kernel``.


    Args:
        kernel: maps distances to weights, e.g. a ``Kernel_BF``.
        metric: called as ``metric(points, centre)``; handles the wrap-around.
        n: neurons per axis.
        offset: where to centre the kernel. Each of the QAN's six CANs (2d at
            general d) shifts it along one axis. None means the origin.
        grid: a precomputed torus_grid(n, d`, if you already have one.
        d: manifold dimension. Inferred from grid when provided, else 3.
    """
    #Create the grid if it is not already created
    n = int(n)
    if grid is None:
        d = 3 if d is None else int(d)
        grid = torus_grid(n, d)
    else:
        d = int(grid.shape[1])
    centre = np.zeros((1, d))
    
    if offset is not None:
        centre[0] = offset
    return kernel(metric(grid, centre).reshape((n,) * d))


@dataclass(kw_only=True)
class CAN3D(CAN):
    """CAN with tunable feedforward drive b.

    b, kernel and dt come from NetworkConfig, by way of the QAN that builds this.
    The recurrent weights are applied by the FFT backend, not a dense matrix.
    """
    b: float
    kernel: object
    dt: float

    def __post_init__(self):
        self.neurons_coordinates = (
            self.manifold.parameter_space.sample_with_spacing(self.spacing)
        )
        self.S = np.zeros((self.neurons_coordinates.shape[0], 1))


@dataclass
class Kernel_BF:
    """Center-surround Kernel by Burak and Fiete, difference of Gaussians recurrent kernel.

    sigma_e < sigma_i gives a narrow excitatory peak minus a broader inhibitory surround.

    Attributes:
        alpha (float): Overall gain of the kernel.
        lambda_net (float): Bump spacing wavelength, in radians.
        ratio (float): How much narrower the excitatory Gaussian is than the inhibitory.
    """

    lambda_net: float #How far apart the bumps end up, in radians
    ratio:      float #How much narrower the excitatory Gaussian is than the inhibitory.
    alpha:      float = 1.0 # Overall gain of the whole kernel, scales everything up
                            # or down uniformly. This is for different network sizes.

    sigma_i: float = field(init=False) #Width of the broad, inhibitory Gaussian
    sigma_e: float = field(init=False) # Width of the narrow, excitatory Gaussian

    def __post_init__(self):
        self.sigma_i = self.lambda_net / np.sqrt(6.0)
        self.sigma_e = self.sigma_i / np.sqrt(self.ratio)

    def __call__(self, d):
        """Applies the kernel"""
        d2 = np.asarray(d, dtype=float) ** 2
        k = (np.exp(-d2 / (2 * self.sigma_e ** 2))
                    - np.exp(-d2 / (2 * self.sigma_i ** 2)))
        return self.alpha * k
