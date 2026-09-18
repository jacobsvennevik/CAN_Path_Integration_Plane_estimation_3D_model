from made.manifolds import AbstractManifold, ParameterSpace, Range
from made.metrics import Metric, PeriodicEuclidean
from dataclasses import dataclass, field
import numpy as np


class ParameterSpaceND(ParameterSpace):
    """Extends ParameterSpace to any d, sampled on an explicit C-order ('ij') lattice."""

    def _meshgrid_columns(self, per_axis_counts: list[int],
                          pads: list[float]) -> np.ndarray:
        """Places points evenly along the axises of the manifold and combines them into
        theta_1, ..., theta_d coordinates.
        That is where each neuron sits.
        """
        assert len(pads) == self.dim, (
            f"Incorrect number of pads for manifold dimension: got {len(pads)}, "
            f"expected {self.dim}")
        axes = [r.sample(count, pad)
                for r, count, pad in zip(self.ranges, per_axis_counts, pads)]
        G = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([g.ravel() for g in G])

    def sample(self, n: int, pads: list[float] = None) -> np.ndarray:
        """
        Returns points sampled from the parameter space.
        Used for visualisation.
        Returns n^d points as an (n^d, d) array.

        Args:
            n (int): Number of points to sample per axis
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim
        return self._meshgrid_columns([n] * self.dim, pads)

    def sample_with_spacing(
        self, spacing: float, pads: list[float] = None
    ) -> np.ndarray:
        """
        Returns points sampled from the parameter space with a fixed spacing.
        Used for neuron placement in the CAN.
        Returns an (n^d, d) array, n set by the spacing:
        n per axis is ceil((end - start) / spacing); on [0, 2π] that is
        ceil(2π / spacing).

        Args:
            spacing (float): Fixed spacing between points
            pads (list[float]): Padding from range boundaries (default: 0.0)

        Returns:
            np.ndarray: Array of sampled points
        """
        if pads is None:
            pads = [0.0] * self.dim
        # neurons per dimension, from each range's own extent
        counts = [int(np.ceil((r.end - r.start) / spacing)) for r in self.ranges]
        return self._meshgrid_columns(counts, pads)


@dataclass
class TorusND(AbstractManifold):
    """
    New manifold structure, d-dimensional torus T^d (T^3 at dim=3).

    All dimensions are periodic, representing the possible axises of movement.
    Topology: T^d = S^1 x ... x S^1. Sheet size is set by CAN spacing.
    """
    dim: int = 2
    parameter_space: ParameterSpaceND = field(init=False)
    metric: Metric = field(init=False)

    def __post_init__(self):
        self.dim = int(self.dim)
        # Periodic dimensions, each with period 2*pi.
        # Points that wrap around, they are periodic,
        # giving the topology of a d-torus T^d.
        self.parameter_space = ParameterSpaceND(
            [Range(0, 2 * np.pi, periodic=True) for _ in range(self.dim)]  # [0, Li]
        )
        # Computes shortest wrap-around distance between two points for all dimensions.
        # MADE's PeriodicEuclidean is correct for any d at period 2π.
        self.metric = PeriodicEuclidean(
            dim=self.dim, periodic=[True] * self.dim
        )
