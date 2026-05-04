from dataclasses import dataclass, field
from made.manifolds import AbstractManifold
from made.can import CAN
from made.qan import QAN
from made import manifolds
from spatial_nav.CAN_IMP.CAN3D import CAN3D
from spatial_nav.CAN_IMP import torus3D_manifold
import numpy as np


@dataclass
class Torus3DQAN(QAN):
    """
    QAN for a 3-torus manifold.
    Uses 6 offset CAN3Ds.
    All three angular dimensions are periodic in [0, 2π].
    Inherits behavior from MADE QAN.
    """
    manifold: AbstractManifold = field(
        default_factory=torus3D_manifold.Torus3D
    )
    spacing: float = 0.5
    alpha: float = 0.8
    sigma: float = 1.5
    offset_magnitude: float = 0.25
    beta: float = 1.25e2
    b: float = 1.6  # feedforward drive, passed to CAN3D

    def __post_init__(self):
        """Override to use CAN3D instead of CAN."""
        self.cans = []
        for d in range(self.manifold.dim):      # dim = 3, so 6 CANs total
            for direction in [1, -1]:
                self.cans.append(
                    CAN3D(
                        self.manifold,
                        self.spacing,
                        self.alpha,
                        self.sigma,
                        b=self.b,               # passes tunable b
                        weights_offset=lambda x, d=d, direction=direction: (
                            self.coordinates_offset(
                                x, d, direction, self.offset_magnitude
                            )
                        ),
                    )
                )

    @staticmethod
    def coordinates_offset(
        theta: np.ndarray, dim: int, direction: int, offset_magnitude: float
    ) -> np.ndarray:
        """Offset coordinates along one dimension creaing an asymetric weight matricies, this makes the QAN drift and wrapping modulo 2π to create the periodicity."""
        theta = theta.copy()
        theta[:, dim] += direction * offset_magnitude
        theta[:, dim] = np.mod(theta[:, dim], 2 * np.pi)
        return theta

    def make_trajectory(self, n_steps: int = 1000) -> np.ndarray:
        """Test path generation with incoomensurate rates. This means that trajectories never 
        repeats and thefore will cover T^3."""
        trajectory = np.zeros((n_steps, self.manifold.dim))
        t = np.linspace(0.25, 6 * np.pi - 0.25, n_steps)
        trajectory[:, 0] = np.mod(t, 2 * np.pi)
        trajectory[:, 1] = np.mod(2/3 * t, 2 * np.pi)
        trajectory[:, 2] = np.mod(3/7 * t, 2 * np.pi)  # third incommensurate freq
        return trajectory

    def compute_theta_dot(
        self, theta: np.ndarray, theta_prev: np.ndarray
    ) -> np.ndarray:
        """Does boundary correction for the angular velocity. So that when animal is at a boundary the velocity updates are correct"""
        delta = theta - theta_prev
        for d in range(3):
            if delta[d] > np.pi:
                delta[d] -= 2 * np.pi
            elif delta[d] < -np.pi:
                delta[d] += 2 * np.pi
        return delta

    def compute_can_input(
        self, i: int, theta_dot: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """Maps the velocity to the correct CAN pairing."""
        dim = i // 2          # 0,0 → dim 0 | 1,1 → dim 1 | 2,2 → dim 2
        sign = 1 if i % 2 == 0 else -1
        return sign * self.beta * theta_dot[dim]
    
