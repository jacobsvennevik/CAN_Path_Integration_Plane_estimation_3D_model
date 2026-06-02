from made.can import CAN, relu, Kernel 
from dataclasses import dataclass
import numpy as np


@dataclass
class CAN3D(CAN):
    """CAN with tunable feedforward drive b. Might not have much of a difference

    Inherits all behavior from CAN, but change step_stateless to be able to tune b 

    Attributes:
        b (float): Constant feedforward excitatory drive.
        build_connectivity: bool = True dense vs torch built matrix memory 
    """
    b: float = 1.0
    build_connectivity: bool = True 
    
    def __post_init__(self):
        if self.build_connectivity:
            super().__post_init__()                 # MADE's normal dense path, unchanged
        else:
            # cheap setup only -- the torch backend reads neurons_coordinates and S.
            self.kernel = Kernel(self.alpha, self.sigma)
            self.neurons_coordinates = (
                self.manifold.parameter_space.sample_with_spacing(self.spacing)
            )
            self.S = np.zeros((self.neurons_coordinates.shape[0], 1))

    def step_stateless(self, S, u=0):
        """
        Override function

        """
        if not hasattr(self, "connectivity_matrix"):
            raise AttributeError(
                "step_stateless needs connectivity_matrix, which was skipped "
                "(build_connectivity=False). Use the torch FFT backend for fine-spacing runs."
            )
        S_dot = self.connectivity_matrix @ S + u + self.b
        new_S = S + (relu(S_dot) - S) / self.tau

        if np.any(np.isnan(new_S)):
            raise ValueError(f"NaN values detected in new state.")

        return new_S
    
    @property
    def weight_matrix(self) -> np.ndarray:
        if not hasattr(self, "connectivity_matrix"):
            raise AttributeError(
                "connectivity_matrix was not built (build_connectivity=False). "
                "Use the torch FFT backend (TorchBackend), which never needs the dense matrix."
            )
        return self.connectivity_matrix
