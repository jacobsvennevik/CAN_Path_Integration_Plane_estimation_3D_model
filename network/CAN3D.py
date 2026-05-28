from made.can import CAN, relu
from dataclasses import dataclass
import numpy as np


@dataclass
class CAN3D(CAN):
    """CAN with tunable feedforward drive b. Might not have much of a difference

    Inherits all behavior from CAN, but change step_stateless to be able to tune b 

    Attributes:
        b (float): Constant feedforward excitatory drive.
    """
    b: float = 1.0
    def step_stateless(self, S, u=0):
        """
        Override function

        """
        S_dot = self.connectivity_matrix @ S + u + self.b
        new_S = S + (relu(S_dot) - S) / self.tau

        if np.any(np.isnan(new_S)):
            raise ValueError(f"NaN values detected in new state.")

        return new_S
    
    @property
    def weight_matrix(self) -> np.ndarray:
        return self.connectivity_matrix
