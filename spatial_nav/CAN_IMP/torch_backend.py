"""
Runner for the torch backend, in larger simulations need to use this class.
For ligther runs just use the MADE framework and the QAN3D.py and CAN3D classes.
"""

import gc
import numpy as np
import torch
from dataclasses import dataclass, field


@dataclass
class TorchBackend:
    """
    Torch-backed simulation engine for a Torus3DQAN.
    """

    qan: object                                  # Torus3DQAN instance
    torch_dtype: torch.dtype = torch.float32    #Uses cheeper float32 

    # filled by __post_init__
    device:       torch.device = field(init=False)
    W_torch:      torch.Tensor = field(init=False)
    S_torch:      torch.Tensor = field(init=False)
    coords_torch: torch.Tensor = field(init=False)
    tau_torch:    torch.Tensor = field(init=False)
    beta_torch:   torch.Tensor = field(init=False)
    b_torch:      torch.Tensor = field(init=False)

    def __post_init__(self):
        self.device = self._get_torch_device()
        self._init_torch_backend()

    # I use mps
    def _get_torch_device(self):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")

    def _init_torch_backend(self):
        """
        Starts the torch backend. Takes the NumPy CAN created through the MADE framework.
        Moves the parts we can make fast into torch, to not blow up memory we remove dense
        duplicate matrices.
        """
        cans = self.qan.cans

        # count CANs (should be 6) and neurons
        n_cans = len(cans)
        N = cans[0].S.shape[0]

        # Creates empty space for all six connectivity matrices
        self.W_torch = torch.empty(     # dont fill it, just finds space
            (n_cans, N, N),             # (6, N, N)
            dtype=self.torch_dtype,
            device=self.device,
        )
        # empty space for activity states (current activity of all neurons)
        self.S_torch = torch.empty(
            (n_cans, N, 1),
            dtype=self.torch_dtype,
            device=self.device,
        )
        
        # Copies one CAN at the time, converts into float32, then copies into Torch tensor
        for i, can in enumerate(cans):
            W_i = can.connectivity_matrix.astype(np.float32, copy=False)
            S_i = can.S.astype(np.float32, copy=False)

            self.W_torch[i].copy_(
                torch.as_tensor(
                    W_i,
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            )

            self.S_torch[i].copy_(
                torch.as_tensor(
                    S_i,
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            )

            # After W_i is copied to torch, we no longer keep the dense NumPy matrices.
            if hasattr(can, "connectivity_matrix"):
                del can.connectivity_matrix
            del W_i
            del S_i
            # Forces python to clean up
            gc.collect()

        self.W_torch = self.W_torch.contiguous()
        self.S_torch = self.S_torch.contiguous()

        # Go through each parameter used during simulation, make them into Torch tensors
        self.tau_torch = torch.tensor(
            cans[0].tau,
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.beta_torch = torch.tensor(
            self.qan.beta,
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.b_torch = torch.tensor(
            self.qan.b,
            dtype=self.torch_dtype,
            device=self.device,
        )
        # Coordinates of each neuron
        self.coords_torch = torch.tensor(
            cans[0].neurons_coordinates.astype(np.float32),
            dtype=self.torch_dtype,
            device=self.device,
        )
        #direction signs
        self.signs_torch = torch.tensor(
            [1, -1, 1, -1, 1, -1],
            dtype=self.torch_dtype,
            device=self.device,
        )
        #dimension signs
        self.dims_torch = torch.tensor(
            [0, 0, 1, 1, 2, 2],
            dtype=torch.long,
            device=self.device,
        )

        gc.collect()

        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def compute_can_inputs_torch(self, theta_dot: np.ndarray):
        """
        Maps the velocity theta_dot to the correct CAN pairing.
        """
        
        td = torch.tensor(
            theta_dot, #movement vector
            dtype=self.torch_dtype,
            device=self.device,
        )

        u = self.signs_torch * self.beta_torch * td[self.dims_torch] #The six velocity inputs
        return u.view(6, 1, 1) #turns into right shape 

    def reset(self, theta_0: np.ndarray, radius: float = 0.05):
        """
        Reset all CANs to the starting trajectory point.
        The neural bump is then at this point.
        """
        theta = torch.as_tensor(
            theta_0.reshape(1, -1), #inital position on the torus
            dtype=self.torch_dtype,
            device=self.device,
        )

        coords = self.coords_torch  # shape (N, 3)

        # compute difference between every neuron and theta_+
        diff = coords.unsqueeze(0) - theta.unsqueeze(1)
        diff = (diff + np.pi) % (2 * np.pi) - np.pi #wrapping differences insidce [0, 2π]

        #torus distance from neuron i to theta_0 in one distance instead of 3D
        distances = torch.linalg.norm(diff, dim=-1).squeeze(0)  # shape (N,)

        #How large the initial bump should be
        effective_radius = torch.max(distances) * radius
        
        #Initiate an all zero acitivty state, one CAN activity vector.
        S0 = torch.zeros(
            (self.coords_torch.shape[0], 1),
            dtype=self.torch_dtype,
            device=self.device,
        )
        #Turn on neurons close to the starting coordinate.
        S0[distances <= effective_radius] = 1.0

        #Copy that same starting bump into all six CANs.
        self.S_torch = S0.unsqueeze(0).expand(len(self.qan.cans), -1, -1).clone()

    def step(self, theta_dot: np.ndarray):
        """Compute S_tot internally and step."""
        S_tot = torch.mean(self.S_torch, dim=0) #average activity acrosse CAN
        return self.step_from_shared(S_tot, theta_dot)

    def step_from_shared(
        self, S_tot: torch.Tensor, theta_dot: np.ndarray, check_nan: bool = False,
    ):
        """
        Update all CANs from a precomputed shared S_tot.

        S_tot = the average neural activity across all six directional CANs. The average bump across the six CANs.
        theta_dot = movement vector
        """
        #velcoity inputs, one scalar input pr CAN
        u = self.compute_can_inputs_torch(theta_dot)

        #Reshapes so it matches the six-CAN tensor shape.
        S_shared = S_tot.unsqueeze(0).expand_as(self.S_torch)             
        Ws = torch.bmm(self.W_torch, S_shared) #batch matrix multiplication of the connecitivty matrices. 

        #updates each CAN from its own previous state
        self.S_torch = self.S_torch + (
            torch.relu(Ws + u + self.b_torch) 
            - self.S_torch
        ) / self.tau_torch

        if check_nan and torch.isnan(self.S_torch).any().item():
            raise ValueError("NaN values detected in torch QAN state.")
        return self.S_torch


    def simulate(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Simulate feeding a generated trajectory into the network. 
        Returning a decoded trajectory of the bump position at each timestep. 
        """
        dT = 1 / 10 #Timestep scaling factor .. Check this

        theta_0 = trajectory[0, :].copy() 
        self.reset(theta_0, radius=0.05) #Puts the bump at the inital seed position

        decoded_trajectory = []

        #Velocity computasion over each timestep and velocity at that timestep
        for t, theta in enumerate(trajectory):
            if t == 0:
                theta_dot = np.zeros(theta.shape, dtype=np.float32) 
            else:
                theta_dot = (
                    self.qan.compute_theta_dot( #computes the displacement between the current position theta and the previous position trajectory[t-1]
                        theta.copy(),
                        trajectory[t - 1, :].copy(),
                    )
                    * dT #scaled by dT
                ).astype(np.float32)

            # compute the average activity of the CANs (cancels out the assymetry)
            S_tot = torch.mean(self.S_torch, dim=0)

            #Update activity state based on the average activity of previous step
            self.step_from_shared(S_tot, theta_dot)

            #Most active neuron
            max_idx = torch.argmax(S_tot)
            #Record where the bump is
            decoded_trajectory.append(self.coords_torch[max_idx].detach())

        #the sequence of positions that the networks activity bump was at each timestep.
        out = torch.stack(decoded_trajectory, dim=0).cpu().numpy()

        assert out.shape == trajectory.shape, (
            f"out.shape: {out.shape}, trajectory.shape: {trajectory.shape}"
        )

        return out

    def run(self, trajectory: np.ndarray):
        """
        Run torch dynamics without decoding.

        Useful when you only care about final CAN state.
        """
        dT = 1 / 10

        theta_0 = trajectory[0, :].copy()
        self.reset(theta_0, radius=0.05)

        for t, theta in enumerate(trajectory):
            if t == 0:
                theta_dot = np.zeros(theta.shape, dtype=np.float32)
            else:
                theta_dot = (
                    self.qan.compute_theta_dot(
                        theta.copy(),
                        trajectory[t - 1, :].copy(),
                    )
                    * dT
                ).astype(np.float32)

            self.step(theta_dot)

        return self.S_torch

    def sync_to_cans(self):
        """
        Copy torch states back into the individual CAN3D NumPy objects.
        """
        S_np = self.S_torch.detach().cpu().numpy()

        for i, can in enumerate(self.qan.cans):
            can.S = S_np[i]

    def get_states(self):
        """
        Return current torch states as a NumPy array with shape (6, N, 1).
        """
        return self.S_torch.detach().cpu().numpy()
