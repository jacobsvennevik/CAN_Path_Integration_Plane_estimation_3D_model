"""
Runner for the torch backend, in larger simulations need to use this class.
For ligther runs just use the MADE framework and the QAN3D.py and CAN3D classes.
"""

import gc
import numpy as np
import math
import torch
from dataclasses import dataclass, field


@dataclass
class TorchBackend:
    """
    Torch-backed simulation engine for a Torus3DQAN.
    """

    qan: object                                  # Torus3DQAN instance
    torch_dtype: torch.dtype = torch.float32    #Uses cheeper float32 

    device: torch.device = field(init=False)
    S:      torch.Tensor = field(init=False) #The current neural activity
    coords: torch.Tensor = field(init=False) #Torus coordinates
    tau:    torch.Tensor = field(init=False) #Neural time constant
    b: torch.Tensor = field(init=False) #bias term to produce activity in every neuron

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
        
    def _compute_fft_kernels(self, n: int) -> torch.Tensor:
        """
        Analytically compute the pre-FFT'd recurrent kernel for each CAN.
        Much faaster than original dense kernel.

 
        Returns
        -------
        W_fft : torch.Tensor, shape (6, n, n, n//2+1), dtype complex64
        """
        alpha      = float(self.qan.alpha)
        sigma      = float(self.qan.sigma)
        offset_mag = float(self.qan.offset_magnitude)
        TWO_PI     = 2.0 * math.pi

        fft_device = torch.device("cpu") if self.device.type == "mps" else self.device
        # Grid of torus coordinates in [0, 2π)³
        idx   = torch.arange(n, dtype=torch.float32, device=fft_device)
        g1, g2, g3 = torch.meshgrid(idx, idx, idx, indexing="ij")

        # theta: shape (n, n, n, 3)
        theta = torch.stack([g1, g2, g3], dim=-1) * (TWO_PI / n)
 
        W_fft = torch.empty(
        (6, n, n, n // 2 + 1), dtype=torch.complex64, device=fft_device
        )
        #Hexagonal metric
        metric = self.qan.manifold.metric
        theta_grid = (np.indices((n, n, n)).reshape(3, -1).T) * (2*np.pi / n)
        for i in range(6):
            dim       = i // 2
            direction = 1.0 if i % 2 == 0 else -1.0
 
            # offset vector for this CAN
            delta = np.zeros((1, 3)); delta[0, dim] = direction * offset_mag
 
            # Euclidean distance on the torus
            dist  = metric(theta_grid, delta).reshape(n, n, n)
 
            # kernel  k(d) = α exp(−d²/2σ²) − α
            kernel = alpha * np.exp(-dist**2 / (2*sigma**2)) - alpha
 
            # store real-FFT of kernel
            W_fft[i] = torch.fft.rfftn(torch.as_tensor(kernel, dtype=torch.float32), dim=(-3,-2,-1))
 
        return W_fft
 
 
    def _apply_W_fft(self, S_shared: torch.Tensor) -> torch.Tensor:
        n = self.n
        N = n ** 3

        S_3d = S_shared[0].squeeze(-1).reshape(n, n, n)

        # Remove the .cpu() calls — stay on device for CUDA
        if self.device.type == "mps":
            S_3d  = S_3d.cpu()
            W_fft = self.W_fft.cpu()
        else:
            W_fft = self.W_fft.to(self.device)  # already there after init, no-op

        S_fft  = torch.fft.rfftn(S_3d, dim=(-3, -2, -1))
        Ws_fft = W_fft * S_fft.unsqueeze(0)
        Ws_3d  = torch.fft.irfftn(Ws_fft, s=(n, n, n), dim=(-3, -2, -1))

        return Ws_3d.reshape(6, N, 1).to(self.device)

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
        self.n = int(round(N ** (1/3)))    
        self.W_fft = self._compute_fft_kernels(self.n)

        # empty space for activity states (current activity of all neurons)
        self.S = torch.empty(
            (n_cans, N, 1),
            dtype=self.torch_dtype,
            device=self.device,
        )
        
        # Copies one CAN at the time, converts into float32, then copies into Torch tensor
        for i, can in enumerate(cans):
            S_i = can.S.astype(np.float32, copy=False)

            self.S[i].copy_(
                torch.as_tensor(
                    S_i,
                    dtype=self.torch_dtype,
                    device=self.device,
                )
            )
            

            # After W_i is copied to torch, we no longer keep the dense NumPy matrices.
            if hasattr(can, "connectivity_matrix"):
                del can.connectivity_matrix
            del S_i
            # Forces python to clean up
            gc.collect()

        self.S = self.S.contiguous()

        # Go through each parameter used during simulation, make them into Torch tensors
        self.tau = torch.tensor(
            cans[0].tau,
            dtype=self.torch_dtype,
            device=self.device,
        )

        self.b = torch.tensor(
            self.qan.b,
            dtype=self.torch_dtype,
            device=self.device,
        )
        # Coordinates of each neuron
        self.coords = torch.tensor(
            cans[0].neurons_coordinates.astype(np.float32),
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

        coords = self.coords  # shape (N, 3)

        # compute difference between every neuron and theta_+
        diff = coords.unsqueeze(0) - theta.unsqueeze(1)
        diff = (diff + np.pi) % (2 * np.pi) - np.pi #wrapping differences insidce [0, 2π]

        #torus distance from neuron i to theta_0 in one distance instead of 3D
        distances = torch.linalg.norm(diff, dim=-1).squeeze(0)  # shape (N,)

        #How large the initial bump should be
        effective_radius = torch.max(distances) * radius
        
        #Initiate an all zero acitivty state, one CAN activity vector.
        S0 = torch.zeros(
            (self.coords.shape[0], 1),
            dtype=self.torch_dtype,
            device=self.device,
        )
        #Turn on neurons close to the starting coordinate.
        S0[distances <= effective_radius] = 1.0

        #Copy that same starting bump into all six CANs.
        self.S = S0.unsqueeze(0).expand(len(self.qan.cans), -1, -1).clone()

    def step(self, theta_dot: np.ndarray):
        """Compute S_tot internally and step."""
        S_tot = torch.mean(self.S, dim=0) #average activity acrosse CAN
        return self.step_from_shared_state(S_tot, theta_dot)

    def step_from_shared_state(self, S_tot: torch.Tensor, theta_dot: np.ndarray,
        check_nan: bool = False,
    ) -> torch.Tensor:
        """
        Implements the bump updates driven by the velcoity using the shared mean neural state, the current neural activity S_tot. 
        In this way we take a step, and the bump moves.
        """
        N = S_tot.shape[0]

        #The current neural activity, gives us where the bump is currently
        S_shared = S_tot.unsqueeze(0).expand(6, N, 1)
        #Apply each CANs shifted weight matricies W to the shared bump of activity       
        Ws = self._apply_W_fft(S_shared)
        # Apply the recurrent input to each neuron, 
        # passed through a relu and shifted up by the bias b.
        drives = torch.relu(Ws + self.b)               

        # velocity blend coefficient — gain comes from the QAN's single-source property
        a = torch.tanh(self.qan.velocity_gains
               * torch.tensor(theta_dot, dtype=torch.float32, device=self.device))
        # blend weights, how much each CAN contributes                                                        
        alpha_pos = (0.5 + 0.5 * a).view(3, 1, 1)              
        alpha_neg = (0.5 - 0.5 * a).view(3, 1, 1)              

        drives_paired = drives.view(3, 2, N, 1)
        #the weighted average of the forward and backward CAN drives for each dimension           
        combined = (
            alpha_pos * drives_paired[:, 0]                     
        + alpha_neg * drives_paired[:, 1]                     
        )                                                     
        #Both CANs in each pair update toward the same target  
        combined_targets = (
            combined.unsqueeze(1)                                
                    .expand(-1, 2, -1, -1)                     
                    .reshape(6, N, 1)                          
        )
        #Updates the current neural state
        self.S = self.S + (
            combined_targets - self.S
        ) / self.tau

        if check_nan and torch.isnan(self.S).any().item():
            raise ValueError("NaN values detected in torch QAN state.")

        return self.S

    def decode_position(self, S_tot):
        """Return current bump position on T³ as (3,) numpy array."""
        #Most active neuron
        max_idx = torch.argmax(S_tot)
        return self.coords[max_idx].cpu().numpy()
    
    def decode_position_com(self, S_tot):
        weights = torch.relu(S_tot).flatten()  # stay in torch
        coords = self.coords             # (N, 3), already on device

        sin_w = (torch.sin(coords) * weights.unsqueeze(1)).sum(dim=0)
        cos_w = (torch.cos(coords) * weights.unsqueeze(1)).sum(dim=0)
        result = torch.atan2(sin_w, cos_w) % (2 * torch.pi)
        return result 

    def simulate(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Simulate feeding a generated trajectory into the network. 
        Returning a decoded trajectory of the bump position at each timestep. 
        """

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
                ).astype(np.float32)

            # compute the average activity of the CANs (cancels out the assymetry)
            S_tot = torch.mean(self.S, dim=0)

            #Update activity state based on the average activity of previous step
            self.step_from_shared_state(S_tot, theta_dot)

            #Record where the bump is
            decoded_trajectory.append(self.decode_position_com(S_tot))

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
                ).astype(np.float32)

            self.step(theta_dot)

        return self.S

    def sync_to_cans(self):
        """
        Copy torch states back into the individual CAN3D NumPy objects.
        """
        S_np = self.S.detach().cpu().numpy()

        for i, can in enumerate(self.qan.cans):
            can.S = S_np[i]

    def get_states(self):
        """
        Return current torch states as a NumPy array with shape (6, N, 1).
        """
        return self.S.detach().cpu().numpy()
    
    def allocate_state_buffer(self, T: int, stride: int = 1) -> "torch.Tensor":
        """Make room an on-device buffer for recording S_tot at each timestep."""
        N = self.S.shape[1]
        n_frames = (T + stride - 1) // stride
        return torch.empty(
            (n_frames, N),
            dtype=self.torch_dtype,
            device="cpu",
        )

    def record_state_to_buffer(self, buf: "torch.Tensor", t: int, stride: int = 1) -> None:
        """Write current S_tot into row t of the buffer. No CPU transfer."""
        buf[t // stride] = self.S.mean(dim=0).squeeze().to(buf.device)

    def buffer_to_numpy(self, buf: "torch.Tensor") -> "np.ndarray":
        """Single CPU transfer of the full buffer."""
        return buf.cpu().numpy()

    def allocate_ratemap(self, bins: int) -> tuple:
        """On-device accumulator for the 2-D per-neuron rate map.."""
        N = self.S.shape[1]         
        sums   = torch.zeros((bins * bins, N), dtype=torch.float32, device=self.device)
        counts = torch.zeros( bins * bins,      dtype=torch.float32, device=self.device)
        return sums, counts

    def record_ratemap(self, acc: tuple, flat_bin: int) -> None:
        """Accumulate current S_tot into bins (much cheeper and quicker).
        The caller computes flat_bin so the backend stays unaware of arena geometry."""
        sums, counts = acc
        with torch.no_grad():            # belt-and-suspenders on MPS
            sums[flat_bin] += self.S.mean(dim=0).squeeze()
        counts[flat_bin] += 1.0

    def ratemap_to_numpy(self, acc: tuple, bins: int) -> tuple:
        """Single CPU transfer from ratemap to numpy"""
        sums, counts = acc
        return (
            sums.cpu().numpy().reshape(bins, bins, -1),
            counts.cpu().numpy().reshape(bins, bins),
        )