import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dynamics import Dynamics

class StandardLinearDynamics(Dynamics):
    def __init__(self, d, dt, mus, omegas, x0):
        super().__init__(d, dt)
        # Initialize learnable omega parameters
        assert len(mus) == len(omegas) == self.num_blocks
        self.mus = mus
        self.omegas = omegas

        # Initialization of the dynamics
        self.start = x0

    def forward(self, i): 
        # Construct the block diagonal matrix
        blocks = torch.zeros(len(i), self.num_blocks, 2, 2)
        for k, (mu, omega) in enumerate(zip(self.mus, self.omegas)):
            decay = torch.exp(-mu * i * self.dt).squeeze()
            sin = torch.sin(omega * i * self.dt).squeeze()
            cos = torch.cos(omega * i * self.dt).squeeze()
            blocks[:,k,0,0] = decay * cos
            blocks[:,k,0,1] = - decay * sin
            blocks[:,k,1,0] = decay * sin
            blocks[:,k,1,1] = decay * cos
        
        # Combine blocks into a block diagonal matrix
        start_blocks = self.start[:self.num_blocks*2].reshape(self.num_blocks, 2)
        out = torch.einsum("Nkab,kb->Nka", blocks, start_blocks).reshape(-1, 2 * self.num_blocks)

        # Handle odd d by padding if necessary
        if self.d % 2 != 0:
            last_item = self.start[-1].unsqueeze(0).expand(len(i), -1)  # Expand to match batch size
            out = torch.cat([out, last_item], dim=-1)  # Concatenate along the last dimension
        return out