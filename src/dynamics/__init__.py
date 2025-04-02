import torch.nn as nn

class Dynamics(nn.Module):
    def __init__(self, d, dt) -> None:
        super().__init__()
        self.d = d
        self.num_blocks = self.d // 2

        # Set the time step as a parameter
        self.dt = dt

    def forward(self, i):
        raise NotImplementedError("Subclasses should implement this")