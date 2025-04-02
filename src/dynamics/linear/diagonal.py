import torch

from src.dynamics import Dynamics

class DiagonalLinearDynamics(Dynamics):

    def __init__(self, lin_dyn_1, lin_dyn_2):
        assert lin_dyn_1.dt == lin_dyn_2.dt
        super().__init__(lin_dyn_1.d + lin_dyn_2.d, lin_dyn_1.dt)

        self.K_1 = lin_dyn_1
        self.K_2 = lin_dyn_2

    def forward(self, i):
        x = self.K_1.forward(i)
        p = self.K_2.forward(i)
        
        # Concatenate x and p along the batch dimension (dimension 0)
        output = torch.cat([x, p], dim=1)
        
        return output
