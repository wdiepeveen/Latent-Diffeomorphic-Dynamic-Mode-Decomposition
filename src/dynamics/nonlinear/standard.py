from src.dynamics import Dynamics

class StandardNonlinearDynamics(Dynamics):

    def __init__(self, d, dt, lin_dyn, diffeo):
        super().__init__(d, dt)
        self.K = lin_dyn
        self.phi = diffeo

    def forward(self, i):
        return self.phi.inverse(self.K.forward(i))