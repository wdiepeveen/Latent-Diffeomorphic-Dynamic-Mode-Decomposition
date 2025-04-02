from src.dynamics import Dynamics

class HybridLowerTriangularNonlinearDynamics(Dynamics):

    def __init__(self, lin_dyn, diffeo, coupling):
        assert lin_dyn.d == diffeo.d == coupling.d_out
        super().__init__(lin_dyn.d, lin_dyn.dt)

        self.d_in = coupling.d_in

        self.K = lin_dyn

        self.phi = diffeo
        self.f = coupling

    def forward(self, i, x):
        return self.phi.inverse(self.K.forward(i) - self.f.forward(x))