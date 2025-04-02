from src.diffeomorphisms.vector.lower_triangular import LowerTriangularVectorDiffeomorphism
from src.dynamics import Dynamics
from src.dynamics.linear.diagonal import DiagonalLinearDynamics

class LowerTriangularNonlinearDynamics(Dynamics):

    def __init__(self, lin_dyn_1, lin_dyn_2, diffeo_1, coupling_21, diffeo_2):
        assert lin_dyn_1.dt == lin_dyn_2.dt
        assert lin_dyn_1.d == diffeo_1.d == coupling_21.d_in
        assert lin_dyn_2.d == diffeo_2.d == coupling_21.d_out
        super().__init__(lin_dyn_1.d + lin_dyn_2.d, lin_dyn_1.dt)

        self.d_1 = lin_dyn_1.d
        self.d_2 = lin_dyn_2.d

        self.K_1 = lin_dyn_1
        self.K_2 = lin_dyn_2

        self.phi_1 = diffeo_1
        self.f_21 = coupling_21
        self.phi_2 = diffeo_2

        self.K = DiagonalLinearDynamics(self.K_1, self.K_2)
        self.Phi = LowerTriangularVectorDiffeomorphism(diffeo_1, coupling_21, diffeo_2)

    def forward(self, i):
        return self.Phi.inverse(self.K.forward(i))