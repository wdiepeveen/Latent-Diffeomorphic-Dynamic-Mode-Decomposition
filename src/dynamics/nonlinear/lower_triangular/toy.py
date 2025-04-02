import torch

from src.couplings.squared_difference import SquaredDifferenceCoupling
from src.diffeomorphisms.vector.banana import BananaVectorDiffeomorphism
from src.diffeomorphisms.vector.river import RiverVectorDiffeomorphism
from src.dynamics.linear.standard import StandardLinearDynamics
from src.dynamics.nonlinear.lower_triangular import LowerTriangularNonlinearDynamics

class ToyLowerTriangularNonlinearDynamics(LowerTriangularNonlinearDynamics):

    def __init__(self):
        shear_1 = 1.
        offset_1 = 0.
        stretch_1_x = 2.
        stretch_1_y = 1/4

        diffeo_1 = RiverVectorDiffeomorphism(shear_1, offset_1, stretch_1_x, stretch_1_y)

        # Construct phi_2
        shear_2 = 1.
        offset_2 = 3.
        stretch_2_x = 2.
        stretch_2_y = 3.

        diffeo_2 = BananaVectorDiffeomorphism(shear_2, offset_2, stretch_2_x, stretch_2_y)

        # Construct f_21
        coupling_21 = SquaredDifferenceCoupling()

        # Construct K
        dt = torch.tensor([1.])
        mu_1 = torch.zeros(1)
        omega_1 = torch.tensor([0.01 * torch.pi])
        mu_2 = torch.zeros(1)
        omega_2 = torch.tensor([0.01 * torch.pi / (10 ** (1/2))])
        y0 = torch.tensor([0., 1.])
        q0 = torch.tensor([1., 1.])

        lin_dyn_1 = StandardLinearDynamics(2, dt, mu_1, omega_1, y0)
        lin_dyn_2 = StandardLinearDynamics(2, dt, mu_2, omega_2, q0)
        
        super().__init__(lin_dyn_1, lin_dyn_2, diffeo_1, coupling_21, diffeo_2)