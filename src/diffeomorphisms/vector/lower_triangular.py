import torch

from src.diffeomorphisms.vector import VectorDiffeomorphism

class LowerTriangularVectorDiffeomorphism(VectorDiffeomorphism):

    def __init__(self, diffeo_1, coupling_21, diffeo_2):
        super().__init__(diffeo_1.d + diffeo_2.d)

        self.d_1 = diffeo_1.d
        self.d_2 = diffeo_2.d

        self.phi_1 = diffeo_1
        self.f_21 = coupling_21
        self.phi_2 = diffeo_2

    def forward(self, x):
        y = torch.zeros_like(x)
        y[:,0:self.d_1] = self.phi_1.forward(x[:,0:self.d_1])
        y[:,self.d_1:] = self.f_21.forward(x[:,0:self.d_1]) + self.phi_2.forward(x[:,self.d_1:])
        return y
    
    def inverse(self, y):
        x = torch.zeros_like(y)
        x[:,0:self.d_1] = self.phi_1.inverse(y[:,0:self.d_1])
        x[:,self.d_1:] = self.phi_2.inverse(y[:,self.d_1:] - self.f_21.forward(self.phi_1.inverse(y[:,0:self.d_1])))
        return x

    def differential_forward(self, x, X):
        D_x = torch.zeros_like(X)
        D_x[:,0:self.d_1] = self.phi_1.differential_forward(x[:,0:self.d_1], X[:,0:self.d_1])
        D_x[:,self.d_1:] = self.f_21.differential_forward(x[:,0:self.d_1], X[:,0:self.d_1]) + self.phi_2.differential_forward(x[:,self.d_1:], X[:,self.d_1:])
        return D_x

    def differential_inverse(self, y, Y):
        D_y = torch.zeros_like(Y)
        D_y[:,0:self.d_1] = self.phi_1.differential_inverse(y[:,0:self.d_1], Y[:,0:self.d_1])
        y_tmp_1 = y[:,self.d_1:] - self.f_21.forward(self.phi_1.inverse(y[:,0:self.d_1]))
        y_tmp_2 = self.phi_1.inverse(y[:,0:self.d_1])
        D_y[:,self.d_1:] = self.phi_2.inverse(y_tmp_1, Y[:,self.d_1:] - self.f_21.differential_forward(y_tmp_2, self.phi_1.differential_inverse(y[:,0:self.d_1], Y[:,0:self.d_1])))