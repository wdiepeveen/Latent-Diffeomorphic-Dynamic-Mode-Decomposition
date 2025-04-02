import torch
import torch.nn as nn

from src.couplings import Coupling

class SquaredDifferenceCoupling(Coupling):
    def __init__(self) -> None:
        super().__init__(2,2)

    def forward(self, x):
        """
        Computes the forward transformation.
        :param x: Tensor of shape (N, 2), where N is the batch size.
        :return: Transformed tensor of shape (N, 2).
        """
        y = torch.zeros_like(x)
        y[:,0] = x[:,0]**2 + x[:,1]**2
        y[:,1] = x[:,0] - x[:,1]
        return y
    
    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 2), inputs.
        :param X: Tensor of shape (N, 2), differentials to transform.
        :return: Transformed differential tensor of shape (N, 2).
        """
        D_x = torch.zeros_like(X)
        D_x[:,0] = 2 * x[:,0] * X[:,0] + 2 * x[:,1] * X[:,1]
        D_x[:,1] = X[:,0] - X[:,1]
        return D_x