import torch
import torch.nn as nn
from torch.autograd.functional import jvp

from src.couplings import Coupling

class FeedForwardCoupling(Coupling):
    def __init__(self, d_in, d_out, ffnet) -> None:
        super().__init__(d_in, d_out)

        self.mapping = ffnet

    def forward(self, x):
        """
        Computes the forward transformation.
        :param x: Tensor of shape (N, d_in), where N is the batch size.
        :return: Transformed tensor of shape (N, d_out).
        """
        return self.mapping(x)
    
    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, d_in), inputs.
        :param X: Tensor of shape (N, d_in), differentials to transform.
        :return: Transformed differential tensor of shape (N, d_out).
        """
        _, D_x = jvp(self.forward, (x,), (X,))
        return D_x