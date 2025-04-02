import torch
import torch.nn as nn

from src.diffeomorphisms.vector import VectorDiffeomorphism

class BananaVectorDiffeomorphism(VectorDiffeomorphism):
    def __init__(self, shear, offset, stretch_1, stretch_2) -> None:
        super().__init__(2)
        self.a = shear  # float
        self.z = offset  # float
        self.s = torch.tensor([stretch_1, stretch_2])

    def forward(self, x):
        """
        Computes the forward transformation of the diffeomorphism.
        :param x: Tensor of shape (N, 2), where N is the batch size.
        :return: Transformed tensor of shape (N, 2).
        """
        y_0 = x[:, 0] - self.a * x[:, 1].pow(2) - self.z  # Compute new y[:, 0] without in-place operation
        y = torch.stack([y_0, x[:, 1]], dim=1)  # Create new tensor with modified first column
        y = y * self.s[None]  # Use regular multiplication instead of in-place
        return y

    def inverse(self, y):
        """
        Computes the inverse transformation of the diffeomorphism.
        :param y: Tensor of shape (N, 2), where N is the batch size.
        :return: Inverted tensor of shape (N, 2).
        """
        x = y / self.s[None]  # Avoid in-place division
        x_0 = x[:, 0] + self.a * x[:, 1].pow(2) + self.z  # Compute new x[:, 0] without in-place operation
        x = torch.stack([x_0, x[:, 1]], dim=1)  # Create new tensor with modified first column
        return x

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation.
        :param x: Tensor of shape (N, 2), inputs.
        :param X: Tensor of shape (N, 2), differentials to transform.
        :return: Transformed differential tensor of shape (N, 2).
        """
        D_x = X.clone()
        D_x[:, 0] = D_x[:, 0] - 2 * self.a * D_x[:, 1] * X[:, 1]
        D_x *= self.s[None]
        return D_x

    def differential_inverse(self, y, Y):
        """
        Computes the differential of the inverse transformation.
        :param y: Tensor of shape (N, 2), inputs.
        :param Y: Tensor of shape (N, 2), differentials to invert.
        :return: Inverted differential tensor of shape (N, 2).
        """
        D_y = Y.clone()
        D_y /= self.s[None]
        D_y[:, 0] = D_y[:, 0] + 2 * self.a * y[:, 1] * D_y[:, 1]
        return D_y

