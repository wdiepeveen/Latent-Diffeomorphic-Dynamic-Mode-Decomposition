import torch.nn as nn

class Coupling(nn.Module):
    """ 
    Base class describing a coupling f: R^d -> R^n. 

    This class provides a template for defining transformations. 
    It includes methods for forward as well as differentials. 

    Attributes:
        d (tuple): Dimension of one element in the batch. For Euclidean data, 
                   this is a single integer (the dimensionality of the data).
                   For image data, this should be a tuple (c, h, w) representing 
                   the number of channels, height, and width.
    """

    def __init__(self, d_in, d_out) -> None:
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x):
        """
        Applies the forward transformation f to the input data x.

        Args:
            x (torch.Tensor): Input tensor of shape (N, d) for Euclidean data 
                              or (N, c, h, w) for image data.

        Returns:
            torch.Tensor: Transformed data of shape (N, n).
        """
        raise NotImplementedError("Subclasses should implement this")

    def differential_forward(self, x, X):
        """
        Computes the differential of the forward transformation at x 
        for a vector X.

        Args:
            x (torch.Tensor): Points in the domain, shape (N, d) or (N, c, h, w).
            X (torch.Tensor): Tangent vectors at x, shape (N, d) or (N, c, h, w).

        Returns:
            torch.Tensor: Transformed tangent vectors of shape (N, n).
        """
        raise NotImplementedError("Subclasses should implement this")
