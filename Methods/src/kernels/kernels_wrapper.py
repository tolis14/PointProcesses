import torch
import gpytorch.kernels as kernels

class SquaredExponential:
    """
    Wrapper for the SE Kernel
    of gpytorch framework
    """

    def __init__(self, variance: torch.Tensor,
                 lengthscale: torch.Tensor):
        self.variance = variance
        self.covar_module = kernels.RBFKernel().initialize(lengthscale=lengthscale)

    def set_lengthscale(self, lengthscale: torch.Tensor):
        self.covar_module.lengthscale = lengthscale

    def set_variance(self, variance: torch.Tensor):
        self.variance = variance

    def __call__(self, X: torch.Tensor, Y: torch.Tensor=None):
        if Y is None:
            return self.variance * self.covar_module(X).to_dense()
        else:
            return self.variance * self.covar_module(X, Y).to_dense()
