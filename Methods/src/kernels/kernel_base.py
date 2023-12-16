import torch

class KernelBase:
    """
    Base Class for stationary kernels at same family with RBF etc.
    These kernels are parameterized by a variance and
    lengthscales which might differ for each dimension
    """

    def __init__(self, params: dict, jitter: float=1e-5):
        self.variance = params['variance']
        self.lengthscales = params['lengthscales']
        self.jitter = jitter

    def set_params(self, params: dict):
        self.variance = params['variance']
        self.lengthscales = params['lengthscales']

    def set_variance(self, variance: torch.Tensor):
        self.variance = variance

    def set_lengthscales(self, lengthscales: torch.Tensor):
        self.lengthscales = lengthscales

    def get_params(self) -> dict:
        return {'variance': self.variance,
                 'lengthscales': self.lengthscales}

    def get_lengthscales(self) -> torch.Tensor:
        return self.lengthscales

    def get_variance(self) -> torch.Tensor:
        return self.variance

    def __call__(self, X: torch.Tensor, Y: torch.Tensor=None):
        pass
