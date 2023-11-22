import torch

class KernelBase:
    """
    Base Class for stationary kernels at same family
    with RBF etc.
    These kernels are parameterized by a variance and
    lengthscales which might differ for each dimension
    """

    def __init__(self, params: dict):
        self.variance = params['variance']
        self.lengthscales = params['lengthscales']

    def set_params(self, params: dict):
        self.variance = params['variance']
        self.lengthscales = params['lengthscales']

    def get_params(self) -> dict:
        return {'variance': self.variance,
                 'lengthscales': self.lengthscales}

    def get_lengthscale(self) -> torch.Tensor:
        return self.lengthscales

    def get_variance(self) -> torch.Tensor:
        return self.variance

    def __call__(self, X: torch.Tensor, Y: torch.Tensor=None, jitter=1e-5):
        pass
