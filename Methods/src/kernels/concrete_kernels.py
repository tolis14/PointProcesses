from src.kernels.kernel_base import *

class SquaredExponential(KernelBase):

    def __init__(self, params: dict, jitter: float=1e-5):
        super().__init__(params, jitter)

    def __call__(self, X: torch.Tensor, Y: torch.Tensor=None):

        if Y is None:
            Y = X

        # calculate exponential argument
        exp_arg = torch.tensor(-0.5) * ((X[:, None] - Y) ** 2 / (self.lengthscales ** 2)).sum(dim=-1)

        # calculate full covariance matirx
        cov_matrix = self.variance * torch.exp(exp_arg)

        # In case K is square we add jitter
        N, M = len(X), len(Y)
        if N == M:
            cov_matrix += torch.eye(N) * self.jitter

        return cov_matrix