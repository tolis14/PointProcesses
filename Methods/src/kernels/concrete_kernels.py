from src.kernels.kernel_base import *

class SquaredExponential(KernelBase):

    def __init__(self, params: dict):
        super().__init__(params)

    def __call__(self, X: torch.Tensor, Y: torch.Tensor=None, jitter=1e-5):

        if Y is None:
            Y = X

        #calculate exponential argument
        exp_arg = -((X[:, None] - Y) ** 2 / (2 * self.lengthscales)).sum(dim=-1)

        #calculate full covariance matirx
        cov_matrix = self.variance * torch.exp(exp_arg)

        if not jitter:
            return cov_matrix

        jitterred_covariance_matrix = cov_matrix + torch.eye(cov_matrix.shape[0]) * jitter
        return jitterred_covariance_matrix