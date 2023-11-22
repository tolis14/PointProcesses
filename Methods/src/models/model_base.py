import torch
from src.kernels.kernel_base import KernelBase


class ModelBase:

    def __init__(self, X: torch.Tensor, kernel: KernelBase):
        self.X = X
        self.d = X.shape[1]
        self.T = self.create_domain()
        self.kernel = kernel


    def create_domain(self) -> torch.Tensor:
        """
        Provide support for 1D and
        2D domains in this package
        :return: domain of Poisson Process
        """

        if self.d == 1:
            x_min = torch.min(self.X)
            x_max = torch.max(self.X)
            bounds = [x_min, x_max]
            return torch.tensor([bounds])
        elif self.d == 2:
            x_min = torch.min(self.X[:, 0])
            x_max = torch.max(self.X[:, 0])
            y_min = torch.min(self.X[:, 1])
            y_max = torch.max(self.X[:, 1])
            x_bounds = [x_min, x_max]
            y_bounds = [y_min, y_max]
            return torch.tensor([x_bounds, y_bounds])


    def train(self):
        pass

    def predict(self, X_star: torch.Tensor):
        pass