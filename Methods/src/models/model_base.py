import torch
from src.kernels.kernels_wrapper import SquaredExponential


class ModelBase:

    def __init__(self, X: torch.Tensor, kernel: SquaredExponential):
        self.X = X
        self.d = X.shape[1]
        self.T = self.create_domain()
        self.kernel = kernel


    def create_domain(self) -> list:
        """
        Provide support for 1D and
        2D domains in this package
        :return: domain of Poisson Process
        """

        if self.d == 1:
            x_min = torch.min(self.X)
            x_max = torch.max(self.X)
            bounds = (x_min, x_max)
            return [bounds]
        elif self.d == 2:
            x_min = torch.min(self.X[:, 0])
            x_max = torch.max(self.X[:, 0])
            y_min = torch.min(self.X[:, 1])
            y_max = torch.max(self.X[:, 1])
            x_bounds = (x_min, x_max)
            y_bounds = (y_min, y_max)
            return [x_bounds, y_bounds]


    def train(self):
        pass

    def predict(self):
        pass