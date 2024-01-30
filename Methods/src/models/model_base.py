import torch
from src.kernels.kernel_base import KernelBase


class ModelBase:

    def __init__(self, X: torch.Tensor, kernel: KernelBase):
        self.X = X
        self.d = X.shape[1]
        self.domain = self.get_domain()
        self.kernel = kernel


    def get_domain(self) -> torch.Tensor:
        """
        This method creates the domain of the
        point process as a torch.Tensor with elements
        [[x_min, x_max],[y_min, y_max],[z_min, z_max]...,[]]
        :return: domain T of the Point Process
        """
        min_at_each_dim,_ = torch.min(self.X, dim=0)
        max_at_each_dim,_ = torch.max(self.X, dim=0)
        domain = torch.vstack([min_at_each_dim, max_at_each_dim]).t()
        return domain

    def train(self, params: dict):
        pass

    def predict(self, X_star: torch.Tensor):
        pass

    def get_sample(self, X_star: torch.Tensor, num_samples:int=1) -> torch.Tensor:
        """
        Method to draw samples from the predictive distribution
        """
        pass