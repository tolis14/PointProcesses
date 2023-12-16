import torch
from src.models.vbpp.model import *


class VbppWrapper:
    # Class for Wrapping all the
    # functionality of the vbpp
    # model according to the implementation
    # found (https://github.com/st--/vbpp)
    # Planning to write the whole model on pytorch
    # but since this code has been used by others for
    # comparisons with their models we should do the same (?)

    def __init__(self, X: torch.Tensor,
                 kernel: gpflow.kernels.SquaredExponential,
                 M: int):
        self.kernel = kernel
        self.X = X
        self.d = len(X[0])
        self.M = M
        self.domain = self.get_domain()
        self.domain_grid = self.get_domain_grid()
        self.domain_area = self.get_domain_area()

        self.model = self.get_model()

    def get_domain(self) -> torch.Tensor:
        """
        This method creates the domain of the
        point process as a torch.Tensor with elements
        [[x_min, x_max],[y_min, y_max],[z_min, z_max]...,[]]
        :return: domain T of the Point Process
        """
        min_at_each_dim, _ = torch.min(self.X, dim=0)
        max_at_each_dim, _ = torch.max(self.X, dim=0)
        domain = torch.vstack([min_at_each_dim, max_at_each_dim]).t()
        return domain

    def get_domain_grid(self) -> torch.Tensor:
        """
        This method constructs a grid of size M ** d
        over the domain of the point Process.
        :return: grid of size M ** d over the domain
        """

        # get m equally distant points u_dim = (u_1,...,u_m) at each dimension
        # over the domain of Point Process at each dimension
        u_over_dims = []
        for min_val, max_val in zip(self.domain[:, 0], self.domain[:, 1]):
            u = torch.linspace(min_val, max_val, self.M)
            u_over_dims.append(u)

        # fix the m x d grid over the domain
        grid = torch.meshgrid(u_over_dims, indexing='ij')
        grid = torch.vstack([x.flatten() for x in grid]).t()
        return grid

    def get_domain_area(self) -> torch.Tensor:
        """
        This method computed the area of the domain
        T of the Point Process which is simply
        |T| = (x_1_max - x_1_min) * ... * (x_d_max - x_d_min)
        for a d-dimensional domain T
        Check ModelBase to see how the domain is constructed
        :return: area of |T| according to the Lebesgue measure
        """
        area = torch.prod(self.domain[:, 1] - self.domain[:, 0])
        return area

    def get_model(self):

        Z = self.get_domain_grid().numpy()
        inducind_points = gpflow.inducing_variables.InducingPoints(Z)
        gpflow.set_trainable(inducind_points, False)

        q_mu = np.ones(self.M ** self.d)
        q_S = np.eye(self.M ** self.d)
        num_events = len(self.X)
        beta0 = np.sqrt(num_events / self.get_domain_area().numpy())

        model = VBPP(
            inducind_points,
            self.kernel,
            self.get_domain().numpy(),
            q_mu,
            q_S,
            beta0=beta0,
            num_events=len(self.X),
            whiten=True
        )
        return model

    def closure(self):
        return -self.model.elbo(self.X.numpy())

    def train(self):
        gpflow.optimizers.Scipy().minimize(self.closure, self.model.trainable_variables)
        return self

    def predict(self, X_star: torch.Tensor):
        """
        Method to predict at new data points
        :param X_star: test data often referred as
        X* in the GP context
        :return: E[λ], lower bound, upper percentiles for the intensity
        since λ = f^2 where f ~ GP
        Here λ denotes the intensity
        """
        X_star = X_star.numpy()
        lambda_mean, lower, upper = self.model.predict_lambda_and_percentiles(X_star)
        lambda_mean, lower, upper = lambda_mean.numpy(), lower.numpy(), upper.numpy()
        return torch.tensor(lambda_mean), torch.tensor(lower), torch.tensor(upper)