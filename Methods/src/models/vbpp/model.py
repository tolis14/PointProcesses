import torch
from src.models.model_base import ModelBase
from src.kernels.kernels_wrapper import SquaredExponential


class VBPP(ModelBase):

    def __init__(self, X: torch.Tensor, kernel: SquaredExponential, num_points: int,
                 q_mu: torch.Tensor, q_S: torch.Tensor, u_bar: torch.Tensor):
        """
        :param X: Data points
        :param kernel: covariance module
        :param num_points: total number M ** d of inducing points, (M) at each dimension
        :param q_mu: mean of the variational distribution
        :param q_S: covariance of the variational distribution
        :param u_bar: prior mean of p(u) = N(u', Kzz)
        """

        super().__init__(X, kernel)
        self.num_points = num_points
        self.inducing_points = self._get_inducing_points()
        self.q_mu = q_mu
        self.q_S = q_S
        self.ubar = u_bar

    def _one_dimensional_grid(self) -> torch.Tensor:
        bounds = self.T[0]
        x_min = bounds[0]
        x_max = bounds[1]
        return torch.linspace(x_min, x_max, self.num_points)

    def _two_dimensional_grid(self) -> torch.Tensor:
        x_bounds = self.T[0]
        y_bounds = self.T[1]
        z1 = torch.linspace(x_bounds[0], x_bounds[1], self.num_points)
        z2 = torch.linspace(y_bounds[0], y_bounds[1], self.num_points)
        Z1, Z2 = torch.meshgrid(z1, z2, indexing="ij")
        Z = torch.vstack((Z1.flatten(), Z2.flatten())).T
        return Z

    def _get_inducing_points(self) -> torch.Tensor:
        """
        Construct a regular grid of the domain
        and sample (evenly) M inducing points
        M stands for num_points parameter
        :return: M inducing points
        """
        d = self.d
        return self._one_dimensional_grid() if d == 1 else self._two_dimensional_grid()

    def _kl_divergance(self, K_zz: torch.Tensor, K_zz_inv: torch.Tensor):

        # store some intermidiate variables
        means_distance = self.ubar - self.q_mu

        # calculate KL divergance terms
        first_term = torch.trace(K_zz_inv @ self.q_S)
        second_term = -torch.logdet(K_zz) + torch.logdet(self.q_S)
        third_term = -self.num_points ** self.d
        fourth_term = means_distance @ K_zz_inv @ means_distance

        # full KL divergance
        return 0.5 * (first_term + second_term + third_term + fourth_term)


    def _caclulate_psi_matrix(self):
        """
        This method calculates
        Ψ = integral{K(z,x) K(x,z')dx} only for the
        squared exponential kernel according to paper
        :return: Ψ integral
        """
        pass

    def _integral_term(self, K_zz_inv: torch.Tensor, Psi: torch.Tensor):
        """
        Method for calculating the integral term in the elbo
        according to equations 12, 13, 14, 15, 16
        """
        pass


    def elbo(self):
       pass


    def train(self):
        pass

    def predict(self):
        pass 