import torch
from src.models.model_base import ModelBase, KernelBase
from src.models.vbpp.G_lookup import np_Gtilde_lookup
from torchmin import minimize


class VBPP(ModelBase):

    def __init__(self, X: torch.Tensor, kernel: KernelBase, num_points: int,
                 q_mu: torch.Tensor, L: torch.Tensor, u_bar: torch.Tensor):
        """
        :param X: Data points
        :param kernel: covariance module
        :param num_points: total number M points at each dimension
        :param q_mu: mean of the variational distribution
        :param L: lower triangular of covariance matrix
        :param u_bar: prior mean of p(u) = N(u', Kzz)
        """

        super().__init__(X, kernel)
        self.num_points = num_points
        self.inducing_points = self._get_inducing_points()
        self.q_mu = q_mu
        self.L = L
        self.ubar = u_bar

    def _one_dimensional_grid(self) -> torch.Tensor:
        bounds = self.T[0]
        x_min = bounds[0]
        x_max = bounds[1]
        return torch.linspace(x_min, x_max, self.num_points).reshape(-1, 1)

    def _two_dimensional_grid(self) -> torch.Tensor:
        x_bounds = self.T[0]
        y_bounds = self.T[1]
        z1 = torch.linspace(x_bounds[0], x_bounds[1], self.num_points)
        z2 = torch.linspace(y_bounds[0], y_bounds[1], self.num_points)
        Z1, Z2 = torch.meshgrid(z1, z2, indexing="ij")
        Z = torch.vstack((Z1.flatten(), Z2.flatten())).T
        return Z

    def _domain_area(self):
        return torch.prod(self.T[:, 1] - self.T[:, 0])

    def _get_inducing_points(self) -> torch.Tensor:
        """
        Construct a regular grid of the domain
        and sample (evenly) M inducing points
        M stands for num_points parameter
        :return: M ** d inducing points
        """
        d = self.d
        return self._one_dimensional_grid() if d == 1 else self._two_dimensional_grid()

    def _kl_divergance(self, K_zz: torch.Tensor, K_zz_inv: torch.Tensor):

        # store some intermidiate variables
        means_distance = self.ubar - self.q_mu
        q_S = self._compute_q_S(self.L)

        # calculate KL divergance terms
        first_term = torch.trace(K_zz_inv @ q_S)
        second_term = -torch.logdet(K_zz) + torch.logdet(q_S)
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
        Z = self.inducing_points
        Z_bar = (Z[: ,None] + Z) / 2.
        a = self.kernel.get_lengthscale()


        exp_arg = -((Z[:, None] - Z) ** 2 / (4 * a)).sum(dim=-1)
        exp_term = torch.exp(exp_arg)
        lengthscales_term = torch.prod(-torch.sqrt(torch.pi * a) / 2.)
        left_product_term = lengthscales_term * exp_term

        left_erf_arg = (Z_bar - self.T[:, 1]) / torch.sqrt(a)
        right_erf_arg = (Z_bar - self.T[:, 0]) / torch.sqrt(a)
        prod_term = (torch.erf(left_erf_arg) - torch.erf(right_erf_arg))
        erf_term = torch.prod(prod_term, dim=-1)

        psi_matrix = self.kernel.get_variance() ** 2 * left_product_term * erf_term

        return psi_matrix + torch.eye(psi_matrix.shape[0]) * 1e-4

    def _integral_term(self, K_zz_inv: torch.Tensor, Psi: torch.Tensor):
        """
        Method for calculating the integral term in the elbo
        according to equations 12, 13, 14, 15, 16
        """
        q_S = self._compute_q_S(self.L)

        left_term = self.q_mu @ K_zz_inv @ Psi @ K_zz_inv @ self.q_mu
        right_term = self.kernel.get_variance() * self._domain_area() \
                        - torch.trace(K_zz_inv @ Psi) \
                        + torch.trace(K_zz_inv @ q_S @ K_zz_inv @ Psi)
        return left_term + right_term

    def _expectation_term(self, Kxz: torch.Tensor, K_zz_inv: torch.Tensor,
                         K_xx: torch.Tensor):

        Kzx = Kxz.t()
        q_S = self._compute_q_S(self.L)

        mu_hat = Kxz @ K_zz_inv @ self.q_mu
        S_hat = K_xx - Kxz @ K_zz_inv @ Kzx \
                     + Kxz @ K_zz_inv @ q_S @ K_zz_inv @ Kzx
        s_hat = torch.diag(S_hat)
        #s_hat[s_hat <= 0] = 1e-6 #shouldn't be negative but it is for some reason???

        G_arg = -mu_hat ** 2 / (2 * s_hat)
        log_arg = s_hat / 2
        C = 0.57721566 #Euler-Mascheroni constant

        return (-np_Gtilde_lookup(G_arg.detach().numpy())[0] + torch.log(log_arg) - C).sum()

    def elbo(self, augmented_vector: torch.Tensor):
        #update the params of the model
        self.update_params(augmented_vector)

        #compute matrices required by all terms
        K_zz = self.kernel(self.inducing_points, jitter=1e-5)
        K_zz_inv = torch.linalg.inv(K_zz)
        K_xz = self.kernel(self.X, self.inducing_points, jitter=False)
        K_xx = self.kernel(self.X)
        Psi = self._caclulate_psi_matrix()

        #compute terms of the elbo
        expecation_term = self._expectation_term(K_xz, K_zz_inv, K_xx)
        kl_term = self._kl_divergance(K_zz, K_zz_inv)
        integral_term = self._integral_term(K_zz_inv, Psi)

        #compute full elbo
        full_elbo = -integral_term + expecation_term - kl_term
        return full_elbo

    def _compute_q_S(self, trigl: torch.Tensor) -> torch.Tensor:
        L = torch.zeros(size=(self.num_points ** self.d, self.num_points ** self.d))
        L[torch.tril_indices(self.num_points ** self.d, self.num_points ** self.d, offset=0).tolist()] = trigl
        S = L @ L.t()
        return S

    def update_params(self, augemented_vector: torch.Tensor):
        """
        Method which at each optimization step of the
        elbo maximization updates the parameters
        theta -- (kernel hyperparams)
        m -- (variational mean)
        L -- (lower triangular of the variational covariance)
        """
        theta = augemented_vector[0: self.d + 1]
        m = augemented_vector[self.d + 1: self.d + 1 + self.num_points ** self.d]
        L = augemented_vector[self.d + 1 + self.num_points ** self.d:]

        kernel_new_params = {'variance': theta[0], 'lengthscales': theta[1:]}
        #self.kernel.set_params(kernel_new_params)
        self.q_mu = m
        self.L = L

    def train(self):
        """
        Method that maximizes the ELBO
        Define augmented vector for optimization as
        [theta, m, vec(L)] where theta are kernels hyperparams,
        m is the mean and vec(L) is the vectorization of the lower triangular  L of S
        Since this method is impemented only for Squared exponential kernel we have
        y[0: dim] -> theta
        y[dim: dim + num_points] -> m
        y[dim + num_points:] -> vec(L)
        """
        theta_init = torch.cat([self.kernel.get_variance(), self.kernel.get_lengthscale()])
        m_init = self.q_mu
        L_init = self.L
        y = torch.cat([theta_init, m_init, L_init])
        y_hat = y.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([y_hat], lr=0.01, maximize=True)

        for i in range(2500):

            optimizer.zero_grad()
            loss = self.elbo(y_hat)
            loss.backward()
            optimizer.step()

            with (torch.no_grad()):
                y_hat[0 : self.d+1] = y_hat[0 : self.d+1].clamp(0.01, torch.inf)
                y_hat[self.d + 1 + self.num_points ** self.d:] = \
                                y_hat[self.d + 1 + self.num_points ** self.d:].clamp(0.05, torch.inf)

            print(torch.norm(y_hat.grad, p=2), loss)

    def predict(self, X_star: torch.Tensor):
        K_zz = self.kernel(self.inducing_points)
        K_zz_inv = torch.linalg.inv(K_zz)
        K_xz = self.kernel(X_star, self.inducing_points, jitter=False)
        K_zx = K_xz.t()
        K_xx = self.kernel(X_star, X_star)
        S = self._compute_q_S(self.L)

        pred_mean = K_xz @ K_zz_inv @ self.q_mu
        pred_cov = K_xx - K_xz @ K_zz_inv @ K_zx + K_xz @ K_zz_inv @ S @ K_zz_inv @ K_zx

        pred_intensity = pred_mean ** 2 + torch.diag(pred_cov)
        return pred_intensity


