import torch
from scipy.stats import multivariate_normal
from torchmin import ScipyMinimizer
import matplotlib.pyplot as plt

def translate_data(X: torch.Tensor, domain_bounds: torch.Tensor) -> torch.Tensor:

    dimensions = X.shape[1]
    for d in range(dimensions):
        low, high = domain_bounds[d]
        X[:, d] = (X[:, d] - low) / (high - low) * torch.pi
    return X

class LBPP:

    def __init__(self,  data: torch.Tensor,
                        N: int,
                        a: torch.float,
                        b: torch.float,
                        m: int = 2,
                        domain_bounds=None):

        self.data = data
        self.X = translate_data(torch.clone(data), domain_bounds)
        self.N = N
        self.d = data.shape[1]

        self.domain_bounds = domain_bounds

        self.alpha_hat = (torch.rand(len(self.X), dtype=torch.float) + 1) * 0.01
        self.a_init, self.b_init, self.m = a, b, m
        self.theta_hat = torch.tensor([a, b])


    def get_multi_index(self):
        if self.d == 1:
            B = torch.arange(0, self.N, 1)
            return B.reshape(-1, 1)
        elif self.d == 2:
            b1 = torch.arange(0, self.N, 1)
            b2 = torch.arange(0, self.N, 1)
            B1, B2 = torch.meshgrid(b1, b2, indexing="ij")
            B = torch.vstack((B1.flatten(), B2.flatten())).T
            return B

    def create_feature_matrix(self, X: torch.Tensor, B: torch.Tensor) -> torch.Tensor:

        constant_term = (2. / torch.pi) ** (self.d / 2.)

        first_term = torch.where(B == 0, torch.sqrt(torch.tensor(0.5)), 1)
        second_term = torch.cos(B[:, None] * X)

        first_term = torch.prod(first_term, dim=-1)
        second_term = torch.prod(second_term, dim=-1)

        prod_term = first_term[:, None] * second_term

        return constant_term * prod_term

    def get_eigen_matrix(self, B: torch.Tensor, theta_hat: torch.Tensor) -> torch.Tensor:
        a, b = theta_hat[0], theta_hat[1]
        l_b = 1. / (a * ((B ** 2).sum(dim=-1) ** self.m) + b)
        return l_b

    def K_hat(self, theta_hat: torch.Tensor, X: torch.Tensor, Y: torch.Tensor=None):
        B = self.get_multi_index()

        L = self.get_eigen_matrix(B, theta_hat)
        L_inv_I = ( 1. / L ) + 1
        M = torch.diag(1. / L_inv_I)

        PHI_X = self.create_feature_matrix(X, B)
        if Y is not None:
            PHI_Y = self.create_feature_matrix(Y, B)
            return PHI_X.t() @ M @ PHI_Y

        cov_matrix = PHI_X.t() @ M @ PHI_X

        return cov_matrix, L

    @staticmethod
    def log_posterior(alpha: torch.Tensor, K_hat_xx: torch.Tensor):
        """
        :return: The value of the log posterior according
        to equation (8) but in alpha space
        In their paper implementation they used optimization
        over alpha and not w, so we are doing the same here.
        """

        data_term = -2 * torch.sum(torch.log(torch.abs(alpha)))
        reg_term = 0.5 * (alpha @ K_hat_xx @ alpha)
        return data_term + reg_term

    def laplace_map(self, K_hat_xx: torch.Tensor, train_params=False):
        self.alpha_hat.requires_grad_()
        params = [self.alpha_hat]
        tol = 1e-6
        method = 'L-BFGS-B'
        options = {'maxiter': 1} if train_params else {}
        optimizer = ScipyMinimizer(
            params=params,
            method=method,
            tol=tol,
            options=options
        )

        def closure():
            optimizer.zero_grad()
            loss = self.log_posterior(self.alpha_hat, K_hat_xx)
            loss.backward(retain_graph=True)
            return loss

        optimizer.step(closure)
        self.alpha_hat = torch.abs(self.alpha_hat)
        self.alpha_hat = self.alpha_hat.detach()
        return self.alpha_hat

    def marginal_log_likelihood(self, theta: torch.Tensor):
        K_hat_xx, L = self.K_hat(theta, self.X)

        M = K_hat_xx * (self.alpha_hat[:, None] * self.alpha_hat) + 2 * torch.eye(K_hat_xx.shape[0])
        logdet = torch.logdet(M)

        log_h_term = -2 * torch.sum(2 * torch.log(torch.abs(self.alpha_hat)))
        quadr_term = self.alpha_hat @ K_hat_xx @ self.alpha_hat
        log_det_term = logdet
        v_term = (1. / (1. + L)).log().sum()
        mll = log_h_term - 0.5 * (quadr_term - v_term + log_det_term)
        return mll

    def train(self):
        K_hat_xx, _ = self.K_hat(self.theta_hat, self.X)
        self.laplace_map(K_hat_xx)

    def train_full(self):

        theta_init = self.theta_hat.clone()
        last_gradient = torch.zeros_like(theta_init)
        iterations = 0

        while True:

            K_hat_xx, _ = self.K_hat(theta_init, self.X)
            self.laplace_map(K_hat_xx, train_params=True)

            theta_init.requires_grad = True
            theta_init.grad = torch.zeros_like(theta_init)

            loss = self.marginal_log_likelihood(theta_init)
            loss.backward()

            gradient = theta_init.grad

            if iterations > 0:
                # tricky stopping cretirion
                # convergence is not guaranteed there
                # should work this optimization better
                # in the future (explicit + implicit gradient)
                diff = (gradient[0] - last_gradient[0]).abs()
                if diff <= 1e-4 or diff >= 200:
                    break

            theta_init = theta_init - 0.0001 * gradient
            theta_init = theta_init.abs().detach()

            if iterations > 0:
                last_gradient = gradient

            iterations += 1

        self.theta_hat = theta_init

    def predict(self, X_star: torch.Tensor):
        X_star_temp = torch.clone(X_star)
        X_star_translated = translate_data(X_star_temp, self.domain_bounds)
        K_x_xstar = self.K_hat(self.theta_hat, self.X, X_star_translated)
        intensity = 0.5 * (self.alpha_hat @ K_x_xstar ) ** 2
        return intensity.detach()

    def get_sample(self, X_star, num_samples:int=1):
        X_star_temp = torch.clone(X_star)
        X_star_translated = translate_data(X_star_temp, self.domain_bounds)

        K_xx,_ = self.K_hat(self.theta_hat, self.X)
        K_x_xstar = self.K_hat(self.theta_hat, self.X, X_star_translated)
        K_xstar_x = K_x_xstar.t()
        K_xstar_xstar,_ = self.K_hat(self.theta_hat, X_star_translated)


        S = K_xx * (self.alpha_hat[:, None] * self.alpha_hat) + 2 * torch.eye(K_xx.shape[0])
        S_inv = torch.linalg.inv(S)

        pred_mean = self.alpha_hat @ K_x_xstar
        pred_cov = K_xstar_xstar - ((K_xstar_x * self.alpha_hat) @ S_inv @ (self.alpha_hat[:, None] * K_x_xstar))
        pred_cov += torch.eye(pred_cov.shape[0]) * 1e-4

        try:
            sample = torch.distributions.MultivariateNormal(pred_mean, pred_cov).sample(torch.Size([num_samples]))
            sample = 0.5 * sample ** 2
            return sample
        except:
            mn = multivariate_normal(mean=pred_mean.numpy(), cov=pred_cov.numpy())
            sample = mn.rvs(size=num_samples)
            sample = torch.tensor(sample)
            sample = 0.5 * sample ** 2
            return sample
