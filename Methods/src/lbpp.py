import torch
import matplotlib.pyplot as plt
from src.utils.data_loader import load_synth_data
from torchmin import ScipyMinimizer
from src.kernels.concrete_kernels import SquaredExponential
import numpy as np

# 1000000-th attempt to get
# LBPP correctly :)
# fingers crossed :P


def nystrom(X: torch.Tensor, max_time: float, kernel: SquaredExponential, num_points: int):

    """
    Method that approximates the
    eigenfunctions and  eigenvectors of
    mercer's expansion of  K
    """

    # compute mat = K(u,u)
    u = torch.linspace(0, max_time, num_points).reshape(-1, 1)
    K_uu = kernel(u)

    # now approximate φ_i and λ_i
    L, Q = torch.linalg.eig(K_uu)
    L, Q = L.real, Q.real

    PHI_X = ( (torch.sqrt(torch.tensor(num_points)) / L) * ( kernel(X, u) @ Q ) ).t()

    L = L / num_points
    return PHI_X, torch.diag(L)

def objective(w: torch.Tensor, PHI_X: torch.Tensor, M: torch.Tensor):
    f = w @ PHI_X
    data_term = (0.5 * f ** 2).log().sum()
    reg_term = 0.5 * (w @ M @ w)
    return -data_term + reg_term

def optimize_mode(PHI_X: torch.Tensor, L: torch.Tensor) -> torch.Tensor:
    M = torch.linalg.inv(L) + torch.eye(L.shape[0])

    w_hat = torch.rand(PHI_X.shape[0], requires_grad=True, dtype=torch.float)
    params = [w_hat]
    tol = 1e-9
    method = 'L-BFGS-B'
    bounds =  [(0., torch.inf)]
    optimizer = ScipyMinimizer(
        params=params,
        method=method,
        tol=tol,
    )

    def closure():
        optimizer.zero_grad()
        loss = objective(w_hat, PHI_X, M)
        loss.backward()
        return loss

    optimizer.step(closure)
    return w_hat.detach()

def laplace_approximation(PHI_X: torch.Tensor, L: torch.Tensor, num_examples: int):
    # compute the posterior mode
    w_hat = optimize_mode(PHI_X, L)

    # compute the posterior covariance matrix
    f = w_hat @ PHI_X
    a = 2 / f
    D = 0.5 * torch.eye(num_examples)
    V = a * PHI_X

    W = V @ D @ V.t()
    I = torch.eye(PHI_X.shape[0])
    L_inv = torch.linalg.inv(L)
    Q_inv = I + L_inv + W
    Q = torch.linalg.inv(Q_inv)

    return w_hat, Q

def predictive_distribution(w_hat: torch.Tensor, Q: torch.Tensor, PHI_X: torch.Tensor):

    mu = w_hat @ PHI_X
    sigma_squared = torch.diag(PHI_X.t() @ Q @ PHI_X)

    a = (mu ** 2 + sigma_squared) ** 2 / (2 * sigma_squared * (2 * mu ** 2 + sigma_squared))
    b = (2 * mu ** 2 * sigma_squared + sigma_squared ** 2) / (mu ** 2 + sigma_squared)

    return torch.distributions.Gamma(concentration=a, rate=1/b)

def main():
    # simulate observations
    intensity_fn = lambda t: 2 * torch.exp(- t / 15) + torch.exp(-1. * ((t - 25.) / 10.) ** 2)
    max_time = 50.
    bound = 2.2
    X = load_synth_data(intensity_fn, max_time, bound)

    # model parameters
    lengthscale = torch.tensor([9.])
    variance = torch.tensor([0.09])
    kernel = SquaredExponential({'variance': variance,  'lengthscales': lengthscale})



    # approximated basis functions and eigenvalues
    num_points = 10
    PHI_X, L = nystrom(X, max_time, kernel, num_points)

    # laplace approximation
    w_hat, Q = laplace_approximation(PHI_X, L, len(X))

    # predictions
    X_test = torch.linspace(0, max_time, 200).reshape(-1, 1)
    PHI_X_test, _ = nystrom(X_test, max_time, kernel, num_points)

    pred_int = predictive_distribution(w_hat, Q, PHI_X_test)
    pred_mean =  torch.mean(pred_int.sample(torch.Size([1000])), dim=0)

    plt.plot(X_test, 2 * torch.exp(- X_test / 15) + torch.exp(-1. * ((X_test - 25.) / 10.) ** 2))
    plt.plot(X_test, pred_mean)
    plt.scatter(X, torch.zeros_like(X), marker='|', c='black')
    plt.show()


