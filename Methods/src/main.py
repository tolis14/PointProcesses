import gpytorch.kernels
import torch
from src.data_loader import load_synth_data, load_real_data
from kernels.kernels_wrapper import SquaredExponential
from src.models.vbpp.model import VBPP



if __name__ == '__main__':

    intensity = lambda s: 5 * torch.sin(s) + 6
    max_time = 5.
    bound = 11.

    X = load_synth_data(intensity, max_time, bound)
    #X = load_real_data('redwoodfull')

    d = X.shape[1]

    variance = torch.tensor(1.)
    lengthscales = torch.tensor(1.)
    kernel = SquaredExponential(variance, lengthscales)
    num_points = int(20 ** (1/d))
    q_mu = torch.zeros(num_points ** d)
    q_S = torch.eye(num_points ** d)
    u_bar = torch.zeros(num_points ** d)

    method = VBPP(X, kernel, num_points, q_mu, q_S, u_bar)


    exit(0)

