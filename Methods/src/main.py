import gpytorch.kernels
import torch
from src.data_loader import load_synth_data, load_real_data
from kernels.kernels_wrapper import SquaredExponential
from src.models.vbpp.model import VBPP



if __name__ == '__main__':

    #intensity = lambda s: 5 * torch.sin(s) + 6
    #max_time = 5.
    #bound = 11.
    #X = load_synth_data(intensity, max_time, bound)

    X = load_real_data('redwoodfull')

    variance = torch.tensor(0.5)
    lengthscales = torch.tensor(0.2)
    kernel = SquaredExponential(variance, lengthscales)
    num_points = 20
    q_mu = None
    q_S = None
    u_bar = None

    method = VBPP(X, kernel, num_points)

    print(method.T)
    print(method.inducing_points)
    exit(0)

