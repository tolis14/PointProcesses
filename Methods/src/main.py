import torch
from src.data_loader import load_synth_data, load_real_data
from src.kernels.concrete_kernels import SquaredExponential
from src.models.vbpp.model import VBPP

def initialize_L(M: int):
    l = torch.zeros(int((M + 1) * M / 2))
    k = 0
    for i in range(M):
        for j in range(0, i+1):
            if j == i:
                l[k] = 1
            k += 1
    return l

if __name__ == '__main__':

    intensity = lambda s: 5 * torch.sin(s) + 6
    max_time = 5.
    bound = 11.

    X = load_synth_data(intensity, max_time, bound)
    #X = load_real_data('redwoodfull')

    d = X.shape[1]

    variance = torch.tensor([1.])
    lengthscales = torch.ones(d)
    params = {'variance': variance, 'lengthscales': lengthscales}
    kernel = SquaredExponential(params)

    num_points = int(25 ** (1/d))

    q_mu = torch.ones(num_points ** d)
    L = initialize_L(num_points ** d)
    u_bar = torch.zeros(num_points ** d)

    method = VBPP(X, kernel, num_points, q_mu, L, u_bar)
    method.train()
    method.predict()




