"""
import numpy as np
#from src.kernels.concrete_kernels import SquaredExponential
#from src.models.vbpp.model import VBPP
#from src.models.rhks.model import RHKS
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
from src.data_loader import load_synth_data, load_real_data
from general_checker import *

if __name__ == '__main__':

    # 1D synthetic dataset
    X = torch.Tensor([
        1.7345089191961298,
        0.7168442153304712,
        1.522658701236610,
        1.996865664270993,
        1.2825802994353745,
        0.9692800188689177,
        1.045222911792033,
        2.9997956125716008,
        3.548130296149658,
        0.10480472773824301,
        3.5439439802042436,
        0.8757914474587192,
        0.632784556201901,
        0.31985619451470104,
        4.331469130152144,
        2.8507816583680965,
        1.2616261024455833,
        2.9145627753909222,
        2.812175716410698,
        4.48798636611463,
        0.58251251159124,
        3.766580122456904,
        2.7322079425448464,
    ]).reshape(-1, 1)
    intensity = lambda t: 5 * torch.sin(t ** 2) + 6
    max_time = 5.

    # 2D synthetic dataset
    X = load_real_data('redwoodfull')

    # Laplace approximation
    d = len(X[0])
    N = 32
    w_hat, hess_inv = laplace_approximation(X, N)

    # Predictive Distribution
    X_test = X
    PHI_X_test = create_feature_matrix(X_test, N)

    pred_mean = w_hat @ PHI_X_test
    pred_var = torch.diag(PHI_X_test.t() @ torch.linalg.inv(hess_inv) @ PHI_X_test)

    a = (pred_mean ** 2 + pred_var) ** 2 / ((2 * pred_var) *  (2 * pred_mean ** 2 + pred_var))
    b = (2 * pred_mean ** 2 * pred_var + pred_var ** 2) / (pred_mean ** 2 + pred_var)

    sample = torch.distributions.Gamma(concentration=a, rate=1/b).sample(torch.Size([5000]))
    intensity_mean = torch.mean(sample, dim=0)
    print(intensity_mean)

    """
    plt.plot(X_test, intensity_mean)
    plt.plot(X_test, intensity(X_test))
    plt.plot(X, torch.zeros_like(X), linestyle='None', marker='|', c='black', markersize=10,
             label='observations')
    plt.xlim(0, max_time)
    plt.show()
    """









