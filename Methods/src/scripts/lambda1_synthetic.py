import torch
import gpflow.kernels
from src.utils.data_loader import load_synth_data
from src.models.rkhs.model import RKHS
from src.models.vbpp.vbpp_wrapper import VbppWrapper
from src.kernels.concrete_kernels import SquaredExponential
from src.utils.plotter import plot_1D_results

if __name__ == '__main__':


    # simulate the data
    #torch.manual_seed(3)
    intensity_fn = lambda t: 2 * torch.exp(- t / 15) + torch.exp(-1. * ((t - 25.) / 10.) ** 2)
    max_time = 50.
    bound = 2.2
    data = [load_synth_data(intensity_fn, max_time, bound) for _ in range(3)]

    X_test = torch.linspace(0, max_time, 100).reshape(-1, 1)
    ground_truth = intensity_fn(X_test)

    pred_intensities = []
    methods = ['RKHS', 'VBPP']


    # --------------------------------------------------------RKHS----------------------------------------------------#
    # initialize the model
    kernel = SquaredExponential({'variance': torch.tensor([1.]), 'lengthscales': torch.tensor([1.])})
    rkhs_model = RKHS(data[0], kernel, m=100, n=100, low_rank=False)

    # choose hyperparameter space for the model
    # Here it is inevitable to make some assumption
    # and there is not a straightforward way on
    # how to choose them
    scales = torch.tensor([0.1, 0.2, 0.3, 0.5, 1., 1.5])
    gammas = torch.tensor([0.75, 1., 1.25, 1.5])
    lengthscales = torch.tensor([0.5, 1., 2.5, 3.0, 4.0])

    # train the model
    params = {'scales': scales, 'gammas': gammas, 'lengthscales': lengthscales, 'true_intensity': intensity_fn}
    rkhs_model.train_synthetic_data(data, params)
    pred_intensities.append(rkhs_model.predict(X_test))
    #-----------------------------------------------------------------------------------------------------------------#


    #--------------------------------------------------------VBPP----------------------------------------------------#
    # initilize and train the model
    vbpp_model = VbppWrapper(
        X=data[0],
        kernel=gpflow.kernels.SquaredExponential(variance=1., lengthscales=3.5),
        M=20
    ).train()
    pred_intensities.append(vbpp_model.predict(X_test)[0])
    #----------------------------------------------------------------------------------------------------------------#

    # plot results
    plot_1D_results(data[0], X_test, pred_intensities, methods, ground_truth)
