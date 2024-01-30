import time
import torch
import gpflow.kernels
from src.utils.data_loader import load_synth_data
from src.models.rkhs.model import RKHS
from src.models.vbpp.vbpp_wrapper import VbppWrapper
from src.models.lbpp.model import LBPP
from src.kernels.concrete_kernels import SquaredExponential
from src.utils.plotter import plot_1D_results
from src.utils.metrics import get_poisson_realization
from src.utils.results_saver import save_model_realizations

if __name__ == '__main__':


    # simulate the data
    torch.manual_seed(3)
    intensity_fn = lambda t: 2 * torch.exp(- t / 15) + torch.exp(-1. * ((t - 25.) / 10.) ** 2)
    max_time = 50.
    bound = 2.2
    data = [load_synth_data(intensity_fn, max_time, bound) for _ in range(3)]

    X_test = torch.linspace(0, max_time, 100).reshape(-1, 1)
    ground_truth = intensity_fn(X_test)

    pred_intensities = []
    times = []
    methods = ['RKHS', 'VBPP', 'LBPP', 'OSGCP']


    # --------------------------------------------------------RKHS----------------------------------------------------#
    # initialize the model
    kernel = SquaredExponential({'variance': torch.tensor([1.]), 'lengthscales': torch.tensor([1.])})
    rkhs_model = RKHS(data[0], kernel, m=32, n=32, low_rank=False)

    # choose hyperparameter space for the model
    # Here it is inevitable to make some assumption
    # and there is not a straightforward way on
    # how to choose them
    scales = torch.tensor([0.1, 0.2, 0.3, 0.5, 1., 1.5])
    gammas = torch.tensor([0.75, 1., 1.25, 1.5])
    lengthscales = torch.tensor([0.5, 1., 2.5, 3.0, 4.0])

    # train the model
    params = {'scales': scales, 'gammas': gammas, 'lengthscales': lengthscales, 'true_intensity': intensity_fn}
    start = time.time()
    rkhs_model.train_synthetic_data(data, params)
    end = time.time()
    pred_intensities.append(rkhs_model.predict(X_test))
    times.append(end - start)
    #-----------------------------------------------------------------------------------------------------------------#


    #--------------------------------------------------------VBPP----------------------------------------------------#
    # initilize and train the model

    vbpp_model = VbppWrapper(
        X=data[0],
        kernel=gpflow.kernels.SquaredExponential(variance=1.5, lengthscales=3.5),
        M=32
    )
    start = time.time()
    vbpp_model.train()
    end = time.time()
    pred_intensities.append(vbpp_model.predict(X_test)[0])
    times.append(end - start)
    #----------------------------------------------------------------------------------------------------------------#


    #--------------------------------------------------------LBPP----------------------------------------------------#
    #initilize and train the model
    lbpp_model = LBPP(data[0], N=32, a=torch.Tensor([0.08]), b=torch.Tensor([16.]), m=2,
                      domain_bounds=torch.tensor([[0., max_time]]))
    start = time.time()
    lbpp_model.train()
    end = time.time()
    pred_intensities.append(lbpp_model.predict(X_test))
    times.append(end - start)
    # ----------------------------------------------------------------------------------------------------------------#


    # -------------------------------------------------------OSGCP----------------------------------------------------#
    osgcp_predict = torch.load('..//..//william_results//synth1_posterior_mean.pt')
    pred_intensities.append(osgcp_predict)
    # ----------------------------------------------------------------------------------------------------------------#

    # plot results
    plot_1D_results(data[0], X_test, pred_intensities, methods, ground_truth, 'synth1')



    # get Poisson Process realizations for each model
    #vbpp_samples = [get_poisson_realization(vbpp_model, max_time, X_test) for _ in range(100)]
    #lbpp_samples = [get_poisson_realization(lbpp_model, max_time, X_test) for _ in range(100)]
    #rkhs_samples = [get_poisson_realization(rkhs_model, max_time, X_test) for _ in range(100)]

    #save_model_realizations({'lbpp': lbpp_samples, 'vbpp': vbpp_samples, 'rkhs': rkhs_samples}, 'synth1')


    # get running times
    print('times', times)