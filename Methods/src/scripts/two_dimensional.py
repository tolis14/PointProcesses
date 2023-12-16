import torch
import gpflow.kernels
from src.utils.data_loader import load_real_data
from src.models.rkhs.model import RKHS
from src.models.vbpp.vbpp_wrapper import VbppWrapper
from src.kernels.concrete_kernels import SquaredExponential
from src.utils.plotter import plot_2D_results

if __name__ == '__main__':

    # get the name of the selected dataset
    # and load the dataset
    # Check the folder data for all the
    # available datasets
    dataset_name = 'redwoodfull'
    X = load_real_data(dataset_name)

    # create a grid for predictions
    # modify the limits according to the dataset
    # can be easily done by the models domain
    X_axis = torch.linspace(0, 1, 100)
    Y_axis = torch.linspace(0, 1, 100)
    X_axis, Y_axis = torch.meshgrid(X_axis, Y_axis, indexing='ij')
    X_test = torch.stack((X_axis, Y_axis), dim=2).reshape(-1, 2)


    #--------------------------------------------RKHS-------------------------------------------------------#
    #initilize the model
    kernel = SquaredExponential({'variance': torch.tensor([1.]), 'lengthscales': torch.tensor([1.])})
    rkhs_model = RKHS(X, kernel, m=400, n=20, low_rank=True)

    # choose hyperparameter space for the model
    # Here it is inevitable to make some assumption
    # and there is not a straightforward way on
    # how to choose them
    # Different datasets might require different hyperparameters
    # initialization so just modify the following Tensors
    scales = torch.tensor([.05, .1, .2, .5, 1, 2, 3, 4, 5, 10, 50, 100, 150, 200])
    gammas = torch.tensor([.05, .1, .25, .5, 1])
    lengthscales = torch.tensor([0.01, 0.02, 0.03, 0.04, 0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95])

    # train the model
    #params = {'scales': scales, 'gammas': gammas, 'lengthscales': lengthscales, 'k': 2, 'r':2}
    #rkhs_model.train_real_data(params)
    #print(rkhs_model)

    # train the model for a fixed set
    # of hyperparameters since CV is slow
    # this is mainly for demonstration
    params = {'scale': torch.tensor([20.]), 'gamma': torch.Tensor([0.1]), 'lengthscale': torch.tensor([0.35])}
    rkhs_model.train(params)
    pred_intensity = rkhs_model.predict(X_test)
    plot_2D_results(X, X_axis, Y_axis, pred_intensity, method='RKHS')
    #-----------------------------------------------------------------------------------------------------------------#


    #------------------------------------------------VBPP-------------------------------------------------------------#
    #initilize the model
    kernel = gpflow.kernels.SquaredExponential(1., 0.1)
    model = VbppWrapper(
        X=X,
        kernel=kernel,
        M=5
    ).train()

    pred_intensity, _, _ = model.predict(X_test)
    plot_2D_results(X, X_axis, Y_axis, pred_intensity, method='VBPP')
    #-----------------------------------------------------------------------------------------------------------------#