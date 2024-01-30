import torch
from src.utils.data_loader import load_real_data
from src.models.rkhs.model import RKHS
from src.kernels.concrete_kernels import SquaredExponential
from src.utils.plotter import plot_2D_results
import time


if __name__ == '__main__':

    # get the name of the selected dataset
    # and load the dataset
    # Check the folder data for all the
    # available datasets
    dataset_name = 'whiteoak'
    X = load_real_data(dataset_name)

    # create a grid for predictions
    # modify the limits according to the dataset
    # can be easily done by the models domain
    X_axis = torch.linspace(0, 1, 100)
    Y_axis = torch.linspace(0, 1, 100)
    X_axis, Y_axis = torch.meshgrid(X_axis, Y_axis, indexing='ij')
    X_test = torch.stack((X_axis, Y_axis), dim=2).reshape(-1, 2)

    torch.manual_seed(15)

    #initilize the model
    kernel = SquaredExponential({'variance': torch.tensor([1.]), 'lengthscales': torch.tensor([1.])})
    rkhs_model = RKHS(X, kernel, m=200, n=20, low_rank=True)

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
    params = {'scales': scales, 'gammas': gammas, 'lengthscales': lengthscales, 'k': 2, 'r':2}
    start = time.time()
    rkhs_model.train_real_data(params)
    end = time.time()
    print(end - start)




