import torch
import gpflow.kernels
from src.utils.data_loader import load_real_data
from src.models.rkhs.model import RKHS
from src.models.vbpp.vbpp_wrapper import VbppWrapper
from src.models.lbpp.model import LBPP
from src.kernels.concrete_kernels import SquaredExponential
from src.utils.plotter import plot_2D_results
from src.utils.metrics import get_poisson_realization2D
from src.utils.results_saver import save_model_realizations

if __name__ == '__main__':

    torch.manual_seed(15)
    dataset_name = 'whiteoak'
    X = load_real_data(dataset_name)
    domain = torch.tensor([[0., 1.], [0., 1.]])


    # create a grid for predictions
    # modify the limits according to the dataset
    # can be easily done by the models domain
    X_axis = torch.linspace(domain[0][0], domain[0][1], 100)
    Y_axis = torch.linspace(domain[1][0], domain[1][1], 100)
    X_axis, Y_axis = torch.meshgrid(X_axis, Y_axis, indexing='ij')
    X_test = torch.stack((X_axis, Y_axis), dim=2).reshape(-1, 2)

    # --------------------------------------------RKHS--------------------------------------------------------#
    kernel = SquaredExponential({'variance': torch.tensor([1.]), 'lengthscales': torch.tensor([1.])})
    rkhs_model = RKHS(X, kernel, m=200, n=20, low_rank=True)
    params = {'scale': torch.tensor([200.]), 'gamma': torch.Tensor([0.15]), 'lengthscale': torch.tensor([0.45])}
    rkhs_model.train(params)
    pred_intensity = rkhs_model.predict(X_test)
    plot_2D_results(X, X_axis, Y_axis, pred_intensity, method='rkhs', dataset=dataset_name)
    # --------------------------------------------------------------------------------------------------------#

    # ------------------------------------------------VBPP-------------------------------------------------------------#
    # initilize the model
    kernel = gpflow.kernels.SquaredExponential(1., 0.1)
    vbpp_model = VbppWrapper(
        X=X,
        kernel=kernel,
        M=7
    ).train()

    pred_intensity, _, _ = vbpp_model.predict(X_test)
    plot_2D_results(X, X_axis, Y_axis, pred_intensity, method='vbpp', dataset=dataset_name)
    # -----------------------------------------------------------------------------------------------------------------#

    # ------------------------------------------------LBPP-------------------------------------------------------------#
    lbpp_model = LBPP(X, N=32, a=torch.Tensor([0.002]), b=torch.Tensor([0.002]), m=2,
                      domain_bounds=torch.tensor([[0., 1.], [0., 1.]]))
    lbpp_model.train()

    pred_intensity = lbpp_model.predict(X_test)
    plot_2D_results(X, X_axis, Y_axis, pred_intensity, method='lbpp', dataset=dataset_name)
    # -----------------------------------------------------------------------------------------------------------------#
