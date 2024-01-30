# File just for checking purposes
import torch
import matplotlib.pyplot as plt
from src.models.lbpp.model import LBPP
from src.utils.plotter import plot_2D_results
import pandas as pd
import time
from src.utils.data_loader import load_synth_data

if __name__ == '__main__':


    path = '..//data//redwood.csv'
    df = pd.read_csv(path)
    X = torch.tensor(df.values, dtype=torch.float)

    lbpp_model = LBPP(X, N=32, a=torch.Tensor([0.0038]), b=torch.Tensor([0.022]), m=2,
                      domain_bounds=torch.tensor([[0., 1.], [0., 1.]]))

    start = time.time()
    lbpp_model.train_full()
    end = time.time()
    print(end - start)

    X_axis = torch.linspace(0, 1, 100)
    Y_axis = torch.linspace(0, 1, 100)
    X_axis, Y_axis = torch.meshgrid(X_axis, Y_axis, indexing='ij')
    X_test = torch.stack((X_axis, Y_axis), dim=2).reshape(-1, 2)

    pred = lbpp_model.predict(X_test)

    plot_2D_results(X, X_axis, Y_axis, pred, 'LBPP', 'whiteoak')



    path = '..//data//synth1.csv'
    df = pd.read_csv(path)
    X = torch.tensor(df.values, dtype=torch.float)

    max_time = 50.
    X_test = torch.linspace(0, max_time, 100).reshape(-1, 1)

    model = LBPP(X, N=32, a=torch.Tensor([0.15]), b=torch.Tensor([16.]), m=2, domain_bounds=torch.tensor([[0., max_time]]))


    start = time.time()
    model.train_full()
    end = time.time()
    print(end - start)

    pred = model.predict(X_test)

    plt.plot(X_test, pred)
    plt.scatter(model.data, torch.zeros_like(model.data), c='black', s=10., marker='|')
    plt.show()



