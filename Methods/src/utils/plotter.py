import torch
import matplotlib.pyplot as plt

def plot_1D_results(data_points: torch.Tensor,
                    test_points: torch.Tensor,
                    predicted_intensities: list,
                    methods: list,
                    ground_truth: torch.Tensor=None):

    # plot predicted intensities of all methods
    for pred_intensity, method in zip(predicted_intensities, methods):
        plt.plot(test_points, pred_intensity, '--', label=method)

    # plot ground truth if provided
    if ground_truth is not None:
        plt.plot(test_points, ground_truth, c='black', label='true')

    #scatter observations
    plt.plot(data_points, torch.zeros_like(data_points), linestyle='None', marker='|', c='black', markersize=10,
             label='observations')

    plt.title('1D Point Processes')
    plt.xlim((test_points[0], test_points[-1]))
    plt.tight_layout()
    plt.legend()
    plt.show()

def plot_2D_results(data: torch.Tensor,
                    X_axis: torch.Tensor,
                    Y_axis: torch.Tensor,
                    intensity: torch.Tensor,
                    method: str,
                    cmap='YlOrRd'):

    # first reshape intensity so it can
    # be passed correctly to the contourf, contour
    # functions
    num_test_points = len(X_axis)
    intensity = intensity.reshape(num_test_points, num_test_points, -1).squeeze()

    # plot the colormap
    # add the isolines
    # scatter the data points
    contourf = plt.contourf(X_axis, Y_axis, intensity, cmap=cmap, levels=500)
    plt.contour(X_axis, Y_axis, intensity, cmap=cmap)
    plt.scatter(data[:, 0], data[:, 1], s=10., c='black', edgecolors='white')
    plt.colorbar(contourf)
    plt.title(method)
    plt.show()