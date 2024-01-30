import torch
from src.utils.synth_data import HomoPoissonProcess, reject, HomoPoissonPrcoess2D, reject2D

def MSE(pred_intensity, ground_truth):
    pass

def get_poisson_realization(model, max_time: float, X_star: torch.Tensor):
    sample_for_max = model.get_sample(X_star)
    bound = torch.max(sample_for_max)

    points = HomoPoissonProcess(max_time, bound.item()).simulate().get_data()
    sample_for_thinning = model.get_sample(points)

    method_points = reject(points, sample_for_thinning, bound)
    return method_points


def get_poisson_realization2D(model, domain: torch.Tensor, X_star: torch.Tensor):
    sample_for_max = model.get_sample(X_star)
    bound = torch.max(sample_for_max)
    points = HomoPoissonPrcoess2D(domain, bound.item()).simulate().get_data()

    sample_for_thinning = model.get_sample(points, num_samples=1)
    method_points = reject2D(points, sample_for_thinning, bound)

    return method_points
