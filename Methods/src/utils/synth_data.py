import torch
from torchmin import minimize_constr

class PoissonProcess:
    """
    Class simulating observations from IHPP with
    intensity function λ(t), 0 < t <= max_time
    """
    def __init__(self, intensity: callable, max_time: torch.float, bound:torch.float=0.):
        self._data = None
        self.intensity = intensity
        self.bound = bound
        self.max_time = max_time


    def simulate(self):
        """
        Simulate observations from the IHPP with specified intensity function.
        For simulating paper results, we need to provide the number of events
        that they also used there. If 0 events provided, the N(S) is drawn from
        the Poisson distribution with mean λ*|S|
        """
        #torch.manual_seed(2) #reproducability
        num_of_points = int(torch.distributions.Poisson(rate=self.max_time * self.bound).sample())

        homo_samples,_ = torch.sort(torch.distributions.Uniform(0, self.max_time).sample(torch.Size([num_of_points])))
        inhomo_samples = self._reject(homo_samples)
        self._data = inhomo_samples.reshape(-1, 1)

    def _reject(self, homo_samples: torch.Tensor) -> torch.Tensor:
        """
        :param homo_samples: Samples from the homogeneous Poisson Process
        :return: samples from the inhomogeneous Poisson Process via thinning
        """
        u = torch.rand(len(homo_samples))
        keep_idxs = torch.where(u <= self.intensity(homo_samples) / self.bound, True, False)
        return homo_samples[keep_idxs]

    def get_data(self):
        return self._data


class HomoPoissonProcess:

    def __init__(self, max_time: float, bound: float):

        self.max_time = max_time
        self.bound = bound
        self.data = None

    def simulate(self):
        num_of_points = int(torch.distributions.Poisson(rate=self.max_time * self.bound).sample())
        homo_samples, _ = torch.sort(torch.distributions.Uniform(0, self.max_time).sample(torch.Size([num_of_points])))
        self.data = homo_samples.reshape(-1, 1)
        return self

    def get_data(self):
        return self.data


class HomoPoissonPrcoess2D:
    def __init__(self, domain: torch.Tensor, bound: float):
        self.domain = domain
        self.bound = bound
        self.data = None

    def simulate(self):
        x_min, x_max = self.domain[0][0], self.domain[0][1]
        y_min, y_max = self.domain[1][0], self.domain[1][1]
        total_area = (x_max - x_min) * (y_max - y_min)

        num_points = int(torch.distributions.Poisson(rate=total_area * self.bound).sample())

        x_coords = torch.distributions.Uniform(x_min, x_max).sample(torch.Size([num_points]))
        y_coords = torch.distributions.Uniform(y_min, y_max).sample(torch.Size([num_points]))

        sample = torch.column_stack((x_coords, y_coords))
        self.data = sample
        return self

    def get_data(self):
        return self.data


def reject(points: torch.Tensor, intensity_values: torch.Tensor, bound: torch.Tensor):
    points, intensity_values = points.flatten(), intensity_values.flatten()
    u = torch.rand(len(points))
    keep_idxs = torch.where(u <= intensity_values / bound, True, False)
    return points[keep_idxs]


def reject2D(points: torch.Tensor, intensity_values: torch.Tensor, bound: torch.Tensor):
    intensity_values = intensity_values.flatten()
    u = torch.rand(len(points))
    keep_idxs = torch.where(u <= intensity_values / bound, True, False)
    return points[keep_idxs]
