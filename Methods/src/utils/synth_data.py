import torch

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
