import torch
from itertools import chain
from src.models.model_base import ModelBase, KernelBase
from src.utils.optimizer import MinimizerWrapper
from src.models.rkhs.cross_validation import get_folders

class RKHS(ModelBase):

    def __init__(self, X: torch.Tensor, kernel: KernelBase,
                 m: int, n: int, low_rank=False):
        super().__init__(X, kernel)
        self.m = m
        self.n = n
        self.low_rank = low_rank

        self.model_params = {'mu': torch.Tensor([1.]), 'gamma': torch.Tensor([1.])}
        self.a = torch.ones(len(X))

        self.grid = self.get_domain_grid()
        self.u = self.grid if X.shape[0] == 1 else self.get_sample_locations()

    def set_scale(self, scale: torch.Tensor):
        self.model_params['mu'] = scale

    def set_gamma(self, gamma: torch.Tensor):
        self.model_params['gamma'] = gamma

    def set_params(self, params: dict):
        self.model_params['mu'] = params['mu']
        self.model_params['gamma'] = params['gamma']

    def get_domain_grid(self) -> torch.Tensor:
        """
        This method constructs a grid of size n ** d
        over the domain of the point Process.
        This grid is used in the calculation of ~K_ux with the Nystrom method,
        which is required for the estimation of the integral term in the unpenalized log likelihood.
        !!Note that according to their paper, one can re-use the grid points as
        integration points.
        :return: grid of size n ** d over the domain
        """

        # get m equally distant points u_dim = (u_1,...,u_m) at each dimension
        # over the domain of Point Process at each dimension
        u_over_dims = []
        for min_val, max_val in zip(self.domain[:, 0], self.domain[:, 1]):
            u = torch.linspace(min_val, max_val, self.n)
            u_over_dims.append(u)

        # fix the m x d grid over the domain
        grid = torch.meshgrid(u_over_dims, indexing='ij')
        grid = torch.vstack([x.flatten() for x in grid]).t()
        return grid

    def get_sample_locations(self) -> torch.Tensor:
        """
        This method gets uniform samples over
        the domain of the Point Process.
        These sample locations are then used for calculating
        the matrix ~K_xx with the Nystrom method
        :return: uniform sample of size m over the domain
        """

        num_samples = self.m
        sample_locations = torch.distributions.Uniform(self.domain[:, 0], self.domain[:, 1])\
                                .sample(torch.Size([num_samples]))
        return sample_locations

    def kxx_nystrom(self, X: torch.Tensor, Y: torch.Tensor=None, top_eigen: int=20):
        """
        This method computes the full covariance
        ~K_xx via the Nystom approximation, using
        the sample locations points.
        See equation (21) in the paper
        :return: Nystrom approximation of ~K_xx
        """
        if Y is None:
            Y = X

        K_uu = self.kernel(self.u)
        K_xu = self.kernel(X, self.u)
        K_yu = K_xu.t() if Y is None else self.kernel(self.u, Y)

        L, Q = torch.linalg.eig(K_uu) if not self.low_rank else torch.lobpcg(K_uu, top_eigen)
        L, Q = torch.diag(L).real, Q.real

        num_points = len(self.u)
        scale = self.model_params['mu']
        gamma = self.model_params['gamma']

        result = K_xu @ Q \
                   @ torch.linalg.inv((scale / num_points) * L ** 2 + gamma * L) \
                   @ Q.t() @ K_yu

        return result

    def kxu_nystrom(self, X: torch.Tensor):
        """
        This method computes the covariance
        ~K_xu via the Nystom approximation
        using the grid points.
        :return: Nystrom approximation of ~K_xu
        """

        Kuu = self.kernel(self.grid)
        Kxu = self.kernel(X, self.grid)
        m = len(self.grid)
        scale = self.model_params['mu']
        gamma = self.model_params['gamma']
        return Kxu @ torch.linalg.inv((scale / m) * Kuu + gamma * torch.eye(m))

    def objective(self, a: torch.Tensor, K_xx_nystrom: torch.Tensor):
        """
        Computes the objective function that we seek to
        minimize see equation (13), start of section 5 INFERENCE
        In their code when optimizing this function they set γ = 1.
        :return: value of the objective function
        """
        gamma = 1.
        f = K_xx_nystrom @ a
        data_term = (self.model_params['mu'] * f ** 2).log().sum()
        reg_term = gamma * (a.t() @ K_xx_nystrom @ a)
        return -data_term + reg_term

    def unpenilized_log_likelihood(self, K_xu_nystrom: torch.Tensor,
                                   K_xx_nystrom: torch.Tensor, a: torch.Tensor):

        scale = self.model_params['mu']
        data_term = (scale * (K_xx_nystrom @ a) ** 2).log().sum()
        integral_term = scale * ((a @ K_xu_nystrom) ** 2).sum() / len(self.grid)
        return -data_term + integral_term

    def train_real_data(self, params: dict):

        # parameters of the cross validation
        k = params['k']
        r = params['r']

        # parameters of the model and the kernel
        scales = params['scales']
        gammas = params['gammas']
        lengthscales = params['lengthscales']

        # create k-folders and hyperparameters grid
        replications = get_folders(len(self.X), k, r)
        params_grid = torch.cartesian_prod(scales, gammas, lengthscales)

        # train the model via cross validation
        optim = MinimizerWrapper(self.objective)
        errors = torch.zeros(len(params_grid))


        for replication in replications:
            folders = replication
            for f in range(k):
                # get train and test data from the folders
                train_indices = list(chain.from_iterable(folders[:f] + folders[f+1:]))
                test_indices = folders[f]
                train_data = self.X[train_indices]
                test_data = self.X[test_indices]
                model_idx = 0
                # train the model for each set of hyperparams
                for params_tuple in params_grid:
                    scale, gamma, lengthscale = params_tuple
                    self.set_params({'mu': scale, 'gamma': gamma})
                    self.kernel.set_lengthscales(lengthscale)

                    K_xx_nystrom = self.kxx_nystrom(train_data)
                    a_hat = optim.optimize(torch.ones(len(train_data)), K_xx_nystrom)

                    K_xu_nystrom = self.kxu_nystrom(train_data)
                    K_xx_nystrom_test = self.kxx_nystrom(test_data, train_data)

                    # since we do not know the ground truth
                    # we choose to evaluate the model with
                    # the log likelihood
                    errors[model_idx] += self.unpenilized_log_likelihood(K_xu_nystrom, K_xx_nystrom_test, a_hat)
                    print(f, model_idx)
                    model_idx += 1

        errors /= (r * k) # get average error

        # get best parameters
        best_scale, best_gamma, best_lengthscale = params_grid[torch.argmin(errors)]
        self.set_params({'mu': best_scale, 'gamma': best_gamma})
        self.kernel.set_lengthscales(best_lengthscale)

        # optimize the a with whole dataset now
        K_xx_nystrom = self.kxx_nystrom(self.X)
        self.a = optim.optimize(torch.ones(len(self.X)), K_xx_nystrom)

    def train_synthetic_data(self, samples: list, params: dict):
        # create cross validation folders
        num_samples = len(samples)
        indices = torch.arange(0, num_samples, 1)
        train_indices, test_indices = torch.meshgrid(indices, indices, indexing="ij")
        train_test_indices = torch.vstack((train_indices.flatten(), test_indices.flatten())).T
        retain = [i for i in range(len(train_test_indices)) if train_test_indices[i, 0] != train_test_indices[i, 1]]
        train_test_indices = train_test_indices[retain]

        # parameters of the model and the kernel
        scales = params['scales']
        gammas = params['gammas']
        lengthscales = params['lengthscales']
        params_grid = torch.cartesian_prod(scales, gammas, lengthscales)

        # true intensity
        true_intensity = params['true_intensity']

        #init optimizer
        optim = MinimizerWrapper(self.objective)
        errors = torch.zeros(len(params_grid))

        for elem in train_test_indices:
            train_data = samples[elem[0]]
            test_data = samples[elem[1]]

            model_idx = 0
            for params_tuple in params_grid:
                scale, gamma, lengthscale = params_tuple
                self.set_params({'mu': scale, 'gamma': gamma})
                self.kernel.set_lengthscales(lengthscale)

                K_xx_nystrom = self.kxx_nystrom(train_data)
                a_hat = optim.optimize(torch.ones(len(train_data)), K_xx_nystrom)

                # since we know the real intensity
                # we choose to evaluate the model
                # at test dataset with MSE error
                intensity_hat = scale * (K_xx_nystrom @ a_hat) ** 2
                ground_truth = true_intensity(test_data)
                mse = torch.mean((intensity_hat - ground_truth) ** 2)

                errors[model_idx] += mse
                model_idx += 1
                #print(model_idx)

        errors /= len(train_test_indices)  # get average error

        # get best parameters
        best_scale, best_gamma, best_lengthscale = params_grid[torch.argmin(errors)]
        self.set_params({'mu': best_scale, 'gamma': best_gamma})
        self.kernel.set_lengthscales(best_lengthscale)

        # optimize the a with whole dataset now
        K_xx_nystrom = self.kxx_nystrom(self.X)
        self.a = optim.optimize(torch.ones(len(self.X)), K_xx_nystrom)

    def train(self, params: dict):
        # train the model with a
        # fixed set of hyperparameters
        # since the full cross validation
        # training might be slow
        scale = params['scale']
        gamma = params['gamma']
        lengthscale = params['lengthscale']
        self.set_scale(scale)
        self.set_gamma(gamma)
        self.kernel.set_lengthscales(lengthscale)

        optim = MinimizerWrapper(self.objective)
        K_xx_nystrom = self.kxx_nystrom(self.X)
        self.a = optim.optimize(torch.ones(len(self.X)), K_xx_nystrom)

    def predict(self, X_star: torch.Tensor):
        f = self.kxx_nystrom(X_star, self.X) @ self.a
        intensity = self.model_params['mu'] * f ** 2
        return intensity

    def get_sample(self, X_star: torch.Tensor, num_samples:int=1) -> torch.Tensor:
        """
        Normally this method is implemented for Bayesian methods
        i.e a predictive distribution is available.
        This method follows a frequentist approach and
        does not support a predictive distribution.
        However, for comparison reasons with the other models,
        we will just draw the posterior mean at X_star locations
        In other words we just make predictions in the new data points X*.
        Since this is not a random sample, but the posterior mean,
        the results highly favors this method [I wish it was Bayesian....]
        """
        return self.predict(X_star).reshape(1, -1)

    def __str__(self):
        return 'scale: ' + str(self.model_params['mu']) \
               + '\ngamma: ' + str(self.model_params['gamma']) \
               + '\nlengthscale:' + str(self.kernel.get_lengthscales().item())