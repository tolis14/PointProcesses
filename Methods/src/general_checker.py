import torch
from torchmin import ScipyMinimizer
from sklearn.model_selection import KFold

#General for VBPP
"""
def k(x: torch.Tensor, y: torch.Tensor, var: torch.Tensor, lengthscale: torch.Tensor):
    return var * torch.exp(-(x - y) ** 2 / (2 * lengthscale))

def K(X: torch.Tensor, Y: torch.Tensor, var: torch.Tensor, lengthscale: torch.Tensor):
    N = len(X)
    M = len(Y)
    jitter = 1e-5
    cov_matrix = torch.zeros((N, M))
    for i in range(N):
        for j in range(M):
            cov_matrix[i, j] = k(X[i], Y[j], var, lengthscale)
    if N == M:
        cov_matrix += torch.eye(N) * jitter
    return cov_matrix

def psi(z1: torch.Tensor, z2: torch.Tensor, var: torch.Tensor, lengthscale: torch.Tensor, T: float):
    z_bar = 0.5 * (z1 + z2)

    exp_term = - 0.5 * torch.sqrt(torch.pi * lengthscale) * torch.exp(-(z1 - z2) ** 2 / (4 * lengthscale))

    left_erf = torch.erf((z_bar - T) / torch.sqrt(lengthscale))
    right_erf = torch.erf((z_bar - 0.) / torch.sqrt(lengthscale))
    erf_term = left_erf - right_erf

    return (var ** 2) * exp_term * erf_term

def Psi(Z: torch.Tensor, var: torch.Tensor, lengthscale: torch.Tensor, T:float):
    M = len(Z)
    jitter = 1e-5
    PSI = torch.zeros((M, M))
    for i in range(M):
        for j in range(M):
            PSI[i, j] = psi(Z[i], Z[j], var, lengthscale, T)
    PSI += torch.eye(M) * jitter
    return PSI

def full_q_S(vec_L: torch.Tensor, M: int):
    L = torch.zeros(size=(M, M))
    L[torch.tril_indices(M, M, offset=0).tolist()] = vec_L
    S = L @ L.t()
    return S

def initialize_L(M: int):
    l = torch.zeros(int((M + 1) * M / 2), dtype=torch.float)
    idx = 0
    for i in range(M):
        for j in range(0, i+1):
            if j == i:
                l[idx] = 1.
            idx += 1
    return l

def kl_distance(K_zz: torch.Tensor, K_zz_inv: torch.Tensor, q_mu: torch.Tensor, q_S: torch.Tensor, u_bar: torch.Tensor):
    first_term = torch.trace(K_zz_inv @ q_S)
    second_term = -torch.logdet(K_zz) + torch.logdet(q_S)
    third_term = -len(u_bar)
    fourth_term = (u_bar - q_mu) @ K_zz_inv @ (u_bar - q_mu)
    return 0.5 * (first_term + second_term + third_term+ fourth_term)

def integral_term(q_mu: torch.Tensor, q_S: torch.Tensor, K_zz_inv: torch.Tensor, Z: torch.Tensor, variance: torch.Tensor, lengthscale: torch.Tensor, T: float):
    PSI = Psi(Z, variance, lengthscale, T)

    left_term = q_mu @ K_zz_inv @ PSI @ K_zz_inv @ q_mu
    right_term = variance * T - torch.trace(K_zz_inv @ PSI) + torch.trace(K_zz_inv @ q_S @ K_zz_inv @ PSI)

    return left_term + right_term

def data_term(X: torch.Tensor, q_mu: torch.Tensor,  q_S: torch.Tensor, K_zz_inv: torch.Tensor, Z: torch.Tensor, variance: torch.Tensor, lengthscale: torch.Tensor):
    N = len(X)
    mu_hat = torch.zeros(N)
    sigma_squared_hat = torch.zeros(N)

    for i in range(N):
        mu_hat[i] = K(X[i], Z, variance, lengthscale) @ K_zz_inv @ q_mu

        K_xz = K(X[i], Z, variance, lengthscale)
        K_zx = K_xz.t()

        sigma_squared_hat[i] = k(X[i], X[i], variance, lengthscale) \
                               - K_xz @ K_zz_inv @ K_zx \
                               + K_xz @ K_zz_inv @ q_S @ K_zz_inv @ K_zx

    G_arg = -mu_hat ** 2 / (2 * sigma_squared_hat)
    log_arg = 0.5 * sigma_squared_hat
    C = 0.5721566
    return (-np_Gtilde_lookup(G_arg.detach().numpy())[0] + torch.log(log_arg) - C).sum()

def elbo(q_mu: torch.Tensor, vec_L: torch.Tensor, u_bar: torch.Tensor, variance: torch.Tensor, lengthscale: torch.Tensor, Z: torch.Tensor, X: torch.Tensor, T:float):
    K_zz = K(Z, Z, variance, lengthscale)
    K_zz_inv = torch.linalg.inv(K_zz)

    M = len(Z)
    q_S = full_q_S(vec_L, M)

    kl_term = kl_distance(K_zz, K_zz_inv, q_mu, q_S, u_bar)
    int_term = integral_term(q_mu, q_S, K_zz_inv, Z, variance, lengthscale, T)
    d_term = data_term(X, q_mu, q_S, K_zz_inv, Z, variance, lengthscale)

    full_elbo = -int_term + d_term - kl_term
    return -full_elbo

def optimize_elbo(X: torch.Tensor, Z: torch.Tensor, u_bar: torch.Tensor, T: float):
    M = len(Z)

    #kernel hyperparameters theta
    var = torch.tensor([1.], requires_grad=True)
    lengthscale = torch.tensor([1.], requires_grad=True)

    #variational mean
    q_mu = torch.ones(M, requires_grad=True)

    #variational covariance in vector form
    vec_L = initialize_L(M)
    vec_L = vec_L.clone().requires_grad_(True)

    #optimizer params
    params = [q_mu, vec_L, var, lengthscale]
    bounds = [(0., torch.inf), (0.01, torch.inf), (0.01, torch.inf), (0.01, torch.inf)]
    tol = 1e-9
    method = 'L-BFGS-B'

    optimizer = ScipyMinimizer(
        params=params,
        method=method,
        bounds=bounds,
        tol=tol
    )

    def closure():
        optimizer.zero_grad()
        loss = elbo(params[0], params[1], u_bar, params[2], params[3], Z, X, T)
        loss.backward()
        print(torch.norm(vec_L.grad, p=2), -loss)
        return loss

    optimizer.step(closure)

    return params

def predict(X_test: torch.Tensor, Z: torch.Tensor, params: list):
    M = len(Z)

    m = params[0]
    S = full_q_S(params[1], M)
    var = params[2]
    lengthscale = params[3]


    K_zz = K(Z, Z, var, lengthscale)
    K_zz_inv = torch.linalg.inv(K_zz)

    K_xz = K(X_test, Z, var, lengthscale)
    K_zx = K_xz.t()
    K_xx = K(X_test, X_test, var, lengthscale)

    pred_mean = K_xz @ K_zz_inv @ m
    pred_cov = K_xx - K_xz @ K_zz_inv @ K_zx + K_xz @ K_zz_inv @ S @ K_zz_inv @ K_zx

    pred_intensity = pred_mean ** 2 + torch.diag(pred_cov)
    return pred_intensity
"""


#General for RHKS
"""
def K(X: torch.Tensor, Y: torch.Tensor, lengthscale: float):
    
    Return the squared exponential kernel
    k(x, y) = exp{(-0.5 * l ** 2) ** (-1) * ||x - y|| ** 2}
    
    jitter = 1e-5
    exp_arg = torch.tensor(-0.5) * ((X[:, None] - Y) ** 2 / (lengthscale ** 2)).sum(dim=-1)
    cov_matrix = torch.exp(exp_arg)
    if len(X) == len(Y):
        cov_matrix += torch.eye(len(X)) * jitter
    return cov_matrix

def K_hat(X: torch.Tensor, Y: torch.Tensor, u: torch.Tensor, a: float, gamma: float, lengthscale: float):
    m = u.shape[0]
    K_uu = K(u, u, lengthscale)

    L, Q = torch.linalg.eig(K_uu)
    L, Q = torch.diag(L).real, Q.real

    K_xu = K(X, u, lengthscale)
    K_uy = K(u, Y, lengthscale)

    res = K_xu @ Q @ torch.linalg.inv((a / m) * L ** 2 + gamma * L) @ Q.t() @ K_uy
    return res

def K_hat_xu(X: torch.Tensor, grid: torch.Tensor, a: float, gamma: float, lengthscale: float):
    Kuu = K(grid, grid, lengthscale)
    Kxu = K(X, grid, lengthscale)
    m = grid.shape[0]
    return Kxu @ torch.linalg.inv((a / m) * Kuu + gamma * torch.eye(m))

def objective(a_hat: torch.Tensor, gram_matrix: torch.Tensor, a: float, gamma: float):
    f = gram_matrix @ a_hat
    return -torch.sum(torch.log(a * (f ** 2))) + gamma * (a_hat.t() @ gram_matrix @ a_hat)

def optimize_a(gram_matrix: torch.Tensor, a: float, gamma: float):

    a_hat = torch.ones(gram_matrix.shape[0], dtype=torch.float, requires_grad=True)
    params = [a_hat]
    tol = 1e-9
    method = 'L-BFGS-B'

    optimizer = ScipyMinimizer(
        params=params,
        method=method,
        tol=tol,
    )

    def closure():
        optimizer.zero_grad()
        loss = objective(params[0], gram_matrix, a, gamma)
        loss.backward()
        return loss
    optimizer.step(closure)

    return a_hat.detach()

def train_model(u: torch.Tensor, grid: torch.Tensor, train_data: torch.Tensor, test_data: torch.Tensor, mu: float, gamma: float, lengthscale: float):

    gram_matrix = K_hat(train_data, train_data, u, mu, gamma, lengthscale)
    a_hat = optimize_a(gram_matrix, mu, gamma = 1.)

    K_grid = K_hat_xu(train_data, grid, mu, gamma, lengthscale)
    pred_intensity =  mu * (a_hat @  K_grid) ** 2

    K_hat_xtest_x = K_hat(test_data, train_data, u, mu, gamma, lengthscale)
    ppl = -(mu * (K_hat_xtest_x @ a_hat) ** 2).log().sum() + torch.mean(pred_intensity)

    return ppl

def cross_validation(X: torch.Tensor, u: torch.Tensor, grid: torch.Tensor, hyperparameters_grid: torch.Tensor):

    kfolds = 2
    kfold = KFold(n_splits=kfolds, shuffle=True)
    ppl_history = torch.zeros((hyperparameters_grid.shape[0], kfolds))

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X)):
        train_data = X[train_idx]
        test_data = X[test_idx]
        model_idx  = 0
        for hyperparams in hyperparameters_grid:
            lengthscale, gamma, mu = hyperparams
            ppl =  train_model(u, grid, train_data, test_data, mu, gamma, lengthscale)
            ppl_history[model_idx, fold] = ppl
            model_idx += 1
            print('mu =', mu, 'gamma =', gamma, 'lengthscale =', lengthscale, 'ppl =', ppl)

    return ppl_history

def cross_validation(data:list, u: torch.Tensor,hyperparameters_grid: torch.Tensor):
    
    #Code for reproducing the flaxman results in paper
    
    results =  torch.zeros((hyperparameters_grid.shape[0], 6))
    folds = torch.tensor([
        [2, 1],
        [3, 1],
        [1, 2],
        [3, 2],
        [1, 3],
        [2, 3]
    ])

    for i in range(6):
        train = data[folds[i, 0] - 1]
        test = data[folds[i, 1] - 1]
        for j in range(hyperparameters_grid.shape[0]):
            lengthscale, gamma, mu = hyperparameters_grid[j]
            res = train_model(u, train, test, mu, gamma, lengthscale)
            results[j, i] = res
            print('lengthscale =', lengthscale, 'mu =', mu, 'gamma =', gamma, 'ppl =', res)
    return results
"""

#General for LBBP

def get_multi_index(d: int, N: int):
    if d == 1:
        B = torch.arange(0, N, 1)
        return B.reshape(-1, 1)
    elif d == 2:
        b1 = torch.arange(0, N, 1)
        b2 = torch.arange(0, N, 1)
        B1, B2 = torch.meshgrid(b1, b2, indexing="ij")
        B = torch.vstack((B1.flatten(), B2.flatten())).T
        return B

def create_feature_matrix(X: torch.Tensor, N: int) -> torch.Tensor:
    d = len(X[0])
    B = get_multi_index(d, N)

    constant_term = (2 / torch.pi) ** (d / 2.)

    first_term = torch.where(B == 0, torch.sqrt(torch.tensor(0.5)), 1)
    second_term = torch.cos(B[:, None] * X)

    first_term = torch.prod(first_term, dim=-1)
    second_term = torch.prod(second_term, dim=-1)

    prod_term = first_term[:, None] * second_term

    return constant_term * prod_term

def get_eigen_matrix(d: int, N: int, a:float, b:float, m: int) ->  torch.Tensor:
    B = get_multi_index(d, N)
    l_b = 1. / (a * ((B ** 2).sum(dim=-1) ** m) + b)
    return torch.diag(l_b)

def objective(X: torch.Tensor, w: torch.Tensor, N: int):

    d = len(X[0])
    PHI_X = create_feature_matrix(X, N)
    L = get_eigen_matrix(d, N, a=1e-4, b=1e-4, m=2)
    M = torch.eye(N ** d) + torch.linalg.inv(L)

    log_h_term = (0.5 * (w @ PHI_X) ** 2).log().sum()
    quadratic_term = 0.5 * (w @ M @ w)

    return -log_h_term + quadratic_term

def laplace_approximation(X: torch.Tensor, N: int):

    d = len(X[0])

    w_hat = torch.ones(N ** d, dtype=torch.float, requires_grad=True)
    params = [w_hat]
    tol = 1e-9
    method = 'L-BFGS-B'

    optimizer = ScipyMinimizer(
        params=params,
        method=method,
        tol=tol,
    )

    def closure():
        optimizer.zero_grad()
        loss = objective(X, w_hat, N)
        loss.backward()
        return loss
    optimizer.step(closure)


    L = get_eigen_matrix(d, N, a=1e-4, b=1e-4, m=2)
    PHI_X = create_feature_matrix(X, N)

    L_inv = torch.linalg.inv(L)
    I = torch.eye(N ** d)

    a = 2 / (w_hat @ PHI_X)
    V = PHI_X * a
    D = 0.5 * torch.eye(X.shape[0])

    W = V @ D @  V.t()
    hess_inv = I + L_inv + W

    return w_hat.detach(), hess_inv
