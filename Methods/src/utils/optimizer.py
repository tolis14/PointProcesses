import torch
from torchmin import ScipyMinimizer

class MinimizerWrapper:

    def __init__(self, objective: callable,
                 method: str='L-BFGS-B', tol: float=1e-9, max_iter: int=1000):

        self.objective = objective
        self.method = method
        self.tol = tol
        self.max_iter = max_iter


    def optimize(self, initial_guess: torch.Tensor,*args):
        initial_guess.requires_grad = True
        params = [initial_guess]

        optimizer = ScipyMinimizer(
            params=params,
            method=self.method,
            tol=self.tol,
            options={'maxiter': self.max_iter}
        )

        def closure():
            optimizer.zero_grad()
            loss = self.objective(params[0], *args)
            loss.backward()
            return loss
        optimizer.step(closure)

        return initial_guess.detach()

