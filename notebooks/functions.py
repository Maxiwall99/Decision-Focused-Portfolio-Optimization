# functions.py

import numpy as np
import cvxpy as cp
import torch
from torch.autograd import Function

def create_time_series_data_with_lags(return_matrix, max_lag=3):
    # Create training and test data for the prediction model.
    # Every entry of X are the return series of the last {max_lag} months.
    # The corresponding entry in Y is always the return series next closest afterwards.

    X = []
    Y = []

    max_lag = 3
    n_rows = len(return_matrix)

    n = max_lag
    while n < n_rows:
        # Get X
        X_row = (return_matrix[n - max_lag:n]).values.flatten()
        X.append(X_row)

        # Get Y
        Y_row = (return_matrix.iloc[n]).values.flatten()
        Y.append(Y_row)
        n = n + 1

    X = np.array(X)
    Y = np.array(Y)

    return X, Y

class CVaRSolver:

    def __init__(
        self,
        loss_matrix,
        N,
        S,
        alpha,
        beta
    ):

        self.loss_matrix = loss_matrix
        self.alpha = alpha
        self.S = S
        self.beta = beta

        # expected return parameter
        self.mu = cp.Parameter(N)

        # optimization variables
        self.w = cp.Variable(N)
        self.z = cp.Variable(S)
        self.h = cp.Variable()

        losses = self.loss_matrix @ self.w

        constraints = [

            # fully invested
            cp.sum(self.w) == 1,

            # long-only + box constraints
            self.w >= 0,
            self.w <= 0.2,

            # CVaR auxiliary constraints
            self.z >= 0,
            self.z >= losses - self.h,

            # CVaR risk limit
            self.h +
            (1 / ((1 - self.alpha) * self.S))
            * cp.sum(self.z)
            <= self.beta
        ]

        objective = cp.Maximize(
            self.mu @ self.w
        )

        self.problem = cp.Problem(
            objective,
            constraints
        )


    def solve(self, mu):

        self.mu.value = np.asarray(mu)

        self.problem.solve(
            solver=cp.HIGHS,
            warm_start=True
        )

        if self.problem.status not in (
            "optimal",
            "optimal_inaccurate"
        ):

            raise ValueError(
                f"Solver failed: {self.problem.status}"
            )

        return self.w.value.copy()


    def compute_cvar(
        self,
        portfolio_losses
    ):

        sorted_losses = np.sort(
            portfolio_losses
        )

        k = int(
            np.ceil(
                self.alpha *
                len(sorted_losses)
            )
        )

        return sorted_losses[k:].mean()


    def retrieve_stats(self):

        w = self.w.value.copy()

        VaR = float(
            self.h.value
        )

        portfolio_cvar = (
            self.compute_cvar(
                self.loss_matrix @ w
            )
            *100
        )

        portfolio_return = (
            self.mu.value @ w
            *100
        )

        return (
            portfolio_return,
            portfolio_cvar,
            VaR
        )
    
class SPOPlus(Function):

    @staticmethod
    def forward(ctx, mu_hat, mu_true, oracle_w, solver):
        """
        mu_hat:   predicted returns [N]
        mu_true:  true returns [N]
        oracle_w: precomputed w*(mu_true) [N]
        solver:   CVaR solver (only used for perturbed solve now)
        """

        # oracle_w is precomputed

        # perturbed solve
        mu_hat_np = mu_hat.detach().cpu().numpy()
        mu_true_np = mu_true.detach().cpu().numpy()
        perturbed_mu = 2 * mu_hat_np - mu_true_np
        perturbed_w = solver.solve(perturbed_mu)
        perturbed_w = torch.tensor(
            perturbed_w,
            dtype=mu_hat.dtype,
            device=mu_hat.device
        )

        ctx.save_for_backward(oracle_w, perturbed_w)

        loss = (
            -torch.dot(perturbed_w, mu_true)
            + 2 * torch.dot(mu_hat, oracle_w)
            - torch.dot(oracle_w, mu_true)
        )

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        oracle_w, perturbed_w = ctx.saved_tensors
        grad_mu_hat = 2 * (oracle_w - perturbed_w)

        return (
            grad_output * grad_mu_hat,
            None,   # mu_true
            None,   # oracle_w
            None,   # solver
        )
    

