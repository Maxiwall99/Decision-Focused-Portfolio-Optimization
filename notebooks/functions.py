# functions.py

import numpy as np
import cvxpy as cp
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Function
import copy

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

        # expected loss parameter
        self.c = cp.Parameter(N)

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

        objective = cp.Minimize(
            self.c @ self.w
        )

        self.problem = cp.Problem(
            objective,
            constraints
        )


    def solve(self, c):

        self.c.value = np.asarray(c)

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

        portfolio_return = - (
            self.c.value @ w
            *100
        )

        return (
            portfolio_return,
            portfolio_cvar,
            VaR
        )
    
class SPOPlus(Function):

    @staticmethod
    def forward(ctx, c_hat, c_true, oracle_w, solver):
        """
        c_hat:   predicted costs (negative returns) [N]
        c_true:  true costs [N]
        oracle_w: precomputed w*(- mu_true) [N]
        solver:   CVaR solver (only used for perturbed solve now)
        """

        # oracle_w is precomputed

        # perturbed solve
        c_hat_np = c_hat.detach().cpu().numpy()
        c_true_np = c_true.detach().cpu().numpy()
        # perturbed_c = c_true_np + 15 * (c_hat_np - c_true_np)       # !!!!!! increased perturubation strength
        perturbed_c = 2 * c_hat_np - c_true_np 
        perturbed_w = solver.solve(perturbed_c)
        perturbed_w = torch.tensor(
            perturbed_w,
            dtype=c_hat.dtype,
            device=c_hat.device
        )

        ctx.save_for_backward(oracle_w, perturbed_w)

        # loss = (
        #     torch.dot(c_true, perturbed_w)
        #     - torch.dot(c_true, oracle_w)
        # )
        loss = (
            2 * torch.dot(c_hat, oracle_w)
            - torch.dot(c_true, oracle_w)
            - torch.dot(2 * c_hat - c_true, perturbed_w)
        )
                                    
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        oracle_w, perturbed_w = ctx.saved_tensors
        grad_mu_hat = 2 * (oracle_w - perturbed_w)  

        return (
            grad_output * grad_mu_hat,
            None,   # c_true
            None,   # oracle_w
            None,   # solver
        )
    
class VARasNN(nn.Module):

    def __init__(self,input_dim,output_dim):

        super().__init__()

        self.linear = nn.Linear(
            input_dim,
            output_dim
        )

    def forward(self,x):

        return self.linear(x)
    

def compute_loss_normalization(model, train_loader, solver, criterion):
    '''
    Compute the normalization factors of the two losses SPO+ and MSE to make them comparable for training with a combined loss.
    Take the average loss over all losses computed from the batches of the whole training set.
    '''
    model.train()
    spo_vals = []
    mse_vals = []

    with torch.no_grad():  # no updates in this pass, just loss computation
        for X_batch, Y_batch, oracle_batch in tqdm(train_loader, desc="Computing loss scales for normalization"):
            predictions = model(X_batch)

            spo_losses = []
            for i in range(X_batch.shape[0]):
                loss_i = SPOPlus.apply(
                    - predictions[i],           # c_hat
                    - Y_batch[i],               # c_true
                    oracle_batch[i],
                    solver
                )
                spo_losses.append(loss_i.item())

            spo_vals.append(np.mean(spo_losses))
            mse_vals.append(criterion(predictions, Y_batch).item())

    spo_scale = np.mean(spo_vals)
    mse_scale = np.mean(mse_vals)

    return spo_scale, mse_scale

def create_train_test_split(X, Y, test_index, val_length=3):
    # Create train/test split and scenario loss matrix for CVaR computation.

    X_train = X[:-(test_index + val_length)] # take all rows up to the test_index last row minus the val_length for training
    X_val = X[-(test_index + val_length): -test_index] # take rows inbetween for validation
    X_test = X[-test_index] # take the test_index last row for testing

    Y_train = Y[:-(test_index + val_length)]
    Y_val = Y[-(test_index + val_length): -test_index]
    Y_test = Y[-test_index]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def get_scenario_loss_matrix(return_matrix, index, num_scenarios=1000, random_seed=42):
    # Need return matrix because otherwise the first max_lag entries are excluded.
    # Perform bootstrap for return scenarios on the training period (excluding validation and test period)
    return_array = return_matrix[:-index].values

    np.random.seed(random_seed)
    # Sample row indices with replacement
    sample_indices = np.random.choice(return_array.shape[0], size=num_scenarios, replace=True)
    # Generate bootstrapped scenario loss matrix
    scenario_loss_matrix = -return_array[sample_indices, :]

    return scenario_loss_matrix

def train_model(model, n_epochs, train_loader, optimizer, criterion, solver, spo_scale, mse_scale, gamma, val_X, val_Y, oracle_val_tensor, early_stopping_patience):

    history = {
        "combined_loss": [],
        "spo_loss": [],
        "mse_loss": [],
        "val_regret": [],
        "early_stopping_epoch": int,
        "best_epoch": int,
    }

    best_val_regret = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in tqdm(range(n_epochs), desc="epochs"):

        model.train()

        epoch_spo_loss = 0.0
        epoch_mse_loss = 0.0
        epoch_combined_loss = 0.0

        for X_batch, Y_batch, oracle_batch in train_loader:

            optimizer.zero_grad()

            predictions = model(X_batch)

            spo_losses = []
            for i in range(X_batch.shape[0]):

                loss_i = SPOPlus.apply(
                    - predictions[i],           # c_hat
                    - Y_batch[i],               # c_true
                    oracle_batch[i],
                    solver
                )

                spo_losses.append(loss_i)

            spo_loss = torch.stack(spo_losses).mean()
            spo_loss_scaled = spo_loss / spo_scale

            mse_loss = criterion(predictions, Y_batch)
            mse_loss_scaled = mse_loss / mse_scale

            loss = (
                gamma * (spo_loss_scaled)
                + (1 - gamma) * (mse_loss_scaled)
            )

            loss.backward()

            # Log gradients every first batch of each epoch
            # if len(history["combined_loss"]) == 0 or True:  # or gate by batch index
            #     for name, param in model.named_parameters():
            #         if param.grad is not None:
            #             print(f"  {name}: grad_norm={param.grad.norm().item():.6f}, grad_mean={param.grad.mean().item():.6f}")

            # gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_spo_loss += spo_loss_scaled.item()
            epoch_mse_loss += mse_loss_scaled.item()
            epoch_combined_loss += loss.item()

        n_batches = len(train_loader)

        avg_spo = epoch_spo_loss / n_batches
        avg_mse = epoch_mse_loss / n_batches
        avg_combined = epoch_combined_loss / n_batches

        # Compute the validation regret for early stopping─
        model.eval()
        total_regret = 0.0

        with torch.no_grad():
            for i in range(len(val_X)):
                y_hat = model(val_X[i]).numpy()
                y_true = val_Y[i].numpy()

                w_hat  = solver.solve(-y_hat)
                w_star = oracle_val_tensor[i].numpy()

                total_regret += float(y_true @ w_star - y_true @ w_hat)

        avg_val_regret = total_regret / len(val_X)

        if avg_val_regret < best_val_regret:
            best_val_regret = avg_val_regret
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        history["spo_loss"].append(avg_spo)
        history["mse_loss"].append(avg_mse)
        history["combined_loss"].append(avg_combined)
        history["val_regret"].append(avg_val_regret)

        print(
            f"Epoch {epoch+1}: "
            f"combined={avg_combined:.6f} | "
            f"SPO+ scaled={avg_spo:.6f} | "
            f"MSE scaled={avg_mse:.6f} | "
            f"val_regret={avg_val_regret:.6f}"
            f"{'  ✓ regret improved' if epochs_without_improvement == 0 else ''}"
        )

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}. Best val regret: {best_val_regret:.6f}")

            history["early_stopping_epoch"] = epoch+1
            history["best_epoch"] = epoch+1 - early_stopping_patience

            break

    # restore weights from best epoch
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return history
