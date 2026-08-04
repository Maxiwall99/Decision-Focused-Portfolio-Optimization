# functions.py

import pandas as pd
import numpy as np
import cvxpy as cp
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import Function
import copy


def create_time_series_data_with_lags(return_matrix, max_lag):
    """
    Builds lagged feature matrix (X) and corresponding targets (Y) for an
    autoregressive prediction model from monthly return data.

    For each time point t (with t >= max_lag), X[i] contains the flattened
    returns of the {max_lag} preceding months (t-max_lag to t-1), and Y[i]
    contains the return of the following month t (all assets).

    Args:
        return_matrix (pd.DataFrame): Return matrix with rows = months,
            columns = assets.
        max_lag (int): Number of lagged months used as features.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            X with shape (n_rows - max_lag, max_lag * n_assets),
            Y with shape (n_rows - max_lag, n_assets).
    """

    X = []
    Y = []

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
    """
    Solves the CVaR-constrained portfolio optimization problem in its
    linear-programming form (Rockafellar-Uryasev formulation).

    Minimizes expected loss w.r.t. portfolio weights w, subject to the
    CVaR at confidence level alpha staying below the risk budget beta,
    plus full-investment, long-only, and box constraints on w.

    Args:
        loss_matrix (np.ndarray): Scenario loss matrix, shape (S, N)
            — S scenarios (rows), N assets (columns).
        N (int): Number of assets.
        S (int): Number of scenarios (used to average the CVaR tail).
        alpha (float): CVaR confidence level (e.g. 0.95).
        beta (float): CVaR risk limit (risk-aversion threshold).
    """

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
        """
        Solves the CVaR-LP for a given expected-loss vector c and returns
        the optimal portfolio weights.

        Args:
            c (array-like): Expected loss (or negative expected return)
                per asset, shape (N,).

        Returns:
            np.ndarray: Optimal portfolio weights w, shape (N,).

        Raises:
            ValueError: If the solver does not reach an optimal solution.
        """

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


    def compute_cvar(self, portfolio_losses):
        """
        Computes the empirical CVaR at level alpha from a sample of
        portfolio losses, as the mean of the losses in the worst
        (1 - alpha) tail.

        Args:
            portfolio_losses (array-like): Realized portfolio losses
                across scenarios, shape (S,).

        Returns:
            float: Empirical CVaR of the given loss sample.
        """

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
        """
        Computes summary statistics for the currently solved portfolio:
        realized return, empirical CVaR (via compute_cvar), and the
        LP's VaR variable h. Return and CVaR are expressed in percent.

        Returns:
            tuple[float, float, float]:
                (portfolio_return, portfolio_cvar, VaR)
        """

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
    """
    Custom autograd Function implementing the SPO+ (Smart Predict-then-
    Optimize) surrogate loss for decision-focused learning with a
    CVaR-constrained portfolio oracle.

    Provides a differentiable surrogate for the true (non-differentiable)
    decision loss, using a perturbed re-solve of the CVaR-LP in the
    forward pass and the closed-form SPO+ subgradient in the backward
    pass.
    """

    @staticmethod
    def forward(ctx, c_hat, c_true, oracle_w, solver):
        """
        Computes the SPO+ surrogate loss for one sample.

        Solves the CVaR-LP once at the perturbed cost vector
        `2*c_hat - c_true` to get `perturbed_w`, then evaluates the SPO+
        loss against the precomputed oracle solution `oracle_w`
        (obtained by solving at the true cost `-c_true`). Saves
        `oracle_w` and `perturbed_w` for use in `backward`.

        Args:
            c_hat (torch.Tensor): Predicted costs (negative returns), shape (N,).
            c_true (torch.Tensor): True costs (negative returns), shape (N,).
            oracle_w (torch.Tensor): Precomputed optimal weights at c_true, shape (N,).
            solver: CVaR solver instance used to re-solve at the perturbed costs.

        Returns:
            torch.Tensor: Scalar SPO+ surrogate loss.
        """

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

        loss = (
            2 * torch.dot(c_hat, oracle_w)
            - torch.dot(c_true, oracle_w)
            - torch.dot(2 * c_hat - c_true, perturbed_w)
        )
                                    
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        """
        Computes the SPO+ subgradient w.r.t. c_hat.

        Uses the saved oracle and perturbed solutions from the forward
        pass: grad = 2 * (oracle_w - perturbed_w), scaled by grad_output.
        Gradients for c_true, oracle_w, and solver are not defined (None).

        Args:
            grad_output (torch.Tensor): Upstream gradient (scalar).

        Returns:
            tuple: (grad_c_hat, None, None, None)
        """
        oracle_w, perturbed_w = ctx.saved_tensors
        grad_mu_hat = 2 * (oracle_w - perturbed_w)  

        return (
            grad_output * grad_mu_hat,
            None,   # c_true
            None,   # oracle_w
            None,   # solver
        )


class VARasNN(nn.Module):
    """
    Vector autoregression (VAR) model implemented as a single linear
    layer in PyTorch, with a learnable positive output scaling factor.

    The scaling factor is parameterized as exp(log_scale) to ensure it
    stays positive during unconstrained optimization, while allowing it
    to shrink or grow the linear prediction independently of the
    layer's own weights (this is useful for calibrating output magnitude
    without disturbing the learned VAR coefficients).

    Args:
        input_dim (int): Number of input features (max_lag * n_assets).
        output_dim (int): Number of output targets (n_assets).
        init_scale (float): Initial value of the scaling factor
            (default 0.1); stored internally as its log.
    """

    def __init__(self, input_dim, output_dim, init_scale=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.log_scale = nn.Parameter(torch.tensor(np.log(init_scale), dtype=torch.float32))

    def forward(self, x):
        """
        Computes the scaled VAR prediction for input x.

        Args:
            x (torch.Tensor): Input features, shape (batch, input_dim).

        Returns:
            torch.Tensor: Scaled linear prediction, shape (batch, output_dim).
        """
        return self.linear(x) * torch.exp(self.log_scale)
    

def compute_loss_normalization(model, train_loader, solver, criterion):
    """
    Computes normalization scale factors for the SPO+ and MSE losses so
    they can be combined into a single weighted training objective on a
    comparable scale.

    Runs one no-grad forward pass over the entire training set, computing
    per-batch average SPO+ loss (via SPOPlus.apply on predicted/true
    costs, i.e. negative returns) and per-batch MSE loss, then averages
    each across all batches to get one scale factor per loss type.

    Args:
        model (nn.Module): Prediction model (e.g. VARasNN) in eval mode
            for loss computation (though model.train() is called, no
            gradients are computed due to the no_grad context).
        train_loader (DataLoader): Yields (X_batch, Y_batch, oracle_batch)
            triples for the full training set.
        solver: CVaR solver instance used inside SPOPlus.apply for the
            perturbed re-solve.
        criterion: Loss function for MSE (e.g. nn.MSELoss).

    Returns:
        tuple[float, float]: (spo_scale, mse_scale) — mean SPO+ loss and
            mean MSE loss over the training set, used later to normalize
            each loss term before combining them.
    """

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

def create_train_test_split(X, Y, test_index, val_length=6):
    """
    Splits feature/target arrays into training, validation, and a single
    test sample using a rolling-origin scheme, indexed backward from the
    end of the series.

    Given `test_index` (position counted from the end) and `val_length`
    (number of validation months), the split is:
    - Train: all rows before the validation block, i.e. rows
      [0, len - (test_index + val_length))
    - Validation: the `val_length` rows immediately before the test row,
      i.e. rows [len - (test_index + val_length), len - test_index)
    - Test: the single row at position (len - test_index)

    Args:
        X (np.ndarray): Feature matrix, shape (n_samples, n_features).
        Y (np.ndarray): Target matrix, shape (n_samples, n_targets).
        test_index (int): Position of the test sample, counted from the
            end of the array (e.g. test_index=1 selects the last row).
        val_length (int): Number of rows immediately preceding the test
            sample used for validation (default 3).

    Returns:
        tuple: (X_train, X_val, X_test, Y_train, Y_val, Y_test)
            X_test and Y_test are single rows.
    """

    X_train = X[:-(test_index + val_length)] # take all rows up to the test_index last row minus the val_length for training
    X_val = X[-(test_index + val_length): -test_index] # take rows inbetween for validation
    X_test = X[-test_index] # take the test_index last row for testing

    Y_train = Y[:-(test_index + val_length)]
    Y_val = Y[-(test_index + val_length): -test_index]
    Y_test = Y[-test_index]

    return X_train, X_val, X_test, Y_train, Y_val, Y_test


def get_scenario_loss_matrix(return_matrix, index, num_scenarios=1000, random_seed=42):
    """
    Generates a bootstrapped scenario loss matrix for CVaR estimation by
    resampling historical monthly returns from the training period.

    Uses `return_matrix` directly (rather than the lagged X/Y arrays) so
    that the first `max_lag` months, excluded from X and Y, are still
    available as scenarios. Restricts sampling to the training period by
    excluding the last `index` rows (validation + test period), then
    draws `num_scenarios` row indices with replacement and negates the
    sampled returns to obtain losses.

    Args:
        return_matrix (pd.DataFrame): Full return matrix with rows =
            months, columns = assets.
        index (int): Number of trailing rows (validation + test period)
            to exclude from the sampling pool.
        num_scenarios (int): Number of bootstrap scenarios to draw
            (default 1000).
        random_seed (int): Seed for reproducible sampling.

    Returns:
        np.ndarray: Bootstrapped scenario loss matrix, shape
            (num_scenarios, n_assets), used as `loss_matrix` in
            CVaRSolver.
    """

    return_array = return_matrix[:-index].values

    np.random.seed(random_seed)
    # Sample row indices with replacement
    sample_indices = np.random.choice(return_array.shape[0], size=num_scenarios, replace=True)
    # Generate bootstrapped scenario loss matrix
    scenario_loss_matrix = -return_array[sample_indices, :]

    return scenario_loss_matrix

def train_model(model, n_epochs, train_loader, optimizer, criterion, solver,
                spo_scale, mse_scale, gamma, val_X, val_Y, oracle_val_tensor,
                early_stopping_patience):
    """
    Trains a prediction model with a combined SPO+ / MSE decision-focused
    loss, using validation regret for early stopping and best-epoch
    model selection.

    For each epoch, iterates over training batches computing a per-sample
    SPO+ loss (via SPOPlus.apply, using the CVaR solver's oracle) and an
    MSE loss, each normalized by its precomputed scale factor
    (spo_scale, mse_scale) and combined as
    `gamma * spo_loss_scaled + (1 - gamma) * mse_loss_scaled`. Gradients
    are clipped to max_norm=1.0 before the optimizer step. After each
    epoch, computes average validation regret (true return of the oracle
    portfolio minus true return of the portfolio implied by the model's
    predictions, re-solved via the CVaR solver) over `val_X`/`val_Y`, and
    tracks the best model state by lowest validation regret. Training
    stops early if regret fails to improve for `early_stopping_patience`
    consecutive epochs; the best model's weights are restored before
    returning.

    Args:
        model (nn.Module): Prediction model to train (e.g. VARasNN).
        n_epochs (int): Maximum number of training epochs.
        train_loader (DataLoader): Yields (X_batch, Y_batch, oracle_batch).
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion: MSE loss function.
        solver: CVaR solver instance used for SPO+ and validation regret.
        spo_scale (float): Normalization factor for the SPO+ loss.
        mse_scale (float): Normalization factor for the MSE loss.
        gamma (float): Weight on the SPO+ term in the combined loss
            (0 = pure MSE, 1 = pure SPO+).
        val_X (torch.Tensor): Validation features.
        val_Y (torch.Tensor): Validation targets (true returns).
        oracle_val_tensor (torch.Tensor): Precomputed oracle portfolio
            weights for each validation sample.
        early_stopping_patience (int): Number of epochs without
            improvement in validation regret before stopping.

    Returns:
        dict: History dict with keys "combined_loss", "spo_loss",
            "mse_loss", "val_regret" (lists, one entry per epoch), and
            "early_stopping_epoch", "best_epoch" (int or None).
    """

    history = {
        "combined_loss": [],
        "spo_loss": [],
        "mse_loss": [],
        "val_regret": [],
        "early_stopping_epoch": None,
        "best_epoch": None,
    }

    best_val_regret = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_spo_loss += spo_loss_scaled.item()
            epoch_mse_loss += mse_loss_scaled.item()
            epoch_combined_loss += loss.item()

        n_batches = len(train_loader)

        avg_spo = epoch_spo_loss / n_batches
        avg_mse = epoch_mse_loss / n_batches
        avg_combined = epoch_combined_loss / n_batches

        # Compute the validation regret for early stopping
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


def train_model_diagnosis(model, n_epochs, train_loader, optimizer, criterion,
                           solver, spo_scale, mse_scale, gamma,
                           val_X, val_Y, oracle_val_tensor,
                           early_stopping_patience, tol=1e-4):
    """
    Diagnostic variant of train_model: runs the identical SPO+/MSE
    combined-loss training loop with early stopping, but additionally
    logs per-sample gradient diagnostics at every training step.

    For each training sample, in addition to the normal SPO+.apply call
    used for backpropagation, separately re-solves the CVaR-LP at the perturbed
    cost to get `perturbed_w`, and checks whether it coincides with
    `oracle_w`, i.e. whether the perturbation actually
    moved the LP to a different vertex. Also computes, per sample: the
    raw SPO+ gradient (`2*(oracle_w - perturbed_w)`, matching
    SPOPlus.backward), the raw per-sample MSE gradient (normalized by
    batch_size * n_assets to match nn.MSELoss(reduction='mean')), both
    gradient norms before scaling, after dividing by spo_scale and mse_scale,
    and after gamma-weighting (their actual contribution to the combined
    gradient), plus the cosine similarity between the SPO+ and MSE
    gradient directions (NaN if the SPO+ gradient is exactly zero, i.e.
    oracle_w == perturbed_w).

    Args:
        model, n_epochs, train_loader, optimizer, criterion, solver,
        spo_scale, mse_scale, gamma, val_X, val_Y, oracle_val_tensor,
        early_stopping_patience: Same as train_model.
        tol (float): Absolute tolerance for np.allclose when checking
            oracle_w vs. perturbed_w vertex equality (default 1e-4).

    Returns:
        tuple[dict, pd.DataFrame]:
            history: Same structure as train_model's return value.
            diag_df: One row per (epoch, batch, sample) with columns
                epoch, batch, sample, oracle_w, perturbed_w, vertex_match,
                spo_grad_norm_raw, mse_grad_norm_raw, spo_grad_norm_normed,
                mse_grad_norm_normed, spo_grad_norm_weighted,
                mse_grad_norm_weighted, cosine_similarity.
    """

    history = {
        "combined_loss": [], "spo_loss": [], "mse_loss": [],
        "val_regret": [], "early_stopping_epoch": None, "best_epoch": None,
    }
    diag_log = []  # one row per (epoch, batch, sample)

    best_val_regret = float("inf")
    best_model_state = None
    epochs_without_improvement = 0

    for epoch in range(n_epochs):
        model.train()
        epoch_spo_loss = epoch_mse_loss = epoch_combined_loss = 0.0

        for batch_idx, (X_batch, Y_batch, oracle_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            predictions = model(X_batch)

            batch_size = X_batch.shape[0]
            n_assets = Y_batch.shape[1]

            spo_losses = []
            for i in range(batch_size):
                pred_i = predictions[i]
                y_i = Y_batch[i]
                oracle_w_i = oracle_batch[i].numpy()

                loss_i = SPOPlus.apply(-pred_i, -y_i, oracle_batch[i], solver)
                spo_losses.append(loss_i)

                # diagnostics
                pred_np = pred_i.detach().numpy()
                y_np = y_i.numpy()

                c_hat = -pred_np
                c_true = -y_np
                perturbed_c = 2 * c_hat - c_true
                perturbed_w_i = solver.solve(perturbed_c)

                vertex_match = bool(np.allclose(oracle_w_i, perturbed_w_i, atol=tol))

                # raw per-sample gradients w.r.t. predictions
                spo_grad_raw = 2 * (oracle_w_i - perturbed_w_i)          # SPOPlus.backward formula

                # MSELoss(reduction='mean') averages over all elements in the batch
                # (batch_size * n_assets), so the correct per-sample gradient contribution
                # w.r.t. this sample's prediction carries same normalization factor.
                mse_grad_raw = 2 * (pred_np - y_np) / (batch_size * n_assets)

                spo_norm_raw = float(np.linalg.norm(spo_grad_raw))
                mse_norm_raw = float(np.linalg.norm(mse_grad_raw))

                # after normalization (divide by the scale constants used in training)
                spo_grad_normed = spo_grad_raw / spo_scale
                mse_grad_normed = mse_grad_raw / mse_scale
                spo_norm_normed = float(np.linalg.norm(spo_grad_normed))
                mse_norm_normed = float(np.linalg.norm(mse_grad_normed))

                # after gamma weighting (actual contribution to the combined gradient)
                spo_norm_weighted = float(gamma * spo_norm_normed)
                mse_norm_weighted = float((1 - gamma) * mse_norm_normed)

                if spo_norm_raw > 1e-10 and mse_norm_raw > 1e-10:
                    cos_sim = float(
                        np.dot(spo_grad_raw, mse_grad_raw) / (spo_norm_raw * mse_norm_raw)
                    )
                else:
                    cos_sim = np.nan  # SPO+ gradient exactly zero implies that direction undefined

                diag_log.append({
                    "epoch": epoch, "batch": batch_idx, "sample": i,
                    "oracle_w": oracle_w_i, "perturbed_w": perturbed_w_i,
                    "vertex_match": vertex_match,
                    "spo_grad_norm_raw": spo_norm_raw, "mse_grad_norm_raw": mse_norm_raw,
                    "spo_grad_norm_normed": spo_norm_normed, "mse_grad_norm_normed": mse_norm_normed,
                    "spo_grad_norm_weighted": spo_norm_weighted, "mse_grad_norm_weighted": mse_norm_weighted,
                    "cosine_similarity": cos_sim,
                })

            spo_loss = torch.stack(spo_losses).mean()
            spo_loss_scaled = spo_loss / spo_scale
            mse_loss = criterion(predictions, Y_batch)
            mse_loss_scaled = mse_loss / mse_scale

            loss = gamma * spo_loss_scaled + (1 - gamma) * mse_loss_scaled
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_spo_loss += spo_loss_scaled.item()
            epoch_mse_loss += mse_loss_scaled.item()
            epoch_combined_loss += loss.item()

        n_batches = len(train_loader)
        history["spo_loss"].append(epoch_spo_loss / n_batches)
        history["mse_loss"].append(epoch_mse_loss / n_batches)
        history["combined_loss"].append(epoch_combined_loss / n_batches)

        model.eval()
        total_regret = 0.0
        with torch.no_grad():
            for i in range(len(val_X)):
                y_hat = model(val_X[i]).numpy()
                y_true = val_Y[i].numpy()
                w_hat = solver.solve(-y_hat)
                w_star = oracle_val_tensor[i].numpy()
                total_regret += float(y_true @ w_star - y_true @ w_hat)
        avg_val_regret = total_regret / len(val_X)
        history["val_regret"].append(avg_val_regret)

        if avg_val_regret < best_val_regret:
            best_val_regret = avg_val_regret
            best_model_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"Epoch {epoch+1}: combined={history['combined_loss'][-1]:.6f} | "
              f"val_regret={avg_val_regret:.6f}"
              f"{'  ✓ regret improved' if epochs_without_improvement == 0 else ''}")

        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch+1}. Best val regret: {best_val_regret:.6f}")
            history["early_stopping_epoch"] = epoch + 1
            history["best_epoch"] = epoch + 1 - early_stopping_patience
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    diag_df = pd.DataFrame(diag_log)
    return history, diag_df