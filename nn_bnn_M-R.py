"""
Neural-network and Bayesian-neural-network analysis of the exoplanet M-R relation.

The data set has only 760 cleaned planets and one predictor, log10(M/M_earth).
For that reason the default architecture is deliberately small and smooth:
two hidden layers with 24 units. The deterministic neural network learns a
mean relation plus one global intrinsic-scatter term. The Bayesian neural
network uses the same architecture, but each linear-layer weight has a
diagonal Gaussian variational posterior and is trained with an ELBO objective.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import copy
import importlib.util
from pathlib import Path
import random
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"


def load_mass_radius_module():
    """Load helpers from emcee_M-R.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("nn_bnn_emcee_mr_helpers", EMCEE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["nn_bnn_emcee_mr_helpers"] = module
    spec.loader.exec_module(module)
    return module


MR = load_mass_radius_module()


@dataclass
class Standardizer:
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softplus_inverse(value: float) -> float:
    return float(np.log(np.expm1(value)))


def make_shared_splits(
    raw_df: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create one train/validation/test split shared by the NN and BNN."""
    df = MR.prepare_fit_data(raw_df).copy()
    df["split_row_id"] = np.arange(len(df))
    indices = np.arange(len(df))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=validation_size,
        random_state=seed + 1,
        shuffle=True,
    )

    train_df = df.iloc[train_idx].sort_values("mass").reset_index(drop=True)
    val_df = df.iloc[val_idx].sort_values("mass").reset_index(drop=True)
    test_df = df.iloc[test_idx].sort_values("mass").reset_index(drop=True)
    return train_df, val_df, test_df


def fit_standardizer(train_df: pd.DataFrame) -> Standardizer:
    x = train_df["log_mass"].to_numpy()
    y = train_df["log_radius"].to_numpy()
    return Standardizer(
        x_mean=float(np.mean(x)),
        x_std=float(np.std(x, ddof=0)),
        y_mean=float(np.mean(y)),
        y_std=float(np.std(y, ddof=0)),
    )


def tensor_data(df: pd.DataFrame, scaler: Standardizer, device: torch.device) -> dict[str, torch.Tensor]:
    x = ((df["log_mass"].to_numpy() - scaler.x_mean) / scaler.x_std).reshape(-1, 1)
    y = ((df["log_radius"].to_numpy() - scaler.y_mean) / scaler.y_std).reshape(-1, 1)
    yerr = (df["log_radius_err"].to_numpy() / scaler.y_std).reshape(-1, 1)
    return {
        "x": torch.as_tensor(x, dtype=torch.float32, device=device),
        "y": torch.as_tensor(y, dtype=torch.float32, device=device),
        "yerr": torch.as_tensor(yerr, dtype=torch.float32, device=device),
    }


def x_tensor(x: np.ndarray, scaler: Standardizer, device: torch.device) -> torch.Tensor:
    x_scaled = ((np.asarray(x) - scaler.x_mean) / scaler.x_std).reshape(-1, 1)
    return torch.as_tensor(x_scaled, dtype=torch.float32, device=device)


def activation_fn(name: str, value: torch.Tensor) -> torch.Tensor:
    if name == "tanh":
        return torch.tanh(value)
    if name == "silu":
        return F.silu(value)
    if name == "relu":
        return F.relu(value)
    raise ValueError(f"Unknown activation: {name}")


def gaussian_nll(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    yerr: torch.Tensor,
    sigma_int: torch.Tensor,
) -> torch.Tensor:
    variance = yerr**2 + sigma_int**2 + 1e-8
    return 0.5 * ((y_true - y_pred) ** 2 / variance + torch.log(2.0 * torch.pi * variance))


class DeterministicMLP(nn.Module):
    def __init__(
        self,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        init_sigma_scaled: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 1
        for _ in range(hidden_layers):
            layers.append(nn.Linear(in_features, hidden_width))
            in_features = hidden_width
        self.layers = nn.ModuleList(layers)
        self.output = nn.Linear(in_features, 1)
        self.activation = activation
        self.raw_sigma_int = nn.Parameter(
            torch.tensor(softplus_inverse(max(init_sigma_scaled, 1e-4)), dtype=torch.float32)
        )

    def sigma_int(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma_int) + 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = activation_fn(self.activation, layer(h))
        return self.output(h)


class BayesianLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_sigma: float,
        rho_init: float,
    ) -> None:
        super().__init__()
        self.prior_sigma = float(prior_sigma)
        scale = np.sqrt(2.0 / (in_features + out_features))
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).normal_(0.0, scale))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), rho_init))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), rho_init))

    @staticmethod
    def _sigma(rho: torch.Tensor) -> torch.Tensor:
        return F.softplus(rho) + 1e-6

    def forward(self, x: torch.Tensor, sample: bool) -> torch.Tensor:
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)
        if sample:
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self) -> torch.Tensor:
        prior_var = self.prior_sigma**2
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)
        weight_kl = (
            torch.log(self.prior_sigma / weight_sigma)
            + (weight_sigma**2 + self.weight_mu**2) / (2.0 * prior_var)
            - 0.5
        ).sum()
        bias_kl = (
            torch.log(self.prior_sigma / bias_sigma)
            + (bias_sigma**2 + self.bias_mu**2) / (2.0 * prior_var)
            - 0.5
        ).sum()
        return weight_kl + bias_kl


class BayesianMLP(nn.Module):
    def __init__(
        self,
        hidden_width: int,
        hidden_layers: int,
        activation: str,
        init_sigma_scaled: float,
        prior_sigma: float,
        rho_init: float,
    ) -> None:
        super().__init__()
        layers: list[BayesianLinear] = []
        in_features = 1
        for _ in range(hidden_layers):
            layers.append(BayesianLinear(in_features, hidden_width, prior_sigma, rho_init))
            in_features = hidden_width
        self.layers = nn.ModuleList(layers)
        self.output = BayesianLinear(in_features, 1, prior_sigma, rho_init)
        self.activation = activation
        self.raw_sigma_int = nn.Parameter(
            torch.tensor(softplus_inverse(max(init_sigma_scaled, 1e-4)), dtype=torch.float32)
        )

    def sigma_int(self) -> torch.Tensor:
        return F.softplus(self.raw_sigma_int) + 1e-5

    def forward(self, x: torch.Tensor, sample: bool = True) -> torch.Tensor:
        h = x
        for layer in self.layers:
            h = activation_fn(self.activation, layer(h, sample=sample))
        return self.output(h, sample=sample)

    def kl_divergence(self) -> torch.Tensor:
        total = self.output.kl_divergence()
        for layer in self.layers:
            total = total + layer.kl_divergence()
        return total


def train_deterministic_nn(
    train_data: dict[str, torch.Tensor],
    val_data: dict[str, torch.Tensor],
    scaler: Standardizer,
    hidden_width: int,
    hidden_layers: int,
    activation: str,
    init_scatter_dex: float,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
) -> tuple[DeterministicMLP, pd.DataFrame]:
    model = DeterministicMLP(
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        activation=activation,
        init_sigma_scaled=init_scatter_dex / scaler.y_std,
    ).to(train_data["x"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    stale = 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_pred = model(train_data["x"])
        train_nll = gaussian_nll(
            train_pred,
            train_data["y"],
            train_data["yerr"],
            model.sigma_int(),
        ).mean()
        train_nll.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_data["x"])
            val_nll = gaussian_nll(
                val_pred,
                val_data["y"],
                val_data["yerr"],
                model.sigma_int(),
            ).mean()
            val_value = float(val_nll.cpu())
            rows.append(
                {
                    "method": "Neural Network",
                    "epoch": epoch,
                    "train_loss": float(train_nll.detach().cpu()),
                    "val_loss": val_value,
                    "sigma_int_dex": float(model.sigma_int().detach().cpu()) * scaler.y_std,
                    "kl_weight": 0.0,
                }
            )

        if val_value < best_val - 1e-5:
            best_val = val_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def train_bayesian_nn(
    train_data: dict[str, torch.Tensor],
    val_data: dict[str, torch.Tensor],
    scaler: Standardizer,
    hidden_width: int,
    hidden_layers: int,
    activation: str,
    init_scatter_dex: float,
    prior_sigma: float,
    rho_init: float,
    kl_weight: float,
    kl_warmup_epochs: int,
    learning_rate: float,
    weight_decay: float,
    epochs: int,
    patience: int,
) -> tuple[BayesianMLP, pd.DataFrame]:
    model = BayesianMLP(
        hidden_width=hidden_width,
        hidden_layers=hidden_layers,
        activation=activation,
        init_sigma_scaled=init_scatter_dex / scaler.y_std,
        prior_sigma=prior_sigma,
        rho_init=rho_init,
    ).to(train_data["x"].device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = np.inf
    stale = 0
    rows = []
    n_train = train_data["x"].shape[0]

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        train_pred = model(train_data["x"], sample=True)
        nll = gaussian_nll(
            train_pred,
            train_data["y"],
            train_data["yerr"],
            model.sigma_int(),
        ).mean()
        warmup = min(1.0, epoch / max(1, kl_warmup_epochs))
        effective_kl_weight = kl_weight * warmup
        kl = model.kl_divergence() / n_train
        loss = nll + effective_kl_weight * kl
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(val_data["x"], sample=False)
            val_nll = gaussian_nll(
                val_pred,
                val_data["y"],
                val_data["yerr"],
                model.sigma_int(),
            ).mean()
            val_value = float(val_nll.cpu())
            rows.append(
                {
                    "method": "Bayesian Neural Network",
                    "epoch": epoch,
                    "train_loss": float(loss.detach().cpu()),
                    "train_nll": float(nll.detach().cpu()),
                    "train_kl_per_point": float(kl.detach().cpu()),
                    "val_loss": val_value,
                    "sigma_int_dex": float(model.sigma_int().detach().cpu()) * scaler.y_std,
                    "kl_weight": effective_kl_weight,
                }
            )

        if val_value < best_val - 1e-5:
            best_val = val_value
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, pd.DataFrame(rows)


def predict_nn_mean(
    model: DeterministicMLP | BayesianMLP,
    x: np.ndarray,
    scaler: Standardizer,
    device: torch.device,
    bayesian: bool,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_in = x_tensor(x, scaler, device)
        if bayesian:
            y_scaled = model(x_in, sample=False)
        else:
            y_scaled = model(x_in)
    return y_scaled.cpu().numpy().ravel() * scaler.y_std + scaler.y_mean


def predict_nn(
    model: DeterministicMLP,
    df: pd.DataFrame,
    scaler: Standardizer,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    mean = predict_nn_mean(model, df["log_mass"].to_numpy(), scaler, device, bayesian=False)
    sigma_int = float(model.sigma_int().detach().cpu()) * scaler.y_std
    model_sigma = np.full(len(df), sigma_int)
    return mean, model_sigma


def predict_bnn(
    model: BayesianMLP,
    df: pd.DataFrame,
    scaler: Standardizer,
    device: torch.device,
    n_samples: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    x_in = x_tensor(df["log_mass"].to_numpy(), scaler, device)
    draws = []
    with torch.no_grad():
        for _ in range(n_samples):
            y_scaled = model(x_in, sample=True)
            draws.append(y_scaled.cpu().numpy().ravel() * scaler.y_std + scaler.y_mean)
    draw_array = np.asarray(draws)
    mean = np.mean(draw_array, axis=0)
    epistemic_sigma = np.std(draw_array, axis=0, ddof=1)
    sigma_int = float(model.sigma_int().detach().cpu()) * scaler.y_std
    model_sigma = np.sqrt(epistemic_sigma**2 + sigma_int**2)
    return mean, model_sigma, epistemic_sigma


def estimate_slope(
    model: DeterministicMLP | BayesianMLP,
    x: np.ndarray,
    scaler: Standardizer,
    device: torch.device,
    bayesian: bool,
) -> np.ndarray:
    x = np.asarray(x)
    step = max(1e-4, 1e-4 * (float(np.max(x)) - float(np.min(x))))
    y_plus = predict_nn_mean(model, x + step, scaler, device, bayesian=bayesian)
    y_minus = predict_nn_mean(model, x - step, scaler, device, bayesian=bayesian)
    return (y_plus - y_minus) / (2.0 * step)


def total_sigma(
    df: pd.DataFrame,
    model_sigma: np.ndarray,
    model: DeterministicMLP | BayesianMLP,
    scaler: Standardizer,
    device: torch.device,
    bayesian: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = df["log_mass"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()
    slope = estimate_slope(model, x, scaler, device, bayesian=bayesian)
    propagated = np.abs(slope * xerr)
    sigma_total = np.sqrt(model_sigma**2 + yerr**2 + propagated**2)
    return sigma_total, propagated, slope


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    squared_error = residual**2
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    if len(squared_error) > 1 and rmse > 0:
        rmse_se = float(
            np.std(squared_error, ddof=1) / np.sqrt(len(squared_error)) / (2.0 * rmse)
        )
    else:
        rmse_se = 0.0
    return {
        "rmse": rmse,
        "rmse_se": rmse_se,
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def predictive_pte(
    y_true: np.ndarray,
    y_mean: np.ndarray,
    sigma_total: np.ndarray,
    n_samples: int,
    seed: int,
) -> float:
    rng = np.random.default_rng(seed)
    variance = sigma_total**2
    t_data = float(np.sum((y_true - y_mean) ** 2 / variance))
    t_rep = np.empty(n_samples)
    for i in range(n_samples):
        y_rep = rng.normal(y_mean, sigma_total)
        t_rep[i] = float(np.sum((y_rep - y_mean) ** 2 / variance))
    return float(np.mean(t_rep >= t_data))


def evaluate_method(
    method: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    test_model_sigma: np.ndarray,
    test_total_sigma: np.ndarray,
    test_propagated: np.ndarray,
    test_slope: np.ndarray,
    epistemic_sigma: np.ndarray | None,
    ppd_samples: int,
    seed: int,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
    train_metrics = regression_metrics(train_df["log_radius"].to_numpy(), train_pred)
    val_metrics = regression_metrics(val_df["log_radius"].to_numpy(), val_pred)
    test_metrics = regression_metrics(test_df["log_radius"].to_numpy(), test_pred)
    y_test = test_df["log_radius"].to_numpy()
    distance = np.abs(y_test - test_pred)
    distance_over_sigma = distance / test_total_sigma
    within_1sigma = distance <= test_total_sigma
    within_2sigma = distance <= 2.0 * test_total_sigma
    pte = predictive_pte(y_test, test_pred, test_total_sigma, ppd_samples, seed)

    row = {
        "method": method,
        "n_train": len(train_df),
        "n_validation": len(val_df),
        "n_test": len(test_df),
        "train_rmse": train_metrics["rmse"],
        "train_rmse_se": train_metrics["rmse_se"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "validation_rmse": val_metrics["rmse"],
        "validation_rmse_se": val_metrics["rmse_se"],
        "validation_mae": val_metrics["mae"],
        "validation_r2": val_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_rmse_se": test_metrics["rmse_se"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_ppd_pte": pte,
        "test_fraction_within_1sigma": float(np.mean(within_1sigma)),
        "test_fraction_within_2sigma": float(np.mean(within_2sigma)),
        "test_median_distance_over_sigma": float(np.median(distance_over_sigma)),
        "test_median_model_sigma": float(np.median(test_model_sigma)),
        "test_median_total_sigma": float(np.median(test_total_sigma)),
    }

    table = pd.DataFrame(
        {
            "method": method,
            "split_row_id": test_df["split_row_id"].to_numpy(),
            "mass": test_df["mass"].to_numpy(),
            "radius": test_df["radius"].to_numpy(),
            "log_mass": test_df["log_mass"].to_numpy(),
            "log_radius": y_test,
            "log_mass_err": test_df["log_mass_err"].to_numpy(),
            "log_radius_err": test_df["log_radius_err"].to_numpy(),
            "predicted_log_radius": test_pred,
            "residual_log_radius": y_test - test_pred,
            "distance_log_radius": distance,
            "model_sigma": test_model_sigma,
            "epistemic_sigma": np.zeros(len(test_df)) if epistemic_sigma is None else epistemic_sigma,
            "propagated_log_radius_err": test_propagated,
            "local_slope": test_slope,
            "sigma_total": test_total_sigma,
            "distance_over_sigma": distance_over_sigma,
            "within_1sigma": within_1sigma,
            "within_2sigma": within_2sigma,
        }
    )
    return row, table


def plot_training_history(history: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for method, group in history.groupby("method", sort=False):
        ax.plot(group["epoch"], group["train_loss"], lw=1.4, label=f"{method} train")
        ax.plot(group["epoch"], group["val_loss"], lw=1.4, ls="--", label=f"{method} validation")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("NN and BNN training history")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_fit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    nn_model: DeterministicMLP,
    bnn_model: BayesianMLP,
    scaler: Standardizer,
    device: torch.device,
    bnn_samples: int,
    output: Path,
) -> None:
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    x_grid = np.linspace(all_df["log_mass"].min(), all_df["log_mass"].max(), 500)
    grid_df = pd.DataFrame({"log_mass": x_grid})

    nn_mean = predict_nn_mean(nn_model, x_grid, scaler, device, bayesian=False)
    nn_sigma = float(nn_model.sigma_int().detach().cpu()) * scaler.y_std

    bnn_mean, bnn_model_sigma, _ = predict_bnn(bnn_model, grid_df, scaler, device, bnn_samples)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.errorbar(
        train_df["log_mass"],
        train_df["log_radius"],
        xerr=train_df["log_mass_err"],
        yerr=train_df["log_radius_err"],
        fmt=".",
        color="0.72",
        alpha=0.45,
        label="training planets",
    )
    ax.errorbar(
        test_df["log_mass"],
        test_df["log_radius"],
        xerr=test_df["log_mass_err"],
        yerr=test_df["log_radius_err"],
        fmt=".",
        color="black",
        alpha=0.62,
        label="test planets",
    )
    ax.plot(x_grid, nn_mean, color="C1", lw=2, label="NN mean")
    ax.fill_between(
        x_grid,
        nn_mean - nn_sigma,
        nn_mean + nn_sigma,
        color="C1",
        alpha=0.12,
        linewidth=0,
        label="NN intrinsic scatter",
    )
    ax.plot(x_grid, bnn_mean, color="C0", lw=2, label="BNN posterior mean")
    ax.fill_between(
        x_grid,
        bnn_mean - bnn_model_sigma,
        bnn_mean + bnn_model_sigma,
        color="C0",
        alpha=0.18,
        linewidth=0,
        label="BNN 68% model band",
    )
    ax.fill_between(
        x_grid,
        bnn_mean - 1.96 * bnn_model_sigma,
        bnn_mean + 1.96 * bnn_model_sigma,
        color="C0",
        alpha=0.08,
        linewidth=0,
        label="BNN 95% model band",
    )
    ax.set_xlabel(r"$\log_{10}(M/M_\oplus)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\oplus)$")
    ax.set_title("Neural-network mass-radius relation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def plot_predicted_vs_observed(predictions: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for method, group in predictions.groupby("method", sort=False):
        ax.errorbar(
            group["log_radius"],
            group["predicted_log_radius"],
            yerr=group["model_sigma"],
            fmt=".",
            alpha=0.65,
            label=method,
        )
    min_value = float(
        min(predictions["log_radius"].min(), predictions["predicted_log_radius"].min())
    )
    max_value = float(
        max(predictions["log_radius"].max(), predictions["predicted_log_radius"].max())
    )
    padding = 0.04 * (max_value - min_value)
    ax.plot(
        [min_value - padding, max_value + padding],
        [min_value - padding, max_value + padding],
        color="black",
        ls=":",
        lw=1.3,
        label="perfect prediction",
    )
    ax.set_xlim(min_value - padding, max_value + padding)
    ax.set_ylim(min_value - padding, max_value + padding)
    ax.set_xlabel(r"Observed $\log_{10}(R/R_\oplus)$")
    ax.set_ylabel(r"Predicted $\log_{10}(R/R_\oplus)$")
    ax.set_title("Held-out test predictions")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def parse_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hidden-width", type=int, default=24)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--activation", choices=("tanh", "silu", "relu"), default="tanh")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="fraction of the non-test data reserved for early stopping",
    )
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--bnn-epochs", type=int, default=6000)
    parser.add_argument("--patience", type=int, default=700)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--bnn-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--init-intrinsic-scatter", type=float, default=0.08)
    parser.add_argument("--bnn-prior-sigma", type=float, default=1.5)
    parser.add_argument("--bnn-rho-init", type=float, default=-5.0)
    parser.add_argument(
        "--bnn-kl-weight",
        type=float,
        default=0.10,
        help="tempered KL weight; use 1.0 for the untempered variational objective",
    )
    parser.add_argument("--bnn-kl-warmup-epochs", type=int, default=1000)
    parser.add_argument("--bnn-predictive-samples", type=int, default=300)
    parser.add_argument("--ppd-samples", type=int, default=1000)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = parse_device(args.device)
    PLOT_DIR.mkdir(exist_ok=True)

    raw_df = MR.load_dace_data()
    if args.max_points is not None and args.max_points < len(raw_df):
        rng = np.random.default_rng(args.seed)
        raw_df = raw_df.iloc[rng.choice(len(raw_df), size=args.max_points, replace=False)]
        raw_df = raw_df.sort_values("mass").reset_index(drop=True)
        print(f"Using a random subset of {len(raw_df)} planets for this run.")

    train_df, val_df, test_df = make_shared_splits(
        raw_df,
        test_size=args.test_size,
        validation_size=args.validation_size,
        seed=args.seed,
    )
    train_df.to_csv(PLOT_DIR / "nn_bnn_mass_radius_train.csv", index=False)
    val_df.to_csv(PLOT_DIR / "nn_bnn_mass_radius_validation.csv", index=False)
    test_df.to_csv(PLOT_DIR / "nn_bnn_mass_radius_test.csv", index=False)

    scaler = fit_standardizer(train_df)
    train_data = tensor_data(train_df, scaler, device)
    val_data = tensor_data(val_df, scaler, device)

    print(
        "Shared split: "
        f"{len(train_df)} training, {len(val_df)} validation, {len(test_df)} test planets."
    )
    print(
        "Architecture: "
        f"1 -> {args.hidden_width} x {args.hidden_layers} -> 1, "
        f"activation={args.activation}, device={device}"
    )

    print("\n=== Deterministic neural network ===")
    nn_model, nn_history = train_deterministic_nn(
        train_data=train_data,
        val_data=val_data,
        scaler=scaler,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        init_scatter_dex=args.init_intrinsic_scatter,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        patience=args.patience,
    )

    print("\n=== Bayesian neural network ===")
    bnn_model, bnn_history = train_bayesian_nn(
        train_data=train_data,
        val_data=val_data,
        scaler=scaler,
        hidden_width=args.hidden_width,
        hidden_layers=args.hidden_layers,
        activation=args.activation,
        init_scatter_dex=args.init_intrinsic_scatter,
        prior_sigma=args.bnn_prior_sigma,
        rho_init=args.bnn_rho_init,
        kl_weight=args.bnn_kl_weight,
        kl_warmup_epochs=args.bnn_kl_warmup_epochs,
        learning_rate=args.bnn_learning_rate,
        weight_decay=args.weight_decay,
        epochs=args.bnn_epochs,
        patience=args.patience,
    )

    history = pd.concat([nn_history, bnn_history], ignore_index=True)
    history_path = PLOT_DIR / "nn_bnn_mass_radius_training_history.csv"
    history.to_csv(history_path, index=False)

    nn_train_pred, _ = predict_nn(nn_model, train_df, scaler, device)
    nn_val_pred, _ = predict_nn(nn_model, val_df, scaler, device)
    nn_test_pred, nn_test_model_sigma = predict_nn(nn_model, test_df, scaler, device)
    nn_total, nn_propagated, nn_slope = total_sigma(
        test_df,
        nn_test_model_sigma,
        nn_model,
        scaler,
        device,
        bayesian=False,
    )

    bnn_train_pred, _, _ = predict_bnn(
        bnn_model, train_df, scaler, device, args.bnn_predictive_samples
    )
    bnn_val_pred, _, _ = predict_bnn(
        bnn_model, val_df, scaler, device, args.bnn_predictive_samples
    )
    bnn_test_pred, bnn_test_model_sigma, bnn_epistemic_sigma = predict_bnn(
        bnn_model, test_df, scaler, device, args.bnn_predictive_samples
    )
    bnn_total, bnn_propagated, bnn_slope = total_sigma(
        test_df,
        bnn_test_model_sigma,
        bnn_model,
        scaler,
        device,
        bayesian=True,
    )

    nn_row, nn_predictions = evaluate_method(
        "Neural Network",
        train_df,
        val_df,
        test_df,
        nn_train_pred,
        nn_val_pred,
        nn_test_pred,
        nn_test_model_sigma,
        nn_total,
        nn_propagated,
        nn_slope,
        epistemic_sigma=None,
        ppd_samples=args.ppd_samples,
        seed=args.seed,
    )
    bnn_row, bnn_predictions = evaluate_method(
        "Bayesian Neural Network",
        train_df,
        val_df,
        test_df,
        bnn_train_pred,
        bnn_val_pred,
        bnn_test_pred,
        bnn_test_model_sigma,
        bnn_total,
        bnn_propagated,
        bnn_slope,
        epistemic_sigma=bnn_epistemic_sigma,
        ppd_samples=args.ppd_samples,
        seed=args.seed + 1,
    )

    for row, model in ((nn_row, nn_model), (bnn_row, bnn_model)):
        row["hidden_width"] = args.hidden_width
        row["hidden_layers"] = args.hidden_layers
        row["activation"] = args.activation
        row["intrinsic_sigma_dex"] = float(model.sigma_int().detach().cpu()) * scaler.y_std
    bnn_row["bnn_prior_sigma"] = args.bnn_prior_sigma
    bnn_row["bnn_kl_weight"] = args.bnn_kl_weight
    bnn_row["bnn_predictive_samples"] = args.bnn_predictive_samples

    summary = pd.DataFrame([nn_row, bnn_row]).sort_values("test_rmse").reset_index(drop=True)
    summary["test_rmse_rank"] = np.arange(1, len(summary) + 1)
    summary_path = PLOT_DIR / "nn_bnn_mass_radius_summary.csv"
    summary.to_csv(summary_path, index=False)

    predictions = pd.concat([nn_predictions, bnn_predictions], ignore_index=True)
    predictions_path = PLOT_DIR / "nn_bnn_mass_radius_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    if not args.no_plots:
        plot_training_history(history, PLOT_DIR / "nn_bnn_mass_radius_training_history.png")
        plot_fit(
            train_df,
            test_df,
            nn_model,
            bnn_model,
            scaler,
            device,
            args.bnn_predictive_samples,
            PLOT_DIR / "nn_bnn_mass_radius_fit.png",
        )
        plot_predicted_vs_observed(
            predictions,
            PLOT_DIR / "nn_bnn_mass_radius_predicted_vs_observed.png",
        )

    print("\nNN and BNN comparison")
    print(
        summary[
            [
                "test_rmse_rank",
                "method",
                "train_rmse",
                "validation_rmse",
                "test_rmse",
                "test_rmse_se",
                "test_mae",
                "test_r2",
                "test_ppd_pte",
                "test_fraction_within_1sigma",
                "test_fraction_within_2sigma",
                "intrinsic_sigma_dex",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved test predictions to {predictions_path}")
    print(f"Saved training history to {history_path}")
    if not args.no_plots:
        print(f"Saved plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
