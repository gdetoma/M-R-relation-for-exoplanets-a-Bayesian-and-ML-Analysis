"""
Gaussian-process analysis of the exoplanet radius-mass relation.

The model is fitted in log space:

    log10(R / R_earth) = f(log10(M / M_earth))

The GP uses heteroscedastic measured radius uncertainties, iteratively
propagates the measured mass uncertainty through the local GP slope, and learns
a global white-noise term that plays the role of intrinsic scatter.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.linalg
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit, train_test_split

try:
    import emcee
except ImportError:
    emcee = None

try:
    import corner
except ImportError:
    corner = None


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"
RNG = np.random.default_rng(42)
MASS_REGIME_BREAKPOINTS = (0.824, 2.196)
MASS_REGIME_NAMES = ("low_mass", "intermediate_mass", "giant_planet")
KERNEL_NAMES = ("rbf", "matern32", "matern52", "rq")
HYPERPARAMETER_PRIOR_BOUNDS = {
    "sigma_f": (0.03, 2.0),
    "length_scale": (0.05, 10.0),
    "sigma_white": (0.005, 0.5),
    "rq_alpha": (0.05, 50.0),
}


def load_mass_radius_module():
    """Load helpers from emcee_M-R.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("emcee_mr_helpers", EMCEE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["emcee_mr_helpers"] = module
    spec.loader.exec_module(module)
    return module


MR = load_mass_radius_module()


def mass_regime_codes(df: pd.DataFrame) -> np.ndarray:
    """Return low/intermediate/giant regime codes from Bayesian breakpoints."""
    return np.digitize(df["log_mass"].to_numpy(), MASS_REGIME_BREAKPOINTS)


def add_mass_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add a readable mass-regime label used for stratified splitting."""
    with_regime = df.copy()
    codes = mass_regime_codes(with_regime)
    with_regime["mass_regime"] = [MASS_REGIME_NAMES[code] for code in codes]
    return with_regime


def regime_counts(df: pd.DataFrame) -> dict[str, int]:
    """Count rows in the three mass regimes."""
    codes = mass_regime_codes(df)
    return {
        name: int(np.sum(codes == index))
        for index, name in enumerate(MASS_REGIME_NAMES)
    }


def format_regime_counts(df: pd.DataFrame) -> str:
    counts = regime_counts(df)
    return ", ".join(f"{name}={counts[name]}" for name in MASS_REGIME_NAMES)


def split_train_test(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Make a reproducible mass-regime-stratified train/test split."""
    df = add_mass_regime(df)
    indices = np.arange(len(df))
    regimes = mass_regime_codes(df)
    if np.min(np.bincount(regimes, minlength=len(MASS_REGIME_NAMES))) >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(indices, regimes))
    else:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            shuffle=True,
        )
    train_df = df.iloc[train_idx].sort_values("mass").reset_index(drop=True)
    test_df = df.iloc[test_idx].sort_values("mass").reset_index(drop=True)
    return train_df, test_df


def cv_split_indices(
    df: pd.DataFrame,
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return stratified CV indices when each regime can support it."""
    n_splits = min(n_splits, len(df))
    if n_splits < 2:
        raise ValueError("At least two data points are required for cross-validation.")

    regimes = mass_regime_codes(df)
    regime_min_count = int(np.min(np.bincount(regimes, minlength=len(MASS_REGIME_NAMES))))
    if regime_min_count >= n_splits:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.arange(len(df)), regimes))

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.arange(len(df))))


@dataclass
class GPFit:
    model: GaussianProcessRegressor
    y_mean: float
    training_alpha: np.ndarray
    training_slope: np.ndarray
    kernel_name: str


@dataclass
class GPHyperparameterResult:
    samples: pd.DataFrame
    summary: pd.DataFrame
    metrics: dict[str, float | int | str]
    predictions: pd.DataFrame
    acceptance_fraction: float
    n_prediction_samples: int


def make_kernel(kernel_name: str):
    """Create a one-dimensional GP kernel for the mass-radius relation."""
    amplitude = ConstantKernel(1.0, (1e-3, 1e3))
    white_noise = WhiteKernel(noise_level=0.03**2, noise_level_bounds=(1e-6, 1.0))

    if kernel_name == "rbf":
        smooth_kernel = RBF(length_scale=0.7, length_scale_bounds=(1e-2, 1e2))
    elif kernel_name == "matern32":
        smooth_kernel = Matern(
            length_scale=0.7,
            length_scale_bounds=(1e-2, 1e2),
            nu=1.5,
        )
    elif kernel_name == "matern52":
        smooth_kernel = Matern(
            length_scale=0.7,
            length_scale_bounds=(1e-2, 1e2),
            nu=2.5,
        )
    elif kernel_name == "rq":
        smooth_kernel = RationalQuadratic(
            length_scale=0.7,
            alpha=1.0,
            length_scale_bounds=(1e-2, 1e2),
            alpha_bounds=(1e-2, 1e2),
        )
    else:
        raise ValueError(f"Unknown kernel: {kernel_name}")

    return amplitude * smooth_kernel + white_noise


def predict_gp(fit: GPFit, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return GP predictive mean and standard deviation in log-radius dex."""
    mean_centered, std = fit.model.predict(np.asarray(x).reshape(-1, 1), return_std=True)
    return mean_centered + fit.y_mean, std


def estimate_gp_slope(fit: GPFit, x: np.ndarray) -> np.ndarray:
    """Finite-difference derivative d logR / d logM of the GP mean."""
    x = np.asarray(x)
    step = max(1e-4, 1e-4 * (float(np.max(x)) - float(np.min(x))))
    mean_plus, _ = predict_gp(fit, x + step)
    mean_minus, _ = predict_gp(fit, x - step)
    return (mean_plus - mean_minus) / (2.0 * step)


def fit_gp_model(
    df: pd.DataFrame,
    kernel_name: str,
    n_restarts: int,
    propagation_iterations: int,
    seed: int,
    jitter_floor: float,
) -> GPFit:
    """Fit a GP while iteratively updating the propagated x-error term."""
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    y_mean = float(np.mean(y))
    y_centered = y - y_mean
    initial_slope = np.polyfit(x, y, deg=1)[0]
    slope = np.full_like(x, initial_slope)
    alpha = yerr**2 + (slope * xerr) ** 2 + jitter_floor**2
    kernel = make_kernel(kernel_name)
    fit = None

    for _ in range(propagation_iterations + 1):
        gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=False,
            n_restarts_optimizer=n_restarts,
            random_state=seed,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            gp.fit(x.reshape(-1, 1), y_centered)

        fit = GPFit(
            model=gp,
            y_mean=y_mean,
            training_alpha=alpha,
            training_slope=slope,
            kernel_name=kernel_name,
        )
        slope = estimate_gp_slope(fit, x)
        alpha = yerr**2 + (slope * xerr) ** 2 + jitter_floor**2
        kernel = gp.kernel_

    return GPFit(
        model=fit.model,
        y_mean=y_mean,
        training_alpha=alpha,
        training_slope=slope,
        kernel_name=kernel_name,
    )


def kernel_white_noise_variance(kernel) -> float:
    """Extract any WhiteKernel variance from a fitted kernel tree."""
    if isinstance(kernel, WhiteKernel):
        return float(kernel.noise_level)
    variance = 0.0
    for child_name in ("k1", "k2"):
        child = getattr(kernel, child_name, None)
        if child is not None:
            variance += kernel_white_noise_variance(child)
    return variance


def hyperparameter_names(kernel_name: str) -> tuple[str, ...]:
    """Return sampled log-hyperparameter names for a kernel."""
    names = ("log_sigma_f", "log_length_scale", "log_sigma_white")
    if kernel_name == "rq":
        names += ("log_rq_alpha",)
    return names


def physical_name(log_name: str) -> str:
    """Strip the log_ prefix used by the MCMC parameterization."""
    return log_name.removeprefix("log_")


def log_prior_bounds(kernel_name: str) -> dict[str, tuple[float, float]]:
    """Return finite log-uniform prior bounds for sampled hyperparameters."""
    bounds = {}
    for name in hyperparameter_names(kernel_name):
        low, high = HYPERPARAMETER_PRIOR_BOUNDS[physical_name(name)]
        bounds[name] = (float(np.log(low)), float(np.log(high)))
    return bounds


def optimized_kernel_hyperparameters(fit: GPFit) -> dict[str, float]:
    """Extract physical hyperparameters from the optimized sklearn kernel."""
    kernel = fit.model.kernel_
    product_kernel = kernel.k1
    smooth_kernel = product_kernel.k2
    params = {
        "sigma_f": float(np.sqrt(product_kernel.k1.constant_value)),
        "length_scale": float(np.ravel(smooth_kernel.length_scale)[0]),
        "sigma_white": float(np.sqrt(kernel_white_noise_variance(kernel))),
    }
    if fit.kernel_name == "rq":
        params["rq_alpha"] = float(smooth_kernel.alpha)
    return params


def optimized_hyperparameter_vector(fit: GPFit) -> np.ndarray:
    """Return the optimized kernel hyperparameters in the sampled log space."""
    optimized = optimized_kernel_hyperparameters(fit)
    bounds = log_prior_bounds(fit.kernel_name)
    theta = []
    for name in hyperparameter_names(fit.kernel_name):
        value = np.log(optimized[physical_name(name)])
        low, high = bounds[name]
        theta.append(float(np.clip(value, low + 1e-6, high - 1e-6)))
    return np.array(theta)


def theta_to_physical(theta: np.ndarray, kernel_name: str) -> dict[str, float]:
    """Convert sampled log-hyperparameters to physical kernel parameters."""
    return {
        physical_name(name): float(np.exp(value))
        for name, value in zip(hyperparameter_names(kernel_name), theta)
    }


def log_hyperparameter_prior(theta: np.ndarray, kernel_name: str) -> float:
    """Finite log-uniform priors expressed as uniform priors in log space."""
    if len(theta) != len(hyperparameter_names(kernel_name)):
        return -np.inf
    for value, name in zip(theta, hyperparameter_names(kernel_name)):
        low, high = log_prior_bounds(kernel_name)[name]
        if not low <= value <= high:
            return -np.inf
    return 0.0


def smooth_covariance(
    x1: np.ndarray,
    x2: np.ndarray,
    kernel_name: str,
    params: dict[str, float],
) -> np.ndarray:
    """Evaluate the smooth one-dimensional covariance matrix."""
    x1 = np.asarray(x1, dtype=float).reshape(-1)
    x2 = np.asarray(x2, dtype=float).reshape(-1)
    scaled_distance = np.abs(x1[:, None] - x2[None, :]) / params["length_scale"]
    variance = params["sigma_f"] ** 2

    if kernel_name == "rbf":
        return variance * np.exp(-0.5 * scaled_distance**2)
    if kernel_name == "matern32":
        root3_distance = np.sqrt(3.0) * scaled_distance
        return variance * (1.0 + root3_distance) * np.exp(-root3_distance)
    if kernel_name == "matern52":
        root5_distance = np.sqrt(5.0) * scaled_distance
        return (
            variance
            * (1.0 + root5_distance + 5.0 * scaled_distance**2 / 3.0)
            * np.exp(-root5_distance)
        )
    if kernel_name == "rq":
        alpha = params["rq_alpha"]
        return variance * (1.0 + scaled_distance**2 / (2.0 * alpha)) ** (-alpha)
    raise ValueError(f"Unknown kernel: {kernel_name}")


def factor_training_covariance(
    train_x: np.ndarray,
    training_alpha: np.ndarray,
    theta: np.ndarray,
    kernel_name: str,
) -> tuple[tuple[np.ndarray, bool], dict[str, float]]:
    """Return a Cholesky factor for the sampled GP training covariance."""
    params = theta_to_physical(theta, kernel_name)
    covariance = smooth_covariance(train_x, train_x, kernel_name, params)
    covariance[np.diag_indices_from(covariance)] += (
        np.asarray(training_alpha) + params["sigma_white"] ** 2
    )
    factor = scipy.linalg.cho_factor(covariance, lower=True, check_finite=False)
    return factor, params


def gp_log_marginal_likelihood(
    theta: np.ndarray,
    train_x: np.ndarray,
    y_centered: np.ndarray,
    training_alpha: np.ndarray,
    kernel_name: str,
) -> float:
    """Direct GP log marginal likelihood for sampled hyperparameters."""
    try:
        factor, _ = factor_training_covariance(
            train_x=train_x,
            training_alpha=training_alpha,
            theta=theta,
            kernel_name=kernel_name,
        )
        alpha_vector = scipy.linalg.cho_solve(
            factor,
            y_centered,
            check_finite=False,
        )
    except (np.linalg.LinAlgError, ValueError):
        return -np.inf

    chol, _ = factor
    log_det = 2.0 * np.sum(np.log(np.diag(chol)))
    n_data = len(y_centered)
    return float(
        -0.5 * y_centered @ alpha_vector
        -0.5 * log_det
        -0.5 * n_data * np.log(2.0 * np.pi)
    )


def gp_log_posterior(
    theta: np.ndarray,
    train_x: np.ndarray,
    y_centered: np.ndarray,
    training_alpha: np.ndarray,
    kernel_name: str,
) -> float:
    """Log posterior for kernel hyperparameter sampling."""
    log_prior = log_hyperparameter_prior(theta, kernel_name)
    if not np.isfinite(log_prior):
        return -np.inf
    return log_prior + gp_log_marginal_likelihood(
        theta=theta,
        train_x=train_x,
        y_centered=y_centered,
        training_alpha=training_alpha,
        kernel_name=kernel_name,
    )


def initialize_hyperparameter_walkers(
    theta0: np.ndarray,
    kernel_name: str,
    n_walkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Initialize walkers near the optimized empirical-Bayes solution."""
    names = hyperparameter_names(kernel_name)
    bounds = log_prior_bounds(kernel_name)
    lower = np.array([bounds[name][0] for name in names])
    upper = np.array([bounds[name][1] for name in names])
    width = upper - lower
    position = theta0 + rng.normal(scale=0.02 * width, size=(n_walkers, len(theta0)))
    return np.clip(position, lower + 1e-6, upper - 1e-6)


def hyperparameter_samples_dataframe(
    samples: np.ndarray,
    log_probability: np.ndarray,
    kernel_name: str,
) -> pd.DataFrame:
    """Return posterior samples with both log and physical columns."""
    names = hyperparameter_names(kernel_name)
    table = pd.DataFrame(samples, columns=names)
    table.insert(0, "sample_id", np.arange(len(table)))
    table.insert(1, "kernel", kernel_name)
    table["log_posterior"] = log_probability
    for name in names:
        table[physical_name(name)] = np.exp(table[name])
    if "rq_alpha" not in table:
        table["rq_alpha"] = "not_applicable"
    return table


def summarize_hyperparameter_samples(
    samples: pd.DataFrame,
    kernel_name: str,
    acceptance_fraction: float,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    thin: int,
) -> pd.DataFrame:
    """Summarize sampled hyperparameters and basic MCMC metadata."""
    quantities = ["sigma_f", "length_scale", "sigma_white"]
    if kernel_name == "rq":
        quantities.append("rq_alpha")

    rows = []
    for quantity in quantities:
        values = samples[quantity].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "quantity": quantity,
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "q16": float(np.percentile(values, 16)),
                "median": float(np.median(values)),
                "q84": float(np.percentile(values, 84)),
            }
        )

    rows.extend(
        [
            {
                "quantity": "acceptance_fraction",
                "mean": acceptance_fraction,
                "sd": 0.0,
                "q16": acceptance_fraction,
                "median": acceptance_fraction,
                "q84": acceptance_fraction,
            },
            {
                "quantity": "n_posterior_samples",
                "mean": len(samples),
                "sd": 0.0,
                "q16": len(samples),
                "median": len(samples),
                "q84": len(samples),
            },
            {
                "quantity": "n_walkers",
                "mean": n_walkers,
                "sd": 0.0,
                "q16": n_walkers,
                "median": n_walkers,
                "q84": n_walkers,
            },
            {
                "quantity": "n_steps",
                "mean": n_steps,
                "sd": 0.0,
                "q16": n_steps,
                "median": n_steps,
                "q84": n_steps,
            },
            {
                "quantity": "n_burn",
                "mean": n_burn,
                "sd": 0.0,
                "q16": n_burn,
                "median": n_burn,
                "q84": n_burn,
            },
            {
                "quantity": "thin",
                "mean": thin,
                "sd": 0.0,
                "q16": thin,
                "median": thin,
                "q84": thin,
            },
        ]
    )
    return pd.DataFrame(rows)


def sample_kernel_hyperparameters(
    train_df: pd.DataFrame,
    fit: GPFit,
    n_walkers: int,
    n_steps: int,
    n_burn: int,
    thin: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Sample selected-kernel hyperparameters with emcee."""
    if emcee is None:
        raise ImportError("emcee is required for --sample-hyperparameters.")
    if n_steps <= n_burn:
        raise ValueError("--mcmc-steps must be larger than --mcmc-burn.")
    if thin < 1:
        raise ValueError("--mcmc-thin must be at least 1.")

    train_x = train_df["log_mass"].to_numpy()
    y_centered = train_df["log_radius"].to_numpy() - fit.y_mean
    theta0 = optimized_hyperparameter_vector(fit)
    n_dim = len(theta0)
    if n_walkers < 2 * n_dim:
        adjusted_walkers = 2 * n_dim
        print(
            f"Raising mcmc walkers from {n_walkers} to {adjusted_walkers} "
            "for emcee's ensemble sampler."
        )
        n_walkers = adjusted_walkers

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    initial_position = initialize_hyperparameter_walkers(
        theta0=theta0,
        kernel_name=fit.kernel_name,
        n_walkers=n_walkers,
        rng=rng,
    )

    sampler = emcee.EnsembleSampler(
        n_walkers,
        n_dim,
        gp_log_posterior,
        args=(train_x, y_centered, fit.training_alpha, fit.kernel_name),
    )
    sampler.run_mcmc(
        initial_position,
        n_steps,
        progress=False,
        skip_initial_state_check=True,
    )

    flat_samples = sampler.get_chain(discard=n_burn, thin=thin, flat=True)
    flat_log_probability = sampler.get_log_prob(discard=n_burn, thin=thin, flat=True)
    if len(flat_samples) == 0:
        raise ValueError("No posterior samples remain after burn-in and thinning.")

    samples = hyperparameter_samples_dataframe(
        samples=flat_samples,
        log_probability=flat_log_probability,
        kernel_name=fit.kernel_name,
    )
    acceptance_fraction = float(np.mean(sampler.acceptance_fraction))
    summary = summarize_hyperparameter_samples(
        samples=samples,
        kernel_name=fit.kernel_name,
        acceptance_fraction=acceptance_fraction,
        n_walkers=n_walkers,
        n_steps=n_steps,
        n_burn=n_burn,
        thin=thin,
    )
    return samples, summary, acceptance_fraction


def conditional_gp_prediction(
    train_x: np.ndarray,
    prediction_x: np.ndarray,
    factor: tuple[np.ndarray, bool],
    alpha_vector: np.ndarray,
    kernel_name: str,
    params: dict[str, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Return latent GP conditional mean and variance for one theta sample."""
    cross_covariance = smooth_covariance(train_x, prediction_x, kernel_name, params)
    mean_centered = cross_covariance.T @ alpha_vector
    weights = scipy.linalg.cho_solve(factor, cross_covariance, check_finite=False)
    latent_variance = params["sigma_f"] ** 2 - np.sum(cross_covariance * weights, axis=0)
    return mean_centered, np.maximum(latent_variance, 1e-12)


def evaluate_hyperparameter_posterior_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit: GPFit,
    samples: pd.DataFrame,
    n_prediction_samples: int,
    ppd_samples: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame, int]:
    """Marginalize test predictions over sampled GP hyperparameters."""
    rng = np.random.default_rng(seed)
    if len(samples) > n_prediction_samples:
        selected_index = rng.choice(len(samples), size=n_prediction_samples, replace=False)
        prediction_samples = samples.iloc[np.sort(selected_index)].reset_index(drop=True)
    else:
        prediction_samples = samples.reset_index(drop=True)

    train_x = train_df["log_mass"].to_numpy()
    y_centered = train_df["log_radius"].to_numpy() - fit.y_mean
    test_x = test_df["log_mass"].to_numpy()
    test_xerr = test_df["log_mass_err"].to_numpy()
    test_yerr = test_df["log_radius_err"].to_numpy()
    step = max(1e-4, 1e-4 * (float(np.max(train_x)) - float(np.min(train_x))))
    names = hyperparameter_names(fit.kernel_name)

    mean_sum = np.zeros(len(test_df))
    total_second_moment_sum = np.zeros(len(test_df))
    model_second_moment_sum = np.zeros(len(test_df))
    propagated_variance_sum = np.zeros(len(test_df))
    slope_sum = np.zeros(len(test_df))

    for _, row in prediction_samples.iterrows():
        theta = row.loc[list(names)].to_numpy(dtype=float)
        factor, params = factor_training_covariance(
            train_x=train_x,
            training_alpha=fit.training_alpha,
            theta=theta,
            kernel_name=fit.kernel_name,
        )
        alpha_vector = scipy.linalg.cho_solve(factor, y_centered, check_finite=False)

        mean_centered, latent_variance = conditional_gp_prediction(
            train_x=train_x,
            prediction_x=test_x,
            factor=factor,
            alpha_vector=alpha_vector,
            kernel_name=fit.kernel_name,
            params=params,
        )
        mean = mean_centered + fit.y_mean

        mean_plus, _ = conditional_gp_prediction(
            train_x=train_x,
            prediction_x=test_x + step,
            factor=factor,
            alpha_vector=alpha_vector,
            kernel_name=fit.kernel_name,
            params=params,
        )
        mean_minus, _ = conditional_gp_prediction(
            train_x=train_x,
            prediction_x=test_x - step,
            factor=factor,
            alpha_vector=alpha_vector,
            kernel_name=fit.kernel_name,
            params=params,
        )
        slope = (mean_plus - mean_minus) / (2.0 * step)
        propagated = slope * test_xerr
        model_variance = latent_variance + params["sigma_white"] ** 2
        total_variance = model_variance + test_yerr**2 + propagated**2

        mean_sum += mean
        total_second_moment_sum += total_variance + mean**2
        model_second_moment_sum += model_variance + mean**2
        propagated_variance_sum += propagated**2
        slope_sum += slope

    n_used = len(prediction_samples)
    marginal_mean = mean_sum / n_used
    marginal_total_variance = total_second_moment_sum / n_used - marginal_mean**2
    marginal_model_variance = model_second_moment_sum / n_used - marginal_mean**2
    marginal_total_variance = np.maximum(marginal_total_variance, 1e-12)
    marginal_model_variance = np.maximum(marginal_model_variance, 1e-12)
    propagated_rms = np.sqrt(propagated_variance_sum / n_used)
    slope_mean = slope_sum / n_used

    metrics = evaluate_prediction_distribution(
        df=test_df,
        mean=marginal_mean,
        variance=marginal_total_variance,
        n_samples=ppd_samples,
        seed=seed,
    )
    predictions = prediction_diagnostics_table(
        df=test_df,
        mean=marginal_mean,
        variance=marginal_total_variance,
        gp_std=np.sqrt(marginal_model_variance),
        slope=slope_mean,
        propagated=propagated_rms,
        kernel_name=f"{fit.kernel_name}_hyperparameter_marginalized",
    )
    return metrics, predictions, n_used


def summary_median(summary: pd.DataFrame, quantity: str) -> float | str:
    """Read one median value from a hyperparameter summary table."""
    matches = summary.loc[summary["quantity"] == quantity, "median"]
    return float(matches.iloc[0]) if len(matches) else "not_applicable"


def plot_hyperparameter_corner(
    samples: pd.DataFrame,
    kernel_name: str,
    output: Path,
    show: bool = False,
) -> None:
    """Plot an optional corner plot for sampled log-hyperparameters."""
    if corner is None or len(samples) < 2:
        return
    names = list(hyperparameter_names(kernel_name))
    fig = corner.corner(samples[names].to_numpy(), labels=names, show_titles=True)
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def run_hyperparameter_sampling_analysis(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit: GPFit,
    optimized_test_metrics: dict[str, float | int],
    optimized_ppd_pte: float,
    ppd_samples: int,
    seed: int,
    mcmc_walkers: int,
    mcmc_steps: int,
    mcmc_burn: int,
    mcmc_thin: int,
    mcmc_prediction_samples: int,
    make_plots: bool,
    show: bool,
) -> GPHyperparameterResult:
    """Run the optional Bayesian kernel-hyperparameter robustness check."""
    print("\n=== GP kernel-hyperparameter sampling check ===")
    samples, summary, acceptance_fraction = sample_kernel_hyperparameters(
        train_df=train_df,
        fit=fit,
        n_walkers=mcmc_walkers,
        n_steps=mcmc_steps,
        n_burn=mcmc_burn,
        thin=mcmc_thin,
        seed=seed,
    )
    hyper_metrics, predictions, n_prediction_samples = (
        evaluate_hyperparameter_posterior_predictions(
            train_df=train_df,
            test_df=test_df,
            fit=fit,
            samples=samples,
            n_prediction_samples=mcmc_prediction_samples,
            ppd_samples=ppd_samples,
            seed=seed,
        )
    )

    samples_path = PLOT_DIR / "gp_mass_radius_hyperparameter_samples.csv"
    summary_path = PLOT_DIR / "gp_mass_radius_hyperparameter_summary.csv"
    predictions_path = PLOT_DIR / "gp_mass_radius_hyperparameter_test_predictions.csv"
    metrics_path = PLOT_DIR / "gp_mass_radius_hyperparameter_summary_metrics.csv"
    corner_path = PLOT_DIR / "gp_mass_radius_hyperparameter_corner.png"

    samples.to_csv(samples_path, index=False)
    summary.to_csv(summary_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    if make_plots:
        plot_hyperparameter_corner(samples, fit.kernel_name, corner_path, show=show)

    optimized_params = optimized_kernel_hyperparameters(fit)
    metrics_table = pd.DataFrame(
        [
            {
                "model": "optimized_gp",
                "kernel": fit.kernel_name,
                "rmse": optimized_test_metrics["rmse"],
                "rmse_se": optimized_test_metrics["rmse_se"],
                "mae": optimized_test_metrics["mae"],
                "r2": optimized_test_metrics["r2"],
                "nlpd": optimized_test_metrics["nlpd"],
                "nlpd_se": optimized_test_metrics["nlpd_se"],
                "ppd_pte": optimized_ppd_pte,
                "fraction_within_1sigma": optimized_test_metrics["fraction_within_1sigma"],
                "fraction_within_2sigma": optimized_test_metrics["fraction_within_2sigma"],
                "median_distance_over_sigma": optimized_test_metrics[
                    "median_distance_over_sigma"
                ],
                "sigma_white_dex": optimized_params["sigma_white"],
                "length_scale": optimized_params["length_scale"],
                "rq_alpha": optimized_params.get("rq_alpha", "not_applicable"),
                "acceptance_fraction": "not_applicable",
                "n_posterior_samples": "not_applicable",
                "n_prediction_samples": "not_applicable",
            },
            {
                "model": "hyperparameter_marginalized_gp",
                "kernel": fit.kernel_name,
                "rmse": hyper_metrics["rmse"],
                "rmse_se": hyper_metrics["rmse_se"],
                "mae": hyper_metrics["mae"],
                "r2": hyper_metrics["r2"],
                "nlpd": hyper_metrics["nlpd"],
                "nlpd_se": hyper_metrics["nlpd_se"],
                "ppd_pte": hyper_metrics["ppd_pte"],
                "fraction_within_1sigma": hyper_metrics["fraction_within_1sigma"],
                "fraction_within_2sigma": hyper_metrics["fraction_within_2sigma"],
                "median_distance_over_sigma": hyper_metrics[
                    "median_distance_over_sigma"
                ],
                "sigma_white_dex": summary_median(summary, "sigma_white"),
                "length_scale": summary_median(summary, "length_scale"),
                "rq_alpha": summary_median(summary, "rq_alpha"),
                "acceptance_fraction": acceptance_fraction,
                "n_posterior_samples": len(samples),
                "n_prediction_samples": n_prediction_samples,
            },
        ]
    )
    metrics_table.to_csv(metrics_path, index=False)

    print(
        f"Hyperparameter MCMC acceptance fraction = {acceptance_fraction:.3f}; "
        f"posterior samples = {len(samples)}, prediction samples = {n_prediction_samples}"
    )
    print(
        f"Marginalized test RMSE = {hyper_metrics['rmse']:.4f}, "
        f"NLPD = {hyper_metrics['nlpd']:.4f}, "
        f"PTE = {hyper_metrics['ppd_pte']:.3f}"
    )
    print(
        "Marginalized test coverage: "
        f"{hyper_metrics['fraction_within_1sigma']:.3f} within 1 sigma, "
        f"{hyper_metrics['fraction_within_2sigma']:.3f} within 2 sigma"
    )
    print(f"Saved hyperparameter samples to {samples_path}")

    return GPHyperparameterResult(
        samples=samples,
        summary=summary,
        metrics=hyper_metrics,
        predictions=predictions,
        acceptance_fraction=acceptance_fraction,
        n_prediction_samples=n_prediction_samples,
    )


def total_predictive_variance(
    fit: GPFit,
    x: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return total y variance, GP std, slope, and propagated x-error term."""
    _, gp_std = predict_gp(fit, x)
    slope = estimate_gp_slope(fit, x)
    propagated = slope * xerr
    variance = gp_std**2 + yerr**2 + propagated**2
    return variance, gp_std, slope, propagated


def residual_coverage_check(df: pd.DataFrame, fit: GPFit) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Check residual distances relative to total GP plus measurement sigma."""
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    fitted_y, _ = predict_gp(fit, x)
    variance, gp_std, slope, propagated = total_predictive_variance(fit, x, xerr, yerr)
    sigma_y = np.sqrt(variance)
    distance_y = np.abs(y - fitted_y)

    within_1sigma = distance_y <= sigma_y
    within_2sigma = distance_y <= 2.0 * sigma_y
    normalized_distance = distance_y / sigma_y

    table = pd.DataFrame(
        {
            "mass": df["mass"].to_numpy(),
            "radius": df["radius"].to_numpy(),
            "mass_regime": df.get(
                "mass_regime",
                pd.Series(
                    [MASS_REGIME_NAMES[code] for code in mass_regime_codes(df)],
                    index=df.index,
                ),
            ).to_numpy(),
            "log_mass": x,
            "log_radius": y,
            "fitted_log_radius": fitted_y,
            "distance_y": distance_y,
            "gp_predictive_sigma": gp_std,
            "log_radius_err": yerr,
            "log_mass_err": xerr,
            "local_gp_slope": slope,
            "propagated_log_radius_err": np.abs(propagated),
            "sigma_y_total": sigma_y,
            "two_sigma_y_total": 2.0 * sigma_y,
            "distance_over_sigma": normalized_distance,
            "within_1sigma": within_1sigma,
            "within_2sigma": within_2sigma,
        }
    )

    summary = {
        "n_planets": len(y),
        "n_within_1sigma": int(np.sum(within_1sigma)),
        "fraction_within_1sigma": float(np.mean(within_1sigma)),
        "n_within_2sigma": int(np.sum(within_2sigma)),
        "fraction_within_2sigma": float(np.mean(within_2sigma)),
        "median_distance_over_sigma": float(np.median(normalized_distance)),
    }
    return summary, table


def posterior_predictive_pte(
    df: pd.DataFrame,
    fit: GPFit,
    n_samples: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Approximate GP posterior predictive PTE using chi-square as T."""
    rng = np.random.default_rng(seed)
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    mean, _ = predict_gp(fit, x)
    variance, _, _, _ = total_predictive_variance(fit, x, xerr, yerr)
    t_data_value = float(np.sum((y - mean) ** 2 / variance))

    t_rep = np.empty(n_samples)
    t_data = np.full(n_samples, t_data_value)
    for i in range(n_samples):
        y_rep = rng.normal(mean, np.sqrt(variance))
        t_rep[i] = float(np.sum((y_rep - mean) ** 2 / variance))

    return float(np.mean(t_rep >= t_data)), t_rep, t_data


def predictive_pte_from_mean_variance(
    y: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
    n_samples: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Approximate a predictive PTE from stored means and variances."""
    rng = np.random.default_rng(seed)
    variance = np.maximum(np.asarray(variance), 1e-12)
    y = np.asarray(y)
    mean = np.asarray(mean)
    t_data_value = float(np.sum((y - mean) ** 2 / variance))
    t_data = np.full(n_samples, t_data_value)
    t_rep = np.empty(n_samples)
    for i in range(n_samples):
        y_rep = rng.normal(mean, np.sqrt(variance))
        t_rep[i] = float(np.sum((y_rep - mean) ** 2 / variance))
    return float(np.mean(t_rep >= t_data)), t_rep, t_data


def rmse_with_se(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return RMSE and a delta-method standard error for the RMSE."""
    residual = np.asarray(y_true) - np.asarray(y_pred)
    squared_error = residual**2
    rmse = float(np.sqrt(np.mean(squared_error)))
    if len(squared_error) > 1 and rmse > 0:
        rmse_se = float(
            np.std(squared_error, ddof=1)
            / np.sqrt(len(squared_error))
            / (2.0 * rmse)
        )
    else:
        rmse_se = 0.0
    return rmse, rmse_se


def negative_log_predictive_density(
    y: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> tuple[float, float]:
    """Return mean Gaussian NLPD and its pointwise standard error."""
    variance = np.maximum(np.asarray(variance), 1e-12)
    pointwise = 0.5 * (
        np.log(2.0 * np.pi * variance)
        + (np.asarray(y) - np.asarray(mean)) ** 2 / variance
    )
    nlpd = float(np.mean(pointwise))
    nlpd_se = float(np.std(pointwise, ddof=1) / np.sqrt(len(pointwise))) if len(pointwise) > 1 else 0.0
    return nlpd, nlpd_se


def point_prediction_metrics(y: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute point-prediction metrics for log-radius predictions."""
    rmse, rmse_se = rmse_with_se(y, y_pred)
    return {
        "rmse": rmse,
        "rmse_se": rmse_se,
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


def fit_metrics(df: pd.DataFrame, fit: GPFit) -> dict[str, float]:
    """Compute point-prediction metrics for one dataframe."""
    y = df["log_radius"].to_numpy()
    y_pred, _ = predict_gp(fit, df["log_mass"].to_numpy())
    return point_prediction_metrics(y, y_pred)


def evaluate_prediction_distribution(
    df: pd.DataFrame,
    mean: np.ndarray,
    variance: np.ndarray,
    n_samples: int,
    seed: int,
) -> dict[str, float | int]:
    """Evaluate held-out point predictions and predictive distributions."""
    y = df["log_radius"].to_numpy()
    variance = np.maximum(np.asarray(variance), 1e-12)
    sigma_y = np.sqrt(variance)
    distance_y = np.abs(y - mean)
    within_1sigma = distance_y <= sigma_y
    within_2sigma = distance_y <= 2.0 * sigma_y
    normalized_distance = distance_y / sigma_y
    metrics = point_prediction_metrics(y, mean)
    nlpd, nlpd_se = negative_log_predictive_density(y, mean, variance)
    pte, _, _ = predictive_pte_from_mean_variance(y, mean, variance, n_samples, seed)
    metrics.update(
        {
            "nlpd": nlpd,
            "nlpd_se": nlpd_se,
            "ppd_pte": pte,
            "n_planets": len(y),
            "n_within_1sigma": int(np.sum(within_1sigma)),
            "fraction_within_1sigma": float(np.mean(within_1sigma)),
            "n_within_2sigma": int(np.sum(within_2sigma)),
            "fraction_within_2sigma": float(np.mean(within_2sigma)),
            "median_distance_over_sigma": float(np.median(normalized_distance)),
        }
    )
    return {
        key: float(value) if isinstance(value, np.floating) else value
        for key, value in metrics.items()
    }


def summarize_fit(fit: GPFit, df: pd.DataFrame) -> pd.DataFrame:
    """Save kernel and hyperparameter information."""
    white_variance = kernel_white_noise_variance(fit.model.kernel_)
    rows = [
        {"quantity": "kernel_name", "value": fit.kernel_name},
        {"quantity": "optimized_kernel", "value": str(fit.model.kernel_)},
        {
            "quantity": "log_marginal_likelihood",
            "value": fit.model.log_marginal_likelihood_value_,
        },
        {"quantity": "intrinsic_white_sigma_dex", "value": np.sqrt(white_variance)},
        {"quantity": "n_planets", "value": len(df)},
    ]
    for name, value in fit.model.kernel_.get_params().items():
        if name.endswith("length_scale") or name.endswith("alpha") or name.endswith("noise_level"):
            rows.append({"quantity": name, "value": value})
    return pd.DataFrame(rows)


def information_criteria(fit: GPFit, n_data: int) -> tuple[int, float, float, float]:
    """Return n_params, log marginal likelihood, AIC, and BIC."""
    n_params = len(fit.model.kernel_.theta) + 1
    log_evidence = float(fit.model.log_marginal_likelihood_value_)
    aic = 2.0 * n_params - 2.0 * log_evidence
    bic = n_params * np.log(n_data) - 2.0 * log_evidence
    return n_params, log_evidence, float(aic), float(bic)


def summarize_selected_fit(
    fit: GPFit,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float | int],
    selection_row: dict[str, float | str | int] | None,
    test_size: float,
) -> pd.DataFrame:
    """Save selected-kernel metadata, split counts, and final diagnostics."""
    n_params, log_evidence, aic, bic = information_criteria(fit, len(train_df))
    white_variance = kernel_white_noise_variance(fit.model.kernel_)
    rows = [
        {"quantity": "kernel_name", "value": fit.kernel_name},
        {"quantity": "selection_strategy", "value": "training_stratified_cv_nlpd"},
        {"quantity": "split_strategy", "value": "mass_regime_stratified"},
        {
            "quantity": "mass_regime_breakpoints_log_mass",
            "value": ",".join(str(value) for value in MASS_REGIME_BREAKPOINTS),
        },
        {"quantity": "n_train", "value": len(train_df)},
        {"quantity": "n_test", "value": len(test_df)},
        {"quantity": "test_size", "value": test_size},
        {"quantity": "n_params", "value": n_params},
        {"quantity": "train_log_marginal_likelihood", "value": log_evidence},
        {"quantity": "train_aic", "value": aic},
        {"quantity": "train_bic", "value": bic},
        {"quantity": "optimized_kernel", "value": str(fit.model.kernel_)},
        {"quantity": "intrinsic_white_sigma_dex", "value": np.sqrt(white_variance)},
    ]
    for name, count in regime_counts(train_df).items():
        rows.append({"quantity": f"train_n_{name}", "value": count})
    for name, count in regime_counts(test_df).items():
        rows.append({"quantity": f"test_n_{name}", "value": count})
    for key, value in train_metrics.items():
        rows.append({"quantity": f"train_{key}", "value": value})
    for key, value in test_metrics.items():
        rows.append({"quantity": f"test_{key}", "value": value})
    if selection_row is not None:
        for key, value in selection_row.items():
            if key not in {"optimized_kernel"}:
                rows.append({"quantity": f"selection_{key}", "value": value})
    for name, value in fit.model.kernel_.get_params().items():
        if name.endswith("length_scale") or name.endswith("alpha") or name.endswith("noise_level"):
            rows.append({"quantity": name, "value": value})
    return pd.DataFrame(rows)


def plot_fit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit: GPFit,
    output: Path,
    show: bool = False,
) -> None:
    """Plot GP mass-radius relation and model predictive bands."""
    plot_df = pd.concat([train_df, test_df], ignore_index=True)
    train_x = train_df["log_mass"].to_numpy()
    train_y = train_df["log_radius"].to_numpy()
    test_x = test_df["log_mass"].to_numpy()
    test_y = test_df["log_radius"].to_numpy()

    x_grid = np.linspace(plot_df["log_mass"].min(), plot_df["log_mass"].max(), 500)
    y_grid, std_grid = predict_gp(fit, x_grid)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        train_x,
        train_y,
        xerr=train_df["log_mass_err"].to_numpy(),
        yerr=train_df["log_radius_err"].to_numpy(),
        fmt=".",
        color="0.55",
        alpha=0.35,
        label="training planets",
    )
    ax.errorbar(
        test_x,
        test_y,
        xerr=test_df["log_mass_err"].to_numpy(),
        yerr=test_df["log_radius_err"].to_numpy(),
        fmt=".",
        color="C3",
        alpha=0.75,
        label="test planets",
    )
    ax.fill_between(
        x_grid,
        y_grid - 2.0 * std_grid,
        y_grid + 2.0 * std_grid,
        color="C0",
        alpha=0.14,
        linewidth=0,
        label="95% GP model band",
    )
    ax.fill_between(
        x_grid,
        y_grid - std_grid,
        y_grid + std_grid,
        color="C0",
        alpha=0.28,
        linewidth=0,
        label="68% GP model band",
    )
    ax.plot(x_grid, y_grid, color="black", lw=2, label="GP mean")
    ax.set_xlabel(r"$\log_{10}(M/M_\oplus)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\oplus)$")
    ax.set_title(f"Gaussian-process radius-mass relation: {fit.kernel_name}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_ppd_check(
    t_rep: np.ndarray,
    t_data: np.ndarray,
    pte: float,
    output: Path,
    show: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(t_rep - t_data, bins=35, density=True, alpha=0.75)
    ax.axvline(0.0, color="black", ls=":", lw=1.5)
    ax.set_xlabel(r"$T(y^\mathrm{rep}) - T(y)$")
    ax.set_ylabel("density")
    ax.set_title(f"GP posterior predictive check: PTE = {pte:.3f}")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def prediction_diagnostics_table(
    df: pd.DataFrame,
    mean: np.ndarray,
    variance: np.ndarray,
    gp_std: np.ndarray,
    slope: np.ndarray,
    propagated: np.ndarray,
    kernel_name: str,
    fold: int | None = None,
) -> pd.DataFrame:
    """Build a per-planet prediction table for held-out or CV predictions."""
    variance = np.maximum(np.asarray(variance), 1e-12)
    sigma_y = np.sqrt(variance)
    distance_y = np.abs(df["log_radius"].to_numpy() - mean)
    table = pd.DataFrame(
        {
            "kernel": kernel_name,
            "mass": df["mass"].to_numpy(),
            "radius": df["radius"].to_numpy(),
            "mass_regime": df.get(
                "mass_regime",
                pd.Series(
                    [MASS_REGIME_NAMES[code] for code in mass_regime_codes(df)],
                    index=df.index,
                ),
            ).to_numpy(),
            "log_mass": df["log_mass"].to_numpy(),
            "log_radius": df["log_radius"].to_numpy(),
            "predicted_log_radius": mean,
            "residual_log_radius": df["log_radius"].to_numpy() - mean,
            "distance_y": distance_y,
            "gp_predictive_sigma": gp_std,
            "log_radius_err": df["log_radius_err"].to_numpy(),
            "log_mass_err": df["log_mass_err"].to_numpy(),
            "local_gp_slope": slope,
            "propagated_log_radius_err": np.abs(propagated),
            "sigma_y_total": sigma_y,
            "two_sigma_y_total": 2.0 * sigma_y,
            "distance_over_sigma": distance_y / sigma_y,
            "within_1sigma": distance_y <= sigma_y,
            "within_2sigma": distance_y <= 2.0 * sigma_y,
        }
    )
    if fold is not None:
        table.insert(1, "cv_fold", fold)
    return table


def evaluate_fit_distribution(
    df: pd.DataFrame,
    fit: GPFit,
    n_samples: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Evaluate a fitted GP on a dataframe with full predictive variance."""
    x = df["log_mass"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()
    mean, _ = predict_gp(fit, x)
    variance, gp_std, slope, propagated = total_predictive_variance(fit, x, xerr, yerr)
    metrics = evaluate_prediction_distribution(df, mean, variance, n_samples, seed)
    table = prediction_diagnostics_table(
        df=df,
        mean=mean,
        variance=variance,
        gp_std=gp_std,
        slope=slope,
        propagated=propagated,
        kernel_name=fit.kernel_name,
    )
    return metrics, table


def cross_validate_kernel(
    train_df: pd.DataFrame,
    kernel_name: str,
    cv_folds: int,
    n_restarts: int,
    propagation_iterations: int,
    ppd_samples: int,
    jitter_floor: float,
    seed: int,
) -> tuple[dict[str, float | str | int], pd.DataFrame]:
    """Score one kernel using only stratified CV folds from the training set."""
    fold_tables = []
    fold_log_likelihoods = []
    fold_white_sigmas = []
    fold_n_params = []
    splits = cv_split_indices(train_df, cv_folds, seed)

    for fold_number, (fold_train_idx, validation_idx) in enumerate(
        splits,
        start=1,
    ):
        fold_train_df = train_df.iloc[fold_train_idx].sort_values("mass").reset_index(drop=True)
        validation_df = train_df.iloc[validation_idx].sort_values("mass").reset_index(drop=True)
        fit = fit_gp_model(
            df=fold_train_df,
            kernel_name=kernel_name,
            n_restarts=n_restarts,
            propagation_iterations=propagation_iterations,
            seed=seed + fold_number,
            jitter_floor=jitter_floor,
        )
        x = validation_df["log_mass"].to_numpy()
        xerr = validation_df["log_mass_err"].to_numpy()
        yerr = validation_df["log_radius_err"].to_numpy()
        mean, _ = predict_gp(fit, x)
        variance, gp_std, slope, propagated = total_predictive_variance(fit, x, xerr, yerr)
        fold_tables.append(
            prediction_diagnostics_table(
                df=validation_df,
                mean=mean,
                variance=variance,
                gp_std=gp_std,
                slope=slope,
                propagated=propagated,
                kernel_name=kernel_name,
                fold=fold_number,
            )
        )
        n_params, log_evidence, _, _ = information_criteria(fit, len(fold_train_df))
        fold_log_likelihoods.append(log_evidence)
        fold_white_sigmas.append(float(np.sqrt(kernel_white_noise_variance(fit.model.kernel_))))
        fold_n_params.append(n_params)

    cv_predictions = pd.concat(fold_tables, ignore_index=True)
    metrics = evaluate_prediction_distribution(
        df=cv_predictions,
        mean=cv_predictions["predicted_log_radius"].to_numpy(),
        variance=cv_predictions["sigma_y_total"].to_numpy() ** 2,
        n_samples=ppd_samples,
        seed=seed,
    )
    return (
        {
            "kernel": kernel_name,
            "n_train": len(train_df),
            "cv_folds": len(splits),
            "cv_rmse": metrics["rmse"],
            "cv_rmse_se": metrics["rmse_se"],
            "cv_mae": metrics["mae"],
            "cv_r2": metrics["r2"],
            "cv_nlpd": metrics["nlpd"],
            "cv_nlpd_se": metrics["nlpd_se"],
            "cv_ppd_pte": metrics["ppd_pte"],
            "cv_fraction_within_1sigma": metrics["fraction_within_1sigma"],
            "cv_fraction_within_2sigma": metrics["fraction_within_2sigma"],
            "cv_median_distance_over_sigma": metrics["median_distance_over_sigma"],
            "mean_fold_log_marginal_likelihood": float(np.mean(fold_log_likelihoods)),
            "mean_intrinsic_white_sigma_dex": float(np.mean(fold_white_sigmas)),
            "n_params": int(np.median(fold_n_params)),
        },
        cv_predictions,
    )


def run_analysis(
    raw_df: pd.DataFrame,
    kernel_name: str,
    n_restarts: int,
    propagation_iterations: int,
    ppd_samples: int,
    jitter_floor: float,
    seed: int,
    test_size: float,
    output_prefix: str,
    make_plots: bool,
    show: bool,
    sample_hyperparameters: bool,
    mcmc_walkers: int,
    mcmc_steps: int,
    mcmc_burn: int,
    mcmc_thin: int,
    mcmc_prediction_samples: int,
) -> dict[str, float | str | int]:
    """Run one GP fit with a stratified holdout test set."""
    df = MR.prepare_fit_data(raw_df)
    train_df, test_df = split_train_test(df, test_size=test_size, seed=seed)
    print(f"\n=== GP kernel={kernel_name} ===")
    print(f"Using {len(train_df)} training planets and {len(test_df)} test planets.")
    print(
        "Mass-regime counts: "
        f"train({format_regime_counts(train_df)}), "
        f"test({format_regime_counts(test_df)})"
    )

    fit = fit_gp_model(
        df=train_df,
        kernel_name=kernel_name,
        n_restarts=n_restarts,
        propagation_iterations=propagation_iterations,
        seed=seed,
        jitter_floor=jitter_floor,
    )

    train_metrics = fit_metrics(train_df, fit)
    test_metrics, residual_table = evaluate_fit_distribution(test_df, fit, ppd_samples, seed)
    ppd_pte, t_rep, t_data = posterior_predictive_pte(test_df, fit, ppd_samples, seed)

    n_params, log_evidence, aic, bic = information_criteria(fit, len(train_df))
    white_sigma = float(np.sqrt(kernel_white_noise_variance(fit.model.kernel_)))

    summary_path = PLOT_DIR / f"{output_prefix}_summary.csv"
    residual_path = PLOT_DIR / f"{output_prefix}_residual_coverage.csv"
    summarize_selected_fit(
        fit=fit,
        train_df=train_df,
        test_df=test_df,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        selection_row=None,
        test_size=test_size,
    ).to_csv(summary_path, index=False)
    residual_table.to_csv(residual_path, index=False)

    if make_plots:
        plot_fit(train_df, test_df, fit, PLOT_DIR / f"{output_prefix}_fit.png", show=show)
        plot_ppd_check(t_rep, t_data, ppd_pte, PLOT_DIR / f"{output_prefix}_ppd_pte.png", show=show)

    print(f"Optimized kernel: {fit.model.kernel_}")
    print(f"training log marginal likelihood = {log_evidence:.3f}")
    print(f"Intrinsic white sigma = {white_sigma:.4f} dex")
    print(f"test PPD PTE = {ppd_pte:.3f}")
    print(
        "Residual coverage: "
        f"{test_metrics['fraction_within_1sigma']:.3f} within 1 sigma, "
        f"{test_metrics['fraction_within_2sigma']:.3f} within 2 sigma"
    )

    results = {
        "kernel": kernel_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_regime_counts": format_regime_counts(train_df),
        "test_regime_counts": format_regime_counts(test_df),
        "n_params": n_params,
        "train_log_marginal_likelihood": log_evidence,
        "train_aic": float(aic),
        "train_bic": float(bic),
        "ppd_pte": ppd_pte,
        "intrinsic_white_sigma_dex": white_sigma,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "optimized_kernel": str(fit.model.kernel_),
        "summary_path": str(summary_path),
        "residual_coverage_path": str(residual_path),
    }

    if sample_hyperparameters:
        hyper_result = run_hyperparameter_sampling_analysis(
            train_df=train_df,
            test_df=test_df,
            fit=fit,
            optimized_test_metrics=test_metrics,
            optimized_ppd_pte=ppd_pte,
            ppd_samples=ppd_samples,
            seed=seed,
            mcmc_walkers=mcmc_walkers,
            mcmc_steps=mcmc_steps,
            mcmc_burn=mcmc_burn,
            mcmc_thin=mcmc_thin,
            mcmc_prediction_samples=mcmc_prediction_samples,
            make_plots=make_plots,
            show=show,
        )
        results.update(
            {
                "hyperparameter_acceptance_fraction": hyper_result.acceptance_fraction,
                "hyperparameter_prediction_samples": hyper_result.n_prediction_samples,
                **{
                    f"hyperparameter_test_{key}": value
                    for key, value in hyper_result.metrics.items()
                },
            }
        )

    return results


def run_model_selection(
    raw_df: pd.DataFrame,
    n_restarts: int,
    propagation_iterations: int,
    ppd_samples: int,
    jitter_floor: float,
    seed: int,
    test_size: float,
    cv_folds: int,
    make_plots: bool,
    show: bool,
    sample_hyperparameters: bool,
    mcmc_walkers: int,
    mcmc_steps: int,
    mcmc_burn: int,
    mcmc_thin: int,
    mcmc_prediction_samples: int,
) -> dict[str, float | str | int]:
    """Select a GP kernel by training-set CV and evaluate final test metrics once."""
    df = MR.prepare_fit_data(raw_df)
    train_df, test_df = split_train_test(df, test_size=test_size, seed=seed)
    print(f"\n=== GP kernel selection ===")
    print(f"Using {len(train_df)} training planets and {len(test_df)} untouched test planets.")
    print(
        "Mass-regime counts: "
        f"train({format_regime_counts(train_df)}), "
        f"test({format_regime_counts(test_df)})"
    )

    train_path = PLOT_DIR / "gp_mass_radius_split_train.csv"
    test_path = PLOT_DIR / "gp_mass_radius_split_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    selection_rows = []
    cv_prediction_tables = []
    for kernel_name in KERNEL_NAMES:
        print(f"\nCross-validating kernel={kernel_name}")
        row, cv_predictions = cross_validate_kernel(
            train_df=train_df,
            kernel_name=kernel_name,
            cv_folds=cv_folds,
            n_restarts=n_restarts,
            propagation_iterations=propagation_iterations,
            ppd_samples=ppd_samples,
            jitter_floor=jitter_floor,
            seed=seed,
        )
        selection_rows.append(row)
        cv_prediction_tables.append(cv_predictions)
        print(
            f"CV NLPD = {row['cv_nlpd']:.4f}, "
            f"CV RMSE = {row['cv_rmse']:.4f}, "
            f"CV PTE = {row['cv_ppd_pte']:.3f}"
        )

    comparison = pd.DataFrame(selection_rows)
    comparison = comparison.sort_values(["cv_nlpd", "cv_rmse"]).reset_index(drop=True)
    comparison["delta_cv_nlpd"] = comparison["cv_nlpd"] - comparison["cv_nlpd"].min()
    comparison["selected"] = False
    comparison.loc[0, "selected"] = True
    selected_kernel = str(comparison.loc[0, "kernel"])

    comparison_path = PLOT_DIR / "gp_mass_radius_model_selection.csv"
    legacy_comparison_path = PLOT_DIR / "gp_mass_radius_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)
    comparison.to_csv(legacy_comparison_path, index=False)
    cv_predictions_path = PLOT_DIR / "gp_mass_radius_cv_predictions.csv"
    pd.concat(cv_prediction_tables, ignore_index=True).to_csv(cv_predictions_path, index=False)

    print("\nGP training-set CV model selection")
    print(
        comparison[
            [
                "kernel",
                "cv_nlpd",
                "cv_nlpd_se",
                "cv_rmse",
                "cv_rmse_se",
                "cv_ppd_pte",
                "cv_fraction_within_1sigma",
                "cv_fraction_within_2sigma",
                "mean_intrinsic_white_sigma_dex",
                "selected",
            ]
        ].to_string(index=False)
    )
    print(f"Selected kernel by CV NLPD: {selected_kernel}")
    print(f"Saved selection table to {comparison_path}")

    print(f"\n=== Final GP test evaluation: kernel={selected_kernel} ===")
    fit = fit_gp_model(
        df=train_df,
        kernel_name=selected_kernel,
        n_restarts=n_restarts,
        propagation_iterations=propagation_iterations,
        seed=seed,
        jitter_floor=jitter_floor,
    )
    train_metrics = fit_metrics(train_df, fit)
    test_metrics, residual_table = evaluate_fit_distribution(test_df, fit, ppd_samples, seed)
    ppd_pte, t_rep, t_data = posterior_predictive_pte(test_df, fit, ppd_samples, seed)
    test_metrics["ppd_pte"] = ppd_pte

    output_prefix = "gp_mass_radius_best"
    summary_path = PLOT_DIR / f"{output_prefix}_summary.csv"
    residual_path = PLOT_DIR / f"{output_prefix}_residual_coverage.csv"
    selection_row = comparison.iloc[0].to_dict()
    summarize_selected_fit(
        fit=fit,
        train_df=train_df,
        test_df=test_df,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        selection_row=selection_row,
        test_size=test_size,
    ).to_csv(summary_path, index=False)
    residual_table.to_csv(residual_path, index=False)

    if make_plots:
        plot_fit(train_df, test_df, fit, PLOT_DIR / f"{output_prefix}_fit.png", show=show)
        plot_ppd_check(t_rep, t_data, ppd_pte, PLOT_DIR / f"{output_prefix}_ppd_pte.png", show=show)

    n_params, log_evidence, aic, bic = information_criteria(fit, len(train_df))
    white_sigma = float(np.sqrt(kernel_white_noise_variance(fit.model.kernel_)))
    print(f"Optimized kernel: {fit.model.kernel_}")
    print(f"training log marginal likelihood = {log_evidence:.3f}")
    print(f"Intrinsic white sigma = {white_sigma:.4f} dex")
    print(
        f"Final test RMSE = {test_metrics['rmse']:.4f}, "
        f"MAE = {test_metrics['mae']:.4f}, "
        f"R2 = {test_metrics['r2']:.3f}, "
        f"NLPD = {test_metrics['nlpd']:.4f}, "
        f"PTE = {ppd_pte:.3f}"
    )
    print(
        "Final test coverage: "
        f"{test_metrics['fraction_within_1sigma']:.3f} within 1 sigma, "
        f"{test_metrics['fraction_within_2sigma']:.3f} within 2 sigma"
    )

    results = {
        "kernel": selected_kernel,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_regime_counts": format_regime_counts(train_df),
        "test_regime_counts": format_regime_counts(test_df),
        "n_params": n_params,
        "train_log_marginal_likelihood": log_evidence,
        "train_aic": aic,
        "train_bic": bic,
        "ppd_pte": ppd_pte,
        "intrinsic_white_sigma_dex": white_sigma,
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "optimized_kernel": str(fit.model.kernel_),
        "selection_path": str(comparison_path),
        "cv_predictions_path": str(cv_predictions_path),
        "summary_path": str(summary_path),
        "residual_coverage_path": str(residual_path),
    }

    if sample_hyperparameters:
        hyper_result = run_hyperparameter_sampling_analysis(
            train_df=train_df,
            test_df=test_df,
            fit=fit,
            optimized_test_metrics=test_metrics,
            optimized_ppd_pte=ppd_pte,
            ppd_samples=ppd_samples,
            seed=seed,
            mcmc_walkers=mcmc_walkers,
            mcmc_steps=mcmc_steps,
            mcmc_burn=mcmc_burn,
            mcmc_thin=mcmc_thin,
            mcmc_prediction_samples=mcmc_prediction_samples,
            make_plots=make_plots,
            show=show,
        )
        results.update(
            {
                "hyperparameter_acceptance_fraction": hyper_result.acceptance_fraction,
                "hyperparameter_prediction_samples": hyper_result.n_prediction_samples,
                **{
                    f"hyperparameter_test_{key}": value
                    for key, value in hyper_result.metrics.items()
                },
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=KERNEL_NAMES,
        default="matern32",
        help="GP covariance kernel for a single fit",
    )
    parser.add_argument("--compare", action="store_true", help="compare all GP kernels")
    parser.add_argument("--n-restarts", type=int, default=8, help="kernel optimizer restarts")
    parser.add_argument(
        "--propagation-iterations",
        type=int,
        default=2,
        help="iterations used to update mass-error propagation through the GP slope",
    )
    parser.add_argument("--ppd-samples", type=int, default=1000)
    parser.add_argument("--jitter-floor", type=float, default=1e-6)
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="fraction of planets held out for final stratified test evaluation",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="stratified training-set folds used for GP kernel selection",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="optional random subset for quick smoke tests; default uses all data",
    )
    parser.add_argument(
        "--sample-hyperparameters",
        action="store_true",
        help="sample selected-kernel hyperparameters as a Bayesian robustness check",
    )
    parser.add_argument("--mcmc-walkers", type=int, default=32)
    parser.add_argument("--mcmc-steps", type=int, default=1500)
    parser.add_argument("--mcmc-burn", type=int, default=500)
    parser.add_argument("--mcmc-thin", type=int, default=10)
    parser.add_argument("--mcmc-prediction-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    args = parser.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)
    raw_df = MR.load_dace_data()

    if args.max_points is not None and args.max_points < len(raw_df):
        rng = np.random.default_rng(args.seed)
        raw_df = raw_df.iloc[rng.choice(len(raw_df), size=args.max_points, replace=False)]
        raw_df = raw_df.sort_values("mass").reset_index(drop=True)
        print(f"Using a random subset of {len(raw_df)} planets for this run.")

    if args.compare:
        run_model_selection(
            raw_df=raw_df,
            n_restarts=args.n_restarts,
            propagation_iterations=args.propagation_iterations,
            ppd_samples=args.ppd_samples,
            jitter_floor=args.jitter_floor,
            seed=args.seed,
            test_size=args.test_size,
            cv_folds=args.cv_folds,
            make_plots=True,
            show=args.show,
            sample_hyperparameters=args.sample_hyperparameters,
            mcmc_walkers=args.mcmc_walkers,
            mcmc_steps=args.mcmc_steps,
            mcmc_burn=args.mcmc_burn,
            mcmc_thin=args.mcmc_thin,
            mcmc_prediction_samples=args.mcmc_prediction_samples,
        )
    else:
        run_analysis(
            raw_df=raw_df,
            kernel_name=args.kernel,
            n_restarts=args.n_restarts,
            propagation_iterations=args.propagation_iterations,
            ppd_samples=args.ppd_samples,
            jitter_floor=args.jitter_floor,
            seed=args.seed,
            test_size=args.test_size,
            output_prefix="gp_mass_radius",
            make_plots=True,
            show=args.show,
            sample_hyperparameters=args.sample_hyperparameters,
            mcmc_walkers=args.mcmc_walkers,
            mcmc_steps=args.mcmc_steps,
            mcmc_burn=args.mcmc_burn,
            mcmc_thin=args.mcmc_thin,
            mcmc_prediction_samples=args.mcmc_prediction_samples,
        )

    print(f"\nSaved outputs in {PLOT_DIR}")


if __name__ == "__main__":
    main()
