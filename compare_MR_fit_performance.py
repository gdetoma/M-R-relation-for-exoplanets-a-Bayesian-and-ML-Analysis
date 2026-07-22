"""
Shared held-out comparison for the mass-radius relation.

The script compares three pre-existing analyses on one common split:

1. Bayesian broken-power-law model sampled with emcee.
2. Gaussian Process regression.
3. Random Forest regression.

All methods fit log10(R/R_earth) as a function of log10(M/M_earth). The
shared test set is never used for selecting the number of Bayesian breakpoints,
the GP kernel, the RF configuration, or any continuous hyperparameters.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"
GP_SCRIPT = SCRIPT_DIR / "gp_M-R.py"
RF_SCRIPT = SCRIPT_DIR / "rf_M-R.py"
VARIANCE_FLOOR = 1e-8


def load_local_module(path: Path, module_name: str):
    """Import a local script whose filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MR = load_local_module(EMCEE_SCRIPT, "shared_compare_emcee_mr")
GP = load_local_module(GP_SCRIPT, "shared_compare_gp_mr")
RF = load_local_module(RF_SCRIPT, "shared_compare_rf_mr")


def reset_module_rngs(seed: int) -> None:
    """Reset module-level RNGs used by the imported analysis helpers."""
    np.random.seed(seed)
    if hasattr(MR, "RNG"):
        MR.RNG = np.random.default_rng(seed)
    if hasattr(GP, "RNG"):
        GP.RNG = np.random.default_rng(seed)


def ensure_mass_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with readable mass-regime labels."""
    if "mass_regime" in df.columns:
        return df.copy()
    return RF.add_mass_regime(df)


def make_shared_split(
    raw_df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one reproducible mass-regime-stratified train/test split."""
    df = MR.prepare_fit_data(raw_df).copy()
    df["planet_id_or_index"] = np.arange(len(df), dtype=int)
    df = ensure_mass_regime(df)
    train_df, test_df = RF.split_train_test(df, test_size=test_size, seed=seed)
    return ensure_mass_regime(train_df), ensure_mass_regime(test_df)


def save_split_ids(train_df: pd.DataFrame, test_df: pd.DataFrame, output: Path) -> None:
    """Save the exact held-out split used by all methods."""
    train_ids = train_df.copy()
    train_ids["split"] = "train"
    test_ids = test_df.copy()
    test_ids["split"] = "test"
    split = pd.concat([train_ids, test_ids], ignore_index=True)
    columns = [
        "planet_id_or_index",
        "split",
        "log_mass",
        "log_mass_err",
        "log_radius",
        "log_radius_err",
        "mass_regime",
    ]
    split.loc[:, columns].sort_values("planet_id_or_index").to_csv(output, index=False)


def gaussian_nlpd(y: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> float:
    """Mean Gaussian negative log predictive density in log-radius space."""
    variance = np.maximum(np.asarray(variance, dtype=float), VARIANCE_FLOOR)
    residual = np.asarray(y, dtype=float) - np.asarray(mean, dtype=float)
    pointwise = 0.5 * (np.log(2.0 * np.pi * variance) + residual**2 / variance)
    return float(np.mean(pointwise))


def evaluate_predictions(
    df: pd.DataFrame,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    """Compute shared point and uncertainty-aware metrics."""
    y = df["log_radius"].to_numpy()
    mean = np.asarray(mean, dtype=float)
    variance = np.maximum(np.asarray(variance, dtype=float), VARIANCE_FLOOR)
    residual = y - mean
    sigma = np.sqrt(variance)
    normalized = residual / sigma
    rmse = float(np.sqrt(np.mean(residual**2)))

    return {
        "rmse": rmse,
        "rmse_radius_fraction": float(10.0**rmse - 1.0),
        "nlpd": gaussian_nlpd(y, mean, variance),
        "mae": float(mean_absolute_error(y, mean)),
        "r2": float(r2_score(y, mean)),
        "fraction_within_1sigma": float(np.mean(np.abs(residual) <= sigma)),
        "fraction_within_2sigma": float(np.mean(np.abs(residual) <= 2.0 * sigma)),
        "median_abs_normalized_residual": float(np.median(np.abs(normalized))),
    }


def prediction_table(
    df: pd.DataFrame,
    method: str,
    mean: np.ndarray,
    variance: np.ndarray,
) -> pd.DataFrame:
    """Return one per-planet prediction table with shared columns."""
    variance = np.maximum(np.asarray(variance, dtype=float), VARIANCE_FLOOR)
    sigma = np.sqrt(variance)
    y = df["log_radius"].to_numpy()
    residual = y - mean
    return pd.DataFrame(
        {
            "method": method,
            "planet_id_or_index": df["planet_id_or_index"].to_numpy(),
            "log_mass": df["log_mass"].to_numpy(),
            "log_mass_err": df["log_mass_err"].to_numpy(),
            "log_radius_true": y,
            "log_radius_err": df["log_radius_err"].to_numpy(),
            "log_radius_pred_mean": mean,
            "log_radius_pred_std": sigma,
            "residual": residual,
            "normalized_residual": residual / sigma,
            "mass_regime": df["mass_regime"].to_numpy(),
        }
    )


def comparison_row(
    method: str,
    model_description: str,
    metrics: dict[str, float],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seed: int,
    test_size: float,
    uncertainty_note: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a row for the shared comparison table."""
    row: dict[str, Any] = {
        "method": method,
        "model_description": model_description,
        **metrics,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "seed": seed,
        "test_size": test_size,
        "uncertainty_note": uncertainty_note,
    }
    if extra:
        row.update(extra)
    return row


def select_chain_samples(
    chain: np.ndarray,
    n_samples: int,
    seed: int,
) -> np.ndarray:
    """Return a reproducible subset of posterior samples."""
    if len(chain) == 0:
        raise ValueError("No posterior samples are available for emcee prediction.")
    sample_count = min(n_samples, len(chain))
    rng = np.random.default_rng(seed)
    if sample_count == len(chain):
        return chain
    indices = rng.choice(len(chain), size=sample_count, replace=False)
    return chain[np.sort(indices)]


def moment_matched_emcee_prediction(
    chain: np.ndarray,
    test_df: pd.DataFrame,
    n_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Moment-match the emcee posterior-predictive mixture on test points."""
    samples = select_chain_samples(chain, n_samples=n_samples, seed=seed)
    x = test_df["log_mass"].to_numpy()
    xerr = test_df["log_mass_err"].to_numpy()
    yerr = test_df["log_radius_err"].to_numpy()

    mean_sum = np.zeros(len(test_df), dtype=float)
    second_moment_sum = np.zeros(len(test_df), dtype=float)
    for theta in samples:
        mu = MR.piecewise_log_radius(theta, x)
        variance = MR.total_y_variance(theta, x, xerr, yerr)
        mean_sum += mu
        second_moment_sum += variance + mu**2

    mean = mean_sum / len(samples)
    variance = second_moment_sum / len(samples) - mean**2
    return mean, np.maximum(variance, VARIANCE_FLOOR), len(samples)


def fit_emcee_method(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_breakpoints: int,
    prior: str,
    n_starts: int,
    n_steps: int,
    n_walkers: int,
    prediction_samples: int,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the pre-declared emcee model on train data and score the test set."""
    reset_module_rngs(seed)
    x = train_df["log_mass"].to_numpy()
    y = train_df["log_radius"].to_numpy()
    xerr = train_df["log_mass_err"].to_numpy()
    yerr = train_df["log_radius_err"].to_numpy()
    bounds = MR.make_prior_bounds(x, prior, n_breakpoints)

    map_result = MR.find_map(
        x=x,
        y=y,
        xerr=xerr,
        yerr=yerr,
        bounds=bounds,
        n_starts=n_starts,
        n_breakpoints=n_breakpoints,
    )
    map_theta = np.asarray(map_result.x)
    if MR.emcee is not None:
        chain, _, tau, acceptance = MR.run_emcee_sampler(
            x=x,
            y=y,
            xerr=xerr,
            yerr=yerr,
            bounds=bounds,
            map_theta=map_theta,
            n_walkers=n_walkers,
            n_steps=n_steps,
        )
    else:
        chain, _, tau, acceptance = MR.run_metropolis_sampler(
            x=x,
            y=y,
            xerr=xerr,
            yerr=yerr,
            bounds=bounds,
            map_theta=map_theta,
            n_steps=max(n_steps * n_walkers, 5000),
        )

    mean, variance, n_used = moment_matched_emcee_prediction(
        chain=chain,
        test_df=test_df,
        n_samples=prediction_samples,
        seed=seed,
    )
    method = f"emcee_{n_breakpoints}break"
    metrics = evaluate_predictions(test_df, mean, variance)
    row = comparison_row(
        method=method,
        model_description=(
            f"Pre-declared Bayesian broken-power-law baseline with "
            f"{n_breakpoints} fitted breakpoint positions"
        ),
        metrics=metrics,
        train_df=train_df,
        test_df=test_df,
        seed=seed,
        test_size=float(len(test_df) / (len(train_df) + len(test_df))),
        uncertainty_note=(
            "Posterior-predictive mixture moment-matched to a Gaussian; variance "
            "includes posterior mean-function uncertainty, segment intrinsic scatter, "
            "held-out radius measurement error, and propagated mass error."
        ),
        extra={
            "selection_strategy": "pre_declared_two_breakpoint_baseline",
            "n_breakpoints": n_breakpoints,
            "prior": prior,
            "emcee_steps": n_steps,
            "emcee_walkers": max(n_walkers, 2 * len(map_theta) + 2),
            "emcee_n_starts": n_starts,
            "emcee_acceptance": acceptance,
            "emcee_tau_max": float(np.max(tau)) if tau is not None else np.nan,
            "emcee_prediction_samples": n_used,
            "map_log_posterior": -float(map_result.fun),
        },
    )
    return row, prediction_table(test_df, method, mean, variance)


def select_gp_kernel(
    train_df: pd.DataFrame,
    cv_folds: int,
    n_restarts: int,
    propagation_iterations: int,
    ppd_samples: int,
    jitter_floor: float,
    seed: int,
) -> tuple[str, pd.DataFrame]:
    """Select the GP kernel by training-only CV NLPD."""
    rows = []
    for kernel_name in GP.KERNEL_NAMES:
        result, _ = GP.cross_validate_kernel(
            train_df=train_df,
            kernel_name=kernel_name,
            cv_folds=cv_folds,
            n_restarts=n_restarts,
            propagation_iterations=propagation_iterations,
            ppd_samples=ppd_samples,
            jitter_floor=jitter_floor,
            seed=seed,
        )
        rows.append(result)
    comparison = pd.DataFrame(rows).sort_values(["cv_nlpd", "cv_rmse"]).reset_index(drop=True)
    return str(comparison.iloc[0]["kernel"]), comparison


def fit_gp_method(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    requested_kernel: str,
    cv_folds: int,
    n_restarts: int,
    propagation_iterations: int,
    ppd_samples: int,
    jitter_floor: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the selected or fixed GP on train data and score the test set."""
    reset_module_rngs(seed)
    cv_summary = pd.DataFrame()
    if requested_kernel == "auto":
        kernel_name, cv_summary = select_gp_kernel(
            train_df=train_df,
            cv_folds=cv_folds,
            n_restarts=n_restarts,
            propagation_iterations=propagation_iterations,
            ppd_samples=ppd_samples,
            jitter_floor=jitter_floor,
            seed=seed,
        )
        method = "gp_selected_kernel"
        selection_strategy = "training_cv_nlpd"
        model_description = (
            f"GP with {kernel_name} kernel selected by training-only CV NLPD"
        )
    else:
        kernel_name = requested_kernel
        method = f"gp_{kernel_name}"
        selection_strategy = "fixed_kernel"
        model_description = f"GP with fixed {kernel_name} kernel"

    fit = GP.fit_gp_model(
        df=train_df,
        kernel_name=kernel_name,
        n_restarts=n_restarts,
        propagation_iterations=propagation_iterations,
        seed=seed,
        jitter_floor=jitter_floor,
    )
    x = test_df["log_mass"].to_numpy()
    xerr = test_df["log_mass_err"].to_numpy()
    yerr = test_df["log_radius_err"].to_numpy()
    mean, _ = GP.predict_gp(fit, x)
    variance, gp_std, slope, propagated = GP.total_predictive_variance(fit, x, xerr, yerr)
    metrics = evaluate_predictions(test_df, mean, variance)
    white_sigma = float(np.sqrt(GP.kernel_white_noise_variance(fit.model.kernel_)))

    extra: dict[str, Any] = {
        "selection_strategy": selection_strategy,
        "gp_requested_kernel": requested_kernel,
        "gp_selected_kernel": kernel_name,
        "gp_optimized_kernel": str(fit.model.kernel_),
        "gp_intrinsic_white_sigma_dex": white_sigma,
        "gp_n_restarts": n_restarts,
        "gp_cv_folds": cv_folds if requested_kernel == "auto" else np.nan,
        "gp_propagation_iterations": propagation_iterations,
    }
    if not cv_summary.empty:
        selected_row = cv_summary.iloc[0]
        extra.update(
            {
                "gp_selection_cv_nlpd": float(selected_row["cv_nlpd"]),
                "gp_selection_cv_rmse": float(selected_row["cv_rmse"]),
            }
        )

    row = comparison_row(
        method=method,
        model_description=model_description,
        metrics=metrics,
        train_df=train_df,
        test_df=test_df,
        seed=seed,
        test_size=float(len(test_df) / (len(train_df) + len(test_df))),
        uncertainty_note=(
            "Existing GP total_predictive_variance helper used; scikit-learn "
            "predictive standard deviation handles the fitted WhiteKernel once, "
            "then held-out radius measurement error and propagated mass error are included."
        ),
        extra=extra,
    )
    table = prediction_table(test_df, method, mean, variance)
    table["gp_predictive_std_component"] = gp_std
    table["gp_local_slope"] = slope
    table["gp_propagated_log_radius_err"] = np.abs(propagated)
    return row, table


def rf_config_by_label(label: str) -> dict[str, Any]:
    """Return one named RF configuration from rf_M-R.py."""
    configs = {str(config["label"]): config for config in RF.comparison_configs()}
    if label not in configs:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown RF config '{label}'. Valid labels: {valid}")
    return configs[label]


def fit_rf_method(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config_label: str,
    n_estimators: int,
    max_features: float | int | str | None,
    use_sample_weights: bool,
    error_floor: float,
    seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the pre-declared RF configuration on train data and score the test set."""
    config = rf_config_by_label(config_label)
    fit = RF.fit_random_forest(
        df=train_df,
        n_estimators=n_estimators,
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"]),
        max_features=max_features,
        max_samples=config["max_samples"],
        seed=seed,
        use_sample_weights=use_sample_weights,
        error_floor=error_floor,
        oob_score=True,
        label=config_label,
    )
    x = test_df["log_mass"].to_numpy()
    xerr = test_df["log_mass_err"].to_numpy()
    yerr = test_df["log_radius_err"].to_numpy()
    mean, tree_std = RF.predict_forest(fit, x)

    propagated, slope_proxy = RF.propagate_mass_error(fit, x, xerr)
    variance = tree_std**2 + yerr**2 + propagated**2
    metrics = evaluate_predictions(test_df, mean, variance)
    method = f"rf_{config_label}_proxy_uncertainty"
    row = comparison_row(
        method=method,
        model_description=(
            f"Random Forest {config_label} configuration with raw tree-spread proxy"
        ),
        metrics=metrics,
        train_df=train_df,
        test_df=test_df,
        seed=seed,
        test_size=float(len(test_df) / (len(train_df) + len(test_df))),
        uncertainty_note=(
            "RF NLPD uses raw tree-to-tree scatter as an uncalibrated Gaussian proxy. "
            "It is diagnostic only, not a calibrated probabilistic score. Proxy variance "
            "includes tree spread, held-out radius measurement error, and the existing "
            "deterministic RF finite-difference mass-error propagation helper."
        ),
        extra={
            "selection_strategy": "pre_declared_rf_configuration",
            "rf_config": config_label,
            "rf_n_estimators": n_estimators,
            "rf_max_depth": config["max_depth"],
            "rf_min_samples_leaf": config["min_samples_leaf"],
            "rf_max_features": max_features,
            "rf_max_samples": config["max_samples"],
            "rf_use_sample_weights": use_sample_weights,
            "rf_error_floor": error_floor,
        },
    )
    table = prediction_table(test_df, method, mean, variance)
    table["rf_tree_std_component"] = tree_std
    table["rf_local_slope_proxy"] = slope_proxy
    table["rf_propagated_log_radius_err"] = propagated
    return row, table


def plot_rmse_nlpd(comparison: pd.DataFrame, output: Path) -> None:
    """Save a compact RMSE/NLPD bar plot."""
    labels = comparison["method"].to_numpy()
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    axes[0].bar(x, comparison["rmse"], color="C0", alpha=0.85)
    axes[0].set_ylabel("RMSE [dex]")
    axes[0].set_title("Point prediction")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=35, ha="right")

    colors = ["C1" if not str(method).startswith("rf_") else "0.65" for method in labels]
    hatches = ["" if not str(method).startswith("rf_") else "//" for method in labels]
    bars = axes[1].bar(x, comparison["nlpd"], color=colors, alpha=0.85)
    for bar, hatch, method in zip(bars, hatches, labels):
        bar.set_hatch(hatch)
        if str(method).startswith("rf_"):
            height = bar.get_height()
            max_abs_nlpd = float(np.max(np.abs(comparison["nlpd"].to_numpy(dtype=float))))
            offset = 0.04 * max(1.0, max_abs_nlpd)
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                height + offset if height >= 0 else height - offset,
                "proxy",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8,
            )
    axes[1].set_ylabel("Gaussian NLPD")
    axes[1].set_title("Uncertainty-aware score")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=35, ha="right")

    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def apply_smoke_test_overrides(args: argparse.Namespace) -> None:
    """Make smoke-test runs fast while still exercising every output path."""
    if not args.smoke_test:
        return
    args.emcee_steps = min(args.emcee_steps, 250)
    args.emcee_n_starts = min(args.emcee_n_starts, 4)
    args.n_restarts_gp = min(args.n_restarts_gp, 0)
    args.cv_folds = min(args.cv_folds, 2)
    args.rf_n_estimators = min(args.rf_n_estimators, 80)
    args.ppd_samples = min(args.ppd_samples, 200)
    args.emcee_prediction_samples = min(args.emcee_prediction_samples, 200)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--gp-kernel",
        choices=("auto", "rbf", "matern32", "matern52", "rq"),
        default="auto",
    )
    parser.add_argument("--rf-config", default="smooth_full")
    parser.add_argument("--emcee-breaks", type=int, default=2)
    parser.add_argument("--emcee-prior", choices=("broad", "positive", "wide-scatter"), default="broad")
    parser.add_argument("--emcee-steps", type=int, default=2000)
    parser.add_argument(
        "--emcee-steps-small",
        action="store_true",
        help="shortcut for a shorter emcee run useful for debugging",
    )
    parser.add_argument("--emcee-walkers", type=int, default=32)
    parser.add_argument("--emcee-n-starts", type=int, default=24)
    parser.add_argument("--emcee-prediction-samples", type=int, default=1000)
    parser.add_argument("--n-restarts-gp", type=int, default=3)
    parser.add_argument("--gp-propagation-iterations", type=int, default=2)
    parser.add_argument("--gp-jitter-floor", type=float, default=1e-6)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--ppd-samples", type=int, default=1000)
    parser.add_argument("--rf-n-estimators", type=int, default=600)
    parser.add_argument("--rf-max-features", type=float, default=1.0)
    parser.add_argument("--rf-error-floor", type=float, default=0.01)
    parser.add_argument("--rf-no-sample-weights", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="run a smaller, faster end-to-end check and overwrite the standard outputs",
    )
    parser.add_argument("--smoke-max-points", type=int, default=180)
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.emcee_steps_small:
        args.emcee_steps = min(args.emcee_steps, 500)
    apply_smoke_test_overrides(args)
    reset_module_rngs(args.seed)
    PLOT_DIR.mkdir(exist_ok=True)

    raw_df = MR.load_dace_data()
    if args.smoke_test and args.smoke_max_points < len(raw_df):
        raw_df = raw_df.sample(n=args.smoke_max_points, random_state=args.seed).reset_index(drop=True)
        print(f"Smoke test: using {len(raw_df)} randomly selected cleaned planets.")

    train_df, test_df = make_shared_split(raw_df, test_size=args.test_size, seed=args.seed)
    split_path = PLOT_DIR / "mr_methods_shared_split_ids.csv"
    save_split_ids(train_df, test_df, split_path)
    print(
        "Shared split: "
        f"{len(train_df)} train, {len(test_df)} test; saved IDs to {split_path}"
    )

    rows: list[dict[str, Any]] = []
    prediction_tables: list[pd.DataFrame] = []

    print("\n=== emcee Bayesian broken-power-law ===")
    row, table = fit_emcee_method(
        train_df=train_df,
        test_df=test_df,
        n_breakpoints=args.emcee_breaks,
        prior=args.emcee_prior,
        n_starts=args.emcee_n_starts,
        n_steps=args.emcee_steps,
        n_walkers=args.emcee_walkers,
        prediction_samples=args.emcee_prediction_samples,
        seed=args.seed,
    )
    rows.append(row)
    prediction_tables.append(table)
    print(f"{row['method']}: RMSE={row['rmse']:.4f}, NLPD={row['nlpd']:.4f}")

    print("\n=== Gaussian Process ===")
    row, table = fit_gp_method(
        train_df=train_df,
        test_df=test_df,
        requested_kernel=args.gp_kernel,
        cv_folds=args.cv_folds,
        n_restarts=args.n_restarts_gp,
        propagation_iterations=args.gp_propagation_iterations,
        ppd_samples=args.ppd_samples,
        jitter_floor=args.gp_jitter_floor,
        seed=args.seed,
    )
    rows.append(row)
    prediction_tables.append(table)
    print(f"{row['method']}: RMSE={row['rmse']:.4f}, NLPD={row['nlpd']:.4f}")

    print("\n=== Random Forest ===")
    row, table = fit_rf_method(
        train_df=train_df,
        test_df=test_df,
        config_label=args.rf_config,
        n_estimators=args.rf_n_estimators,
        max_features=args.rf_max_features,
        use_sample_weights=not args.rf_no_sample_weights,
        error_floor=args.rf_error_floor,
        seed=args.seed,
    )
    rows.append(row)
    prediction_tables.append(table)
    print(f"{row['method']}: RMSE={row['rmse']:.4f}, NLPD={row['nlpd']:.4f}")

    comparison = pd.DataFrame(rows)
    comparison_path = PLOT_DIR / "mr_methods_shared_split_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions_path = PLOT_DIR / "mr_methods_shared_split_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    plot_path = PLOT_DIR / "mr_methods_shared_split_rmse_nlpd.png"
    if not args.no_plots:
        plot_rmse_nlpd(comparison, plot_path)

    print("\nShared held-out comparison")
    print(
        comparison[
            [
                "method",
                "rmse",
                "rmse_radius_fraction",
                "nlpd",
                "mae",
                "r2",
                "fraction_within_1sigma",
                "fraction_within_2sigma",
                "median_abs_normalized_residual",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved comparison table to {comparison_path}")
    print(f"Saved test predictions to {predictions_path}")
    if not args.no_plots:
        print(f"Saved RMSE/NLPD plot to {plot_path}")


if __name__ == "__main__":
    main()
