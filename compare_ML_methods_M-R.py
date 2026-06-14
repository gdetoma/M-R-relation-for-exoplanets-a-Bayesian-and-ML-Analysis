"""
Compare machine-learning mass-radius methods on one shared train/test split.

The existing GP and Random Forest scripts use different evaluation conventions:
the GP script reports full-sample diagnostics, while the Random Forest script
uses a held-out test set. This script makes a single reproducible split first,
then trains and evaluates both methods on exactly the same rows.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"
GP_SCRIPT = SCRIPT_DIR / "gp_M-R.py"
RF_SCRIPT = SCRIPT_DIR / "rf_M-R.py"


def load_local_module(path: Path, module_name: str):
    """Import a project script whose filename is not a valid module name."""
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MR = load_local_module(EMCEE_SCRIPT, "compare_ml_emcee_mr")
GP = load_local_module(GP_SCRIPT, "compare_ml_gp_mr")
RF = load_local_module(RF_SCRIPT, "compare_ml_rf_mr")


def make_shared_split(
    raw_df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create one shared train/test split for every ML model."""
    df = MR.prepare_fit_data(raw_df).copy()
    df["split_row_id"] = np.arange(len(df))

    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )
    train_df = df.iloc[train_idx].sort_values("mass").reset_index(drop=True)
    test_df = df.iloc[test_idx].sort_values("mass").reset_index(drop=True)
    return train_df, test_df


def regression_metrics(df: pd.DataFrame, y_pred: np.ndarray) -> dict[str, float]:
    """Common point-prediction metrics in log-radius dex."""
    y = df["log_radius"].to_numpy()
    residual = y - y_pred
    squared_error = residual**2
    rmse = float(np.sqrt(mean_squared_error(y, y_pred)))
    if len(squared_error) > 1 and rmse > 0:
        rmse_se = float(
            np.std(squared_error, ddof=1) / np.sqrt(len(squared_error)) / (2.0 * rmse)
        )
    else:
        rmse_se = 0.0
    return {
        "rmse": rmse,
        "rmse_se": rmse_se,
        "mae": float(mean_absolute_error(y, y_pred)),
        "r2": float(r2_score(y, y_pred)),
    }


def prediction_table(
    df: pd.DataFrame,
    method: str,
    model_label: str,
    y_pred: np.ndarray,
    model_sigma: np.ndarray,
    total_sigma: np.ndarray,
) -> pd.DataFrame:
    """Return a consistent prediction table for one fitted model."""
    return pd.DataFrame(
        {
            "method": method,
            "model_label": model_label,
            "split_row_id": df["split_row_id"].to_numpy(),
            "mass": df["mass"].to_numpy(),
            "radius": df["radius"].to_numpy(),
            "log_mass": df["log_mass"].to_numpy(),
            "log_radius": df["log_radius"].to_numpy(),
            "log_radius_err": df["log_radius_err"].to_numpy(),
            "log_mass_err": df["log_mass_err"].to_numpy(),
            "predicted_log_radius": y_pred,
            "residual_log_radius": df["log_radius"].to_numpy() - y_pred,
            "model_sigma": model_sigma,
            "total_sigma": total_sigma,
        }
    )


def fit_gp_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    kernel_name: str,
    n_restarts: int,
    propagation_iterations: int,
    ppd_samples: int,
    jitter_floor: float,
    seed: int,
) -> tuple[dict[str, object], pd.DataFrame, object]:
    """Fit and evaluate one GP model on the shared split."""
    fit = GP.fit_gp_model(
        df=train_df,
        kernel_name=kernel_name,
        n_restarts=n_restarts,
        propagation_iterations=propagation_iterations,
        seed=seed,
        jitter_floor=jitter_floor,
    )

    train_pred, _ = GP.predict_gp(fit, train_df["log_mass"].to_numpy())
    test_pred, _ = GP.predict_gp(fit, test_df["log_mass"].to_numpy())
    train_metrics = regression_metrics(train_df, train_pred)
    test_metrics = regression_metrics(test_df, test_pred)
    coverage, _ = GP.residual_coverage_check(test_df, fit)
    ppd_pte, _, _ = GP.posterior_predictive_pte(test_df, fit, ppd_samples, seed)

    x = test_df["log_mass"].to_numpy()
    xerr = test_df["log_mass_err"].to_numpy()
    yerr = test_df["log_radius_err"].to_numpy()
    variance, gp_std, _, _ = GP.total_predictive_variance(fit, x, xerr, yerr)

    white_sigma = float(np.sqrt(GP.kernel_white_noise_variance(fit.model.kernel_)))
    result = {
        "method": "Gaussian Process",
        "model_label": kernel_name,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_rmse": train_metrics["rmse"],
        "train_rmse_se": train_metrics["rmse_se"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_rmse_se": test_metrics["rmse_se"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_ppd_pte": ppd_pte,
        "test_fraction_within_1sigma": coverage["fraction_within_1sigma"],
        "test_fraction_within_2sigma": coverage["fraction_within_2sigma"],
        "test_median_distance_over_sigma": coverage["median_distance_over_sigma"],
        "intrinsic_white_sigma_dex": white_sigma,
        "optimized_kernel": str(fit.model.kernel_),
        "n_restarts": n_restarts,
        "propagation_iterations": propagation_iterations,
    }
    predictions = prediction_table(
        test_df,
        "Gaussian Process",
        kernel_name,
        test_pred,
        gp_std,
        np.sqrt(variance),
    )
    return result, predictions, fit


def fit_rf_on_split(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, object],
    n_estimators: int,
    max_features: float | int | str | None,
    seed: int,
    use_sample_weights: bool,
    error_floor: float,
    oob_score: bool,
    ppd_samples: int,
) -> tuple[dict[str, object], pd.DataFrame, object]:
    """Fit and evaluate one Random Forest model on the shared split."""
    label = str(config["label"])
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
        oob_score=oob_score,
        label=label,
    )

    train_metrics = RF.fit_metrics(train_df, fit, include_oob=True)
    test_metrics = RF.fit_metrics(test_df, fit)
    coverage, _ = RF.residual_coverage_check(test_df, fit)
    ppd_pte, _, _ = RF.posterior_predictive_pte(test_df, fit, ppd_samples, seed)

    x = test_df["log_mass"].to_numpy()
    xerr = test_df["log_mass_err"].to_numpy()
    yerr = test_df["log_radius_err"].to_numpy()
    test_pred, _ = RF.predict_forest(fit, x)
    variance, tree_std, _, _ = RF.total_predictive_variance(fit, x, xerr, yerr)

    result = {
        "method": "Random Forest",
        "model_label": label,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_rmse": train_metrics["rmse"],
        "train_rmse_se": train_metrics["rmse_se"],
        "train_mae": train_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "train_oob_rmse": train_metrics["oob_rmse"],
        "train_oob_mae": train_metrics["oob_mae"],
        "train_oob_r2": train_metrics["oob_r2"],
        "test_rmse": test_metrics["rmse"],
        "test_rmse_se": test_metrics["rmse_se"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_ppd_pte": ppd_pte,
        "test_fraction_within_1sigma": coverage["fraction_within_1sigma"],
        "test_fraction_within_2sigma": coverage["fraction_within_2sigma"],
        "test_median_distance_over_sigma": coverage["median_distance_over_sigma"],
        "n_estimators": n_estimators,
        "max_depth": config["max_depth"],
        "min_samples_leaf": config["min_samples_leaf"],
        "max_features": max_features,
        "max_samples": config["max_samples"],
        "use_sample_weights": use_sample_weights,
        "error_floor": error_floor,
    }
    predictions = prediction_table(
        test_df,
        "Random Forest",
        label,
        test_pred,
        tree_std,
        np.sqrt(variance),
    )
    return result, predictions, fit


def plot_metric_comparison(comparison: pd.DataFrame, output: Path) -> None:
    """Plot held-out RMSE and MAE for all compared models."""
    ordered = comparison.sort_values(["test_rmse", "test_mae"]).reset_index(drop=True)
    labels = ordered["method"].str.replace("Gaussian Process", "GP", regex=False)
    labels = labels + "\n" + ordered["model_label"].astype(str)
    x = np.arange(len(ordered))

    fig, ax = plt.subplots(figsize=(max(8.0, 0.75 * len(ordered)), 4.5))
    ax.bar(x - 0.18, ordered["test_rmse"], width=0.36, color="C0", label="test RMSE")
    ax.errorbar(
        x - 0.18,
        ordered["test_rmse"],
        yerr=ordered["test_rmse_se"],
        fmt="none",
        color="black",
        lw=1,
        capsize=3,
    )
    ax.bar(x + 0.18, ordered["test_mae"], width=0.36, color="C2", label="test MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("log-radius error [dex]")
    ax.set_title("Shared-split ML comparison")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def best_models_by_method(comparison: pd.DataFrame) -> pd.DataFrame:
    """Return the best test-RMSE row for each method."""
    ordered = comparison.sort_values(["method", "test_rmse", "test_mae"]).reset_index(drop=True)
    return ordered.groupby("method", as_index=False).first()


def plot_test_predictions(
    predictions: pd.DataFrame,
    comparison: pd.DataFrame,
    output: Path,
) -> None:
    """Plot predicted versus observed test-set radii for the best model per method."""
    best = best_models_by_method(comparison)[["method", "model_label"]]
    plot_df = predictions.merge(best, on=["method", "model_label"], how="inner")

    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    for (method, label), group in plot_df.groupby(["method", "model_label"], sort=False):
        short_method = "GP" if method == "Gaussian Process" else "RF"
        ax.errorbar(
            group["log_radius"],
            group["predicted_log_radius"],
            yerr=group["model_sigma"],
            fmt=".",
            alpha=0.65,
            label=f"{short_method}: {label}",
        )

    min_value = float(min(plot_df["log_radius"].min(), plot_df["predicted_log_radius"].min()))
    max_value = float(max(plot_df["log_radius"].max(), plot_df["predicted_log_radius"].max()))
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


def plot_mass_radius_curves(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    comparison: pd.DataFrame,
    fits: dict[tuple[str, str], object],
    output: Path,
) -> None:
    """Plot best GP and RF curves over the shared train/test data."""
    best = best_models_by_method(comparison)
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    x_grid = np.linspace(all_df["log_mass"].min(), all_df["log_mass"].max(), 500)

    fig, ax = plt.subplots(figsize=(8, 5.4))
    ax.errorbar(
        train_df["log_mass"],
        train_df["log_radius"],
        xerr=train_df["log_mass_err"],
        yerr=train_df["log_radius_err"],
        fmt=".",
        color="0.7",
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
        alpha=0.65,
        label="test planets",
    )

    for _, row in best.iterrows():
        method = row["method"]
        label = str(row["model_label"])
        fit = fits[(method, label)]
        if method == "Gaussian Process":
            y_grid, std_grid = GP.predict_gp(fit, x_grid)
            color = "C0"
            short_method = "GP"
        else:
            y_grid, std_grid = RF.predict_forest(fit, x_grid)
            color = "C2"
            short_method = "RF"
        ax.plot(x_grid, y_grid, color=color, lw=2, label=f"{short_method}: {label}")
        ax.fill_between(
            x_grid,
            y_grid - std_grid,
            y_grid + std_grid,
            color=color,
            alpha=0.14,
            linewidth=0,
        )

    ax.set_xlabel(r"$\log_{10}(M/M_\oplus)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\oplus)$")
    ax.set_title("Best shared-split ML fits")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    plt.close(fig)


def selected_rf_configs(labels: list[str]) -> list[dict[str, object]]:
    """Look up named Random Forest configurations from rf_M-R.py."""
    configs = {str(config["label"]): config for config in RF.comparison_configs()}
    missing = sorted(set(labels) - set(configs))
    if missing:
        valid = ", ".join(sorted(configs))
        raise ValueError(f"Unknown RF config(s): {missing}. Valid labels: {valid}")
    return [configs[label] for label in labels]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gp-kernels",
        nargs="+",
        choices=("rbf", "matern32", "matern52", "rq"),
        default=["rq"],
        help="GP kernels to compare; default is the report's Rational Quadratic model",
    )
    parser.add_argument(
        "--rf-configs",
        nargs="+",
        default=["smooth"],
        help="RF smoothness labels from rf_M-R.py; default is the report's smooth model",
    )
    parser.add_argument("--all-gp-kernels", action="store_true")
    parser.add_argument("--all-rf-configs", action="store_true")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--gp-n-restarts", type=int, default=8)
    parser.add_argument("--gp-propagation-iterations", type=int, default=2)
    parser.add_argument("--gp-jitter-floor", type=float, default=1e-6)
    parser.add_argument("--rf-n-estimators", type=int, default=600)
    parser.add_argument("--rf-max-features", type=float, default=1.0)
    parser.add_argument("--rf-error-floor", type=float, default=0.01)
    parser.add_argument("--rf-no-sample-weights", action="store_true")
    parser.add_argument("--rf-no-oob", action="store_true")
    parser.add_argument("--ppd-samples", type=int, default=1000)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)
    raw_df = MR.load_dace_data()

    if args.max_points is not None and args.max_points < len(raw_df):
        rng = np.random.default_rng(args.seed)
        raw_df = raw_df.iloc[rng.choice(len(raw_df), size=args.max_points, replace=False)]
        raw_df = raw_df.sort_values("mass").reset_index(drop=True)
        print(f"Using a random subset of {len(raw_df)} planets for this run.")

    train_df, test_df = make_shared_split(raw_df, test_size=args.test_size, seed=args.seed)
    train_path = PLOT_DIR / "ml_methods_shared_split_train.csv"
    test_path = PLOT_DIR / "ml_methods_shared_split_test.csv"
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Shared split: {len(train_df)} training planets, {len(test_df)} test planets.")
    print(f"Saved split tables to {train_path} and {test_path}")

    gp_kernels = ["rbf", "matern32", "matern52", "rq"] if args.all_gp_kernels else args.gp_kernels
    rf_labels = (
        [str(config["label"]) for config in RF.comparison_configs()]
        if args.all_rf_configs
        else args.rf_configs
    )
    rf_configs = selected_rf_configs(rf_labels)

    rows: list[dict[str, object]] = []
    prediction_tables: list[pd.DataFrame] = []
    fits: dict[tuple[str, str], object] = {}

    for kernel_name in gp_kernels:
        print(f"\n=== Gaussian Process: {kernel_name} ===")
        result, predictions, fit = fit_gp_on_split(
            train_df=train_df,
            test_df=test_df,
            kernel_name=kernel_name,
            n_restarts=args.gp_n_restarts,
            propagation_iterations=args.gp_propagation_iterations,
            ppd_samples=args.ppd_samples,
            jitter_floor=args.gp_jitter_floor,
            seed=args.seed,
        )
        rows.append(result)
        prediction_tables.append(predictions)
        fits[(result["method"], result["model_label"])] = fit
        print(
            f"test RMSE = {result['test_rmse']:.4f}, "
            f"test MAE = {result['test_mae']:.4f}, "
            f"PTE = {result['test_ppd_pte']:.3f}"
        )

    for config in rf_configs:
        print(f"\n=== Random Forest: {config['label']} ===")
        result, predictions, fit = fit_rf_on_split(
            train_df=train_df,
            test_df=test_df,
            config=config,
            n_estimators=args.rf_n_estimators,
            max_features=args.rf_max_features,
            seed=args.seed,
            use_sample_weights=not args.rf_no_sample_weights,
            error_floor=args.rf_error_floor,
            oob_score=not args.rf_no_oob,
            ppd_samples=args.ppd_samples,
        )
        rows.append(result)
        prediction_tables.append(predictions)
        fits[(result["method"], result["model_label"])] = fit
        print(
            f"test RMSE = {result['test_rmse']:.4f}, "
            f"test MAE = {result['test_mae']:.4f}, "
            f"PTE = {result['test_ppd_pte']:.3f}"
        )

    comparison = pd.DataFrame(rows).sort_values(["test_rmse", "test_mae"]).reset_index(drop=True)
    comparison["test_rmse_rank"] = np.arange(1, len(comparison) + 1)
    comparison_path = PLOT_DIR / "ml_methods_shared_split_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions_path = PLOT_DIR / "ml_methods_shared_split_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    if not args.no_plots:
        plot_metric_comparison(comparison, PLOT_DIR / "ml_methods_shared_split_metrics.png")
        plot_test_predictions(
            predictions,
            comparison,
            PLOT_DIR / "ml_methods_shared_split_predicted_vs_observed.png",
        )
        plot_mass_radius_curves(
            train_df,
            test_df,
            comparison,
            fits,
            PLOT_DIR / "ml_methods_shared_split_fit_curves.png",
        )

    print("\nShared-split ML comparison")
    print(
        comparison[
            [
                "test_rmse_rank",
                "method",
                "model_label",
                "train_rmse",
                "test_rmse",
                "test_rmse_se",
                "test_mae",
                "test_r2",
                "test_ppd_pte",
                "test_fraction_within_1sigma",
                "test_fraction_within_2sigma",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved comparison table to {comparison_path}")
    print(f"Saved test predictions to {predictions_path}")
    if not args.no_plots:
        print(f"Saved plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
