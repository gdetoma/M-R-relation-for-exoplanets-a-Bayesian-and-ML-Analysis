"""
Quantile-boosting and support-vector-regression analysis of the exoplanet M-R relation.

Both methods fit the one-dimensional relation

    log10(R / R_earth) = f(log10(M / M_earth))

using the same cleaned DACE data and the same train/validation/test split.
Quantile boosting directly estimates conditional quantiles of log-radius. SVR
is used as a smooth point-prediction baseline, with a global validation-residual
scatter used as a pragmatic uncertainty calibration.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"


def load_mass_radius_module():
    """Load helpers from emcee_M-R.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("qboost_svr_emcee_mr_helpers", EMCEE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["qboost_svr_emcee_mr_helpers"] = module
    spec.loader.exec_module(module)
    return module


MR = load_mass_radius_module()


def make_shared_splits(
    raw_df: pd.DataFrame,
    test_size: float,
    validation_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits shared by both methods."""
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


def xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return df[["log_mass"]].to_numpy(), df["log_radius"].to_numpy()


def training_weights(df: pd.DataFrame, error_floor: float) -> np.ndarray:
    yerr = np.maximum(df["log_radius_err"].to_numpy(), error_floor)
    weights = 1.0 / yerr**2
    return weights / np.median(weights)


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


def qboost_configs() -> list[dict[str, object]]:
    """Small smoothness grid for one-dimensional quantile boosting."""
    return [
        {
            "label": "very_smooth",
            "n_estimators": 250,
            "learning_rate": 0.035,
            "max_depth": 2,
            "min_samples_leaf": 24,
            "subsample": 0.85,
        },
        {
            "label": "smooth",
            "n_estimators": 350,
            "learning_rate": 0.030,
            "max_depth": 2,
            "min_samples_leaf": 16,
            "subsample": 0.85,
        },
        {
            "label": "medium",
            "n_estimators": 450,
            "learning_rate": 0.025,
            "max_depth": 3,
            "min_samples_leaf": 14,
            "subsample": 0.85,
        },
        {
            "label": "flexible",
            "n_estimators": 550,
            "learning_rate": 0.020,
            "max_depth": 3,
            "min_samples_leaf": 10,
            "subsample": 0.90,
        },
    ]


def make_qboost(alpha: float, config: dict[str, object], seed: int) -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        loss="quantile",
        alpha=alpha,
        n_estimators=int(config["n_estimators"]),
        learning_rate=float(config["learning_rate"]),
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        subsample=float(config["subsample"]),
        random_state=seed,
    )


def select_qboost_config(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    seed: int,
    error_floor: float,
    use_sample_weights: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    x_train, y_train = xy(train_df)
    x_val, y_val = xy(val_df)
    weights = training_weights(train_df, error_floor) if use_sample_weights else None
    rows = []
    for config in qboost_configs():
        model = make_qboost(alpha=0.50, config=config, seed=seed)
        model.fit(x_train, y_train, sample_weight=weights)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        train_metrics = regression_metrics(y_train, train_pred)
        val_metrics = regression_metrics(y_val, val_pred)
        rows.append(
            {
                **config,
                "train_rmse": train_metrics["rmse"],
                "validation_rmse": val_metrics["rmse"],
                "validation_mae": val_metrics["mae"],
                "validation_r2": val_metrics["r2"],
            }
        )
    comparison = pd.DataFrame(rows).sort_values(["validation_rmse", "validation_mae"])
    selected = comparison.iloc[0].to_dict()
    return selected, comparison.reset_index(drop=True)


def fit_qboost_models(
    train_df: pd.DataFrame,
    config: dict[str, object],
    seed: int,
    error_floor: float,
    use_sample_weights: bool,
) -> dict[float, GradientBoostingRegressor]:
    x_train, y_train = xy(train_df)
    weights = training_weights(train_df, error_floor) if use_sample_weights else None
    models = {}
    for alpha in (0.025, 0.16, 0.50, 0.84, 0.975):
        model = make_qboost(alpha=alpha, config=config, seed=seed + int(alpha * 1000))
        model.fit(x_train, y_train, sample_weight=weights)
        models[alpha] = model
    return models


def predict_qboost(models: dict[float, GradientBoostingRegressor], df: pd.DataFrame) -> pd.DataFrame:
    x = df[["log_mass"]].to_numpy()
    raw = np.column_stack([models[alpha].predict(x) for alpha in (0.025, 0.16, 0.50, 0.84, 0.975)])
    ordered = np.sort(raw, axis=1)
    return pd.DataFrame(
        {
            "q025": ordered[:, 0],
            "q16": ordered[:, 1],
            "q50": ordered[:, 2],
            "q84": ordered[:, 3],
            "q975": ordered[:, 4],
        }
    )


def qboost_mean_prediction(models: dict[float, GradientBoostingRegressor], x: np.ndarray) -> np.ndarray:
    return models[0.50].predict(np.asarray(x).reshape(-1, 1))


def svr_param_grid() -> list[dict[str, object]]:
    return [
        {"C": C, "gamma": gamma, "epsilon": epsilon}
        for C in (1.0, 3.0, 10.0, 30.0, 100.0)
        for gamma in ("scale", 0.15, 0.35, 0.70, 1.20)
        for epsilon in (0.015, 0.030, 0.060, 0.090)
    ]


def make_svr(params: dict[str, object]) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svr",
                SVR(
                    kernel="rbf",
                    C=float(params["C"]),
                    gamma=params["gamma"],
                    epsilon=float(params["epsilon"]),
                ),
            ),
        ]
    )


def select_svr_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    error_floor: float,
    use_sample_weights: bool,
) -> tuple[Pipeline, dict[str, object], pd.DataFrame, float]:
    x_train, y_train = xy(train_df)
    x_val, y_val = xy(val_df)
    weights = training_weights(train_df, error_floor) if use_sample_weights else None

    rows = []
    best_model = None
    best_params = None
    best_score = np.inf
    for params in svr_param_grid():
        model = make_svr(params)
        fit_params = {"svr__sample_weight": weights} if weights is not None else {}
        model.fit(x_train, y_train, **fit_params)
        train_pred = model.predict(x_train)
        val_pred = model.predict(x_val)
        train_metrics = regression_metrics(y_train, train_pred)
        val_metrics = regression_metrics(y_val, val_pred)
        n_support = int(np.sum(model.named_steps["svr"].n_support_))
        row = {
            **params,
            "n_support": n_support,
            "train_rmse": train_metrics["rmse"],
            "validation_rmse": val_metrics["rmse"],
            "validation_mae": val_metrics["mae"],
            "validation_r2": val_metrics["r2"],
        }
        rows.append(row)
        if val_metrics["rmse"] < best_score:
            best_score = val_metrics["rmse"]
            best_model = model
            best_params = params

    grid = pd.DataFrame(rows).sort_values(["validation_rmse", "validation_mae"]).reset_index(drop=True)
    val_pred = best_model.predict(x_val)
    calibration_sigma = float(np.sqrt(np.mean((y_val - val_pred) ** 2)))
    return best_model, best_params, grid, calibration_sigma


def finite_difference_slope(predict_fn, x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    step = max(1e-4, 1e-4 * (float(np.max(x)) - float(np.min(x))))
    y_plus = predict_fn(x + step)
    y_minus = predict_fn(x - step)
    return (y_plus - y_minus) / (2.0 * step)


def evaluate_predictions(
    method: str,
    model_label: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_pred: np.ndarray,
    val_pred: np.ndarray,
    test_pred: np.ndarray,
    model_sigma: np.ndarray,
    slope: np.ndarray,
    ppd_samples: int,
    seed: int,
    direct_q16: np.ndarray | None = None,
    direct_q84: np.ndarray | None = None,
    direct_q025: np.ndarray | None = None,
    direct_q975: np.ndarray | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    y_train = train_df["log_radius"].to_numpy()
    y_val = val_df["log_radius"].to_numpy()
    y_test = test_df["log_radius"].to_numpy()
    train_metrics = regression_metrics(y_train, train_pred)
    val_metrics = regression_metrics(y_val, val_pred)
    test_metrics = regression_metrics(y_test, test_pred)

    propagated = np.abs(slope * test_df["log_mass_err"].to_numpy())
    sigma_total = np.sqrt(model_sigma**2 + test_df["log_radius_err"].to_numpy() ** 2 + propagated**2)
    distance = np.abs(y_test - test_pred)
    distance_over_sigma = distance / sigma_total
    within_1sigma = distance <= sigma_total
    within_2sigma = distance <= 2.0 * sigma_total
    ppd_pte = predictive_pte(y_test, test_pred, sigma_total, ppd_samples, seed)

    row = {
        "method": method,
        "model_label": model_label,
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
        "test_ppd_pte": ppd_pte,
        "test_fraction_within_1sigma": float(np.mean(within_1sigma)),
        "test_fraction_within_2sigma": float(np.mean(within_2sigma)),
        "test_median_distance_over_sigma": float(np.median(distance_over_sigma)),
        "test_median_model_sigma": float(np.median(model_sigma)),
        "test_median_total_sigma": float(np.median(sigma_total)),
    }

    if direct_q16 is not None and direct_q84 is not None:
        row["test_direct_68_quantile_coverage"] = float(np.mean((y_test >= direct_q16) & (y_test <= direct_q84)))
    if direct_q025 is not None and direct_q975 is not None:
        row["test_direct_95_quantile_coverage"] = float(
            np.mean((y_test >= direct_q025) & (y_test <= direct_q975))
        )

    table = pd.DataFrame(
        {
            "method": method,
            "model_label": model_label,
            "split_row_id": test_df["split_row_id"].to_numpy(),
            "mass": test_df["mass"].to_numpy(),
            "radius": test_df["radius"].to_numpy(),
            "log_mass": test_df["log_mass"].to_numpy(),
            "log_radius": y_test,
            "log_mass_err": test_df["log_mass_err"].to_numpy(),
            "log_radius_err": test_df["log_radius_err"].to_numpy(),
            "predicted_log_radius": test_pred,
            "residual_log_radius": y_test - test_pred,
            "model_sigma": model_sigma,
            "propagated_log_radius_err": propagated,
            "local_slope": slope,
            "sigma_total": sigma_total,
            "distance_over_sigma": distance_over_sigma,
            "within_1sigma": within_1sigma,
            "within_2sigma": within_2sigma,
        }
    )

    for name, values in (
        ("q025", direct_q025),
        ("q16", direct_q16),
        ("q84", direct_q84),
        ("q975", direct_q975),
    ):
        if values is not None:
            table[name] = values
    return row, table


def plot_fit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    qmodels: dict[float, GradientBoostingRegressor],
    svr_model: Pipeline,
    svr_sigma: float,
    output: Path,
) -> None:
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    x_grid = np.linspace(all_df["log_mass"].min(), all_df["log_mass"].max(), 500)
    grid_df = pd.DataFrame({"log_mass": x_grid})
    qpred = predict_qboost(qmodels, grid_df)
    svr_grid = svr_model.predict(x_grid.reshape(-1, 1))

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
    ax.fill_between(
        x_grid,
        qpred["q025"],
        qpred["q975"],
        color="C0",
        alpha=0.09,
        linewidth=0,
        label="quantile boost 95% band",
    )
    ax.fill_between(
        x_grid,
        qpred["q16"],
        qpred["q84"],
        color="C0",
        alpha=0.18,
        linewidth=0,
        label="quantile boost 68% band",
    )
    ax.plot(x_grid, qpred["q50"], color="C0", lw=2, label="quantile boost median")
    ax.plot(x_grid, svr_grid, color="C3", lw=2, label="SVR mean")
    ax.fill_between(
        x_grid,
        svr_grid - svr_sigma,
        svr_grid + svr_sigma,
        color="C3",
        alpha=0.13,
        linewidth=0,
        label="SVR validation scatter",
    )
    ax.set_xlabel(r"$\log_{10}(M/M_\oplus)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\oplus)$")
    ax.set_title("Quantile boosting and support-vector mass-radius fits")
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument(
        "--validation-size",
        type=float,
        default=0.20,
        help="fraction of the non-test data reserved for model selection",
    )
    parser.add_argument("--error-floor", type=float, default=0.01)
    parser.add_argument("--no-sample-weights", action="store_true")
    parser.add_argument("--ppd-samples", type=int, default=1000)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

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
    train_df.to_csv(PLOT_DIR / "qboost_svr_mass_radius_train.csv", index=False)
    val_df.to_csv(PLOT_DIR / "qboost_svr_mass_radius_validation.csv", index=False)
    test_df.to_csv(PLOT_DIR / "qboost_svr_mass_radius_test.csv", index=False)
    print(
        "Shared split: "
        f"{len(train_df)} training, {len(val_df)} validation, {len(test_df)} test planets."
    )

    use_sample_weights = not args.no_sample_weights

    print("\n=== Quantile boosting ===")
    qconfig, qgrid = select_qboost_config(
        train_df,
        val_df,
        seed=args.seed,
        error_floor=args.error_floor,
        use_sample_weights=use_sample_weights,
    )
    qgrid_path = PLOT_DIR / "qboost_svr_mass_radius_qboost_grid.csv"
    qgrid.to_csv(qgrid_path, index=False)
    qmodels = fit_qboost_models(
        train_df,
        qconfig,
        seed=args.seed,
        error_floor=args.error_floor,
        use_sample_weights=use_sample_weights,
    )
    qtrain = predict_qboost(qmodels, train_df)
    qval = predict_qboost(qmodels, val_df)
    qtest = predict_qboost(qmodels, test_df)
    q_model_sigma = np.maximum(0.5 * (qtest["q84"].to_numpy() - qtest["q16"].to_numpy()), 1e-4)
    q_slope = finite_difference_slope(
        lambda x: qboost_mean_prediction(qmodels, x),
        test_df["log_mass"].to_numpy(),
    )
    qrow, qpred_table = evaluate_predictions(
        method="Quantile Boosting",
        model_label=str(qconfig["label"]),
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_pred=qtrain["q50"].to_numpy(),
        val_pred=qval["q50"].to_numpy(),
        test_pred=qtest["q50"].to_numpy(),
        model_sigma=q_model_sigma,
        slope=q_slope,
        ppd_samples=args.ppd_samples,
        seed=args.seed,
        direct_q16=qtest["q16"].to_numpy(),
        direct_q84=qtest["q84"].to_numpy(),
        direct_q025=qtest["q025"].to_numpy(),
        direct_q975=qtest["q975"].to_numpy(),
    )
    qrow.update({f"qboost_{key}": value for key, value in qconfig.items()})
    qrow["use_sample_weights"] = use_sample_weights
    qrow["error_floor"] = args.error_floor
    print(
        f"selected={qconfig['label']}, test RMSE={qrow['test_rmse']:.4f}, "
        f"PTE={qrow['test_ppd_pte']:.3f}"
    )

    print("\n=== Support vector regression ===")
    svr_model, svr_params, svr_grid, svr_sigma = select_svr_model(
        train_df,
        val_df,
        error_floor=args.error_floor,
        use_sample_weights=use_sample_weights,
    )
    svr_grid_path = PLOT_DIR / "qboost_svr_mass_radius_svr_grid.csv"
    svr_grid.to_csv(svr_grid_path, index=False)
    x_train, _ = xy(train_df)
    x_val, _ = xy(val_df)
    x_test, _ = xy(test_df)
    svr_train = svr_model.predict(x_train)
    svr_val = svr_model.predict(x_val)
    svr_test = svr_model.predict(x_test)
    svr_model_sigma = np.full(len(test_df), svr_sigma)
    svr_slope = finite_difference_slope(
        lambda x: svr_model.predict(np.asarray(x).reshape(-1, 1)),
        test_df["log_mass"].to_numpy(),
    )
    svr_row, svr_pred_table = evaluate_predictions(
        method="Support Vector Regression",
        model_label="rbf_svr",
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_pred=svr_train,
        val_pred=svr_val,
        test_pred=svr_test,
        model_sigma=svr_model_sigma,
        slope=svr_slope,
        ppd_samples=args.ppd_samples,
        seed=args.seed + 1,
    )
    svr_row.update({f"svr_{key}": value for key, value in svr_params.items()})
    svr_row["svr_validation_residual_sigma"] = svr_sigma
    svr_row["use_sample_weights"] = use_sample_weights
    svr_row["error_floor"] = args.error_floor
    svr_row["n_support"] = int(np.sum(svr_model.named_steps["svr"].n_support_))
    print(
        f"selected C={svr_params['C']}, gamma={svr_params['gamma']}, "
        f"epsilon={svr_params['epsilon']}, test RMSE={svr_row['test_rmse']:.4f}, "
        f"PTE={svr_row['test_ppd_pte']:.3f}"
    )

    summary = pd.DataFrame([qrow, svr_row]).sort_values(["test_rmse", "test_mae"]).reset_index(drop=True)
    summary["test_rmse_rank"] = np.arange(1, len(summary) + 1)
    summary_path = PLOT_DIR / "qboost_svr_mass_radius_summary.csv"
    summary.to_csv(summary_path, index=False)

    predictions = pd.concat([qpred_table, svr_pred_table], ignore_index=True)
    predictions_path = PLOT_DIR / "qboost_svr_mass_radius_test_predictions.csv"
    predictions.to_csv(predictions_path, index=False)

    if not args.no_plots:
        plot_fit(
            train_df,
            test_df,
            qmodels,
            svr_model,
            svr_sigma,
            PLOT_DIR / "qboost_svr_mass_radius_fit.png",
        )
        plot_predicted_vs_observed(
            predictions,
            PLOT_DIR / "qboost_svr_mass_radius_predicted_vs_observed.png",
        )

    print("\nQuantile boosting and SVR comparison")
    print(
        summary[
            [
                "test_rmse_rank",
                "method",
                "model_label",
                "train_rmse",
                "validation_rmse",
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
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved test predictions to {predictions_path}")
    print(f"Saved quantile-boosting grid to {qgrid_path}")
    print(f"Saved SVR grid to {svr_grid_path}")
    if not args.no_plots:
        print(f"Saved plots in {PLOT_DIR}")


if __name__ == "__main__":
    main()
