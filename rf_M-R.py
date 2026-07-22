"""
Random-forest analysis of the exoplanet radius-mass relation.

The model is fitted in log space:

    log10(R / R_earth) = f(log10(M / M_earth))

The best-fit relation is the mean prediction across trees. As a practical,
non-Bayesian uncertainty proxy, the script uses the standard deviation of the
individual tree predictions. Residual coverage also includes the measured
radius error and a finite-difference propagation of the measured mass error.
"""

from __future__ import annotations

import argparse
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"
MASS_REGIME_BREAKPOINTS = (0.824, 2.196)
MASS_REGIME_NAMES = ("low_mass", "intermediate_mass", "giant_planet")


def load_mass_radius_module():
    """Load helpers from emcee_M-R.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("emcee_mr_helpers", EMCEE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["emcee_mr_helpers"] = module
    spec.loader.exec_module(module)
    return module


MR = load_mass_radius_module()


@dataclass
class RandomForestFit:
    model: RandomForestRegressor
    use_sample_weights: bool
    error_floor: float
    label: str


def make_forest(
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float | int | str | None,
    max_samples: float | int | None,
    seed: int,
    oob_score: bool,
) -> RandomForestRegressor:
    """Create a random forest for the one-dimensional logM-logR fit."""
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        bootstrap=True,
        oob_score=oob_score,
        random_state=seed,
        n_jobs=-1,
    )


def training_weights(df: pd.DataFrame, error_floor: float) -> np.ndarray:
    """Inverse-variance weights from measured log-radius errors."""
    yerr = np.maximum(df["log_radius_err"].to_numpy(), error_floor)
    weights = 1.0 / yerr**2
    return weights / np.median(weights)


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


def fit_random_forest(
    df: pd.DataFrame,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float | int | str | None,
    max_samples: float | int | None,
    seed: int,
    use_sample_weights: bool,
    error_floor: float,
    oob_score: bool,
    label: str,
) -> RandomForestFit:
    """Fit a random forest to log_radius as a function of log_mass."""
    x = df[["log_mass"]].to_numpy()
    y = df["log_radius"].to_numpy()
    sample_weight = training_weights(df, error_floor) if use_sample_weights else None
    model = make_forest(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        seed=seed,
        oob_score=oob_score,
    )
    model.fit(x, y, sample_weight=sample_weight)
    return RandomForestFit(
        model=model,
        use_sample_weights=use_sample_weights,
        error_floor=error_floor,
        label=label,
    )


def tree_predictions(fit: RandomForestFit, x: np.ndarray) -> np.ndarray:
    """Return predictions from every tree."""
    x2d = np.asarray(x).reshape(-1, 1)
    return np.array([tree.predict(x2d) for tree in fit.model.estimators_])


def predict_forest(fit: RandomForestFit, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean prediction and tree-to-tree standard deviation."""
    per_tree = tree_predictions(fit, x)
    mean = np.mean(per_tree, axis=0)
    std = np.std(per_tree, axis=0, ddof=1)
    return mean, std


def propagate_mass_error(
    fit: RandomForestFit,
    x: np.ndarray,
    xerr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate x uncertainty by shifting each point by its measured x error."""
    x = np.asarray(x)
    xerr = np.asarray(xerr)
    y_plus, _ = predict_forest(fit, x + xerr)
    y_minus, _ = predict_forest(fit, x - xerr)
    propagated = 0.5 * np.abs(y_plus - y_minus)
    slope_proxy = np.zeros_like(x)
    valid = xerr > 0
    slope_proxy[valid] = (y_plus[valid] - y_minus[valid]) / (2.0 * xerr[valid])
    return propagated, slope_proxy


def total_predictive_variance(
    fit: RandomForestFit,
    x: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return total variance and its tree, propagated-x, and slope pieces."""
    _, tree_std = predict_forest(fit, x)
    propagated, slope_proxy = propagate_mass_error(fit, x, xerr)
    variance = tree_std**2 + yerr**2 + propagated**2
    return variance, tree_std, propagated, slope_proxy


def residual_coverage_check(
    df: pd.DataFrame,
    fit: RandomForestFit,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Check residual distances relative to total tree plus measurement sigma."""
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    fitted_y, _ = predict_forest(fit, x)
    variance, tree_std, propagated, slope_proxy = total_predictive_variance(
        fit, x, xerr, yerr
    )
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
            "tree_sigma": tree_std,
            "log_radius_err": yerr,
            "log_mass_err": xerr,
            "local_rf_slope_proxy": slope_proxy,
            "propagated_log_radius_err": propagated,
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
    fit: RandomForestFit,
    n_samples: int,
    seed: int,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Approximate posterior predictive PTE using the RF uncertainty proxy."""
    rng = np.random.default_rng(seed)
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    mean, _ = predict_forest(fit, x)
    variance, _, _, _ = total_predictive_variance(fit, x, xerr, yerr)
    t_data_value = float(np.sum((y - mean) ** 2 / variance))

    t_rep = np.empty(n_samples)
    t_data = np.full(n_samples, t_data_value)
    for i in range(n_samples):
        y_rep = rng.normal(mean, np.sqrt(variance))
        t_rep[i] = float(np.sum((y_rep - mean) ** 2 / variance))

    return float(np.mean(t_rep >= t_data)), t_rep, t_data


def rmse_with_se(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    """Return RMSE and a delta-method standard error for the RMSE."""
    residual = y_true - y_pred
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


def fit_metrics(
    df: pd.DataFrame,
    fit: RandomForestFit,
    include_oob: bool = False,
) -> dict[str, float]:
    """Compute metrics for one dataframe and optional training-set OOB metrics."""
    y = df["log_radius"].to_numpy()
    pred, _ = predict_forest(fit, df["log_mass"].to_numpy())
    residual = y - pred
    rmse, rmse_se = rmse_with_se(y, pred)
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))

    metrics = {
        "rmse": rmse,
        "rmse_se": rmse_se,
        "mae": float(mean_absolute_error(y, pred)),
        "r2": float(r2_score(y, pred)),
        "manual_r2": float(1.0 - ss_res / ss_tot),
        "oob_rmse": np.nan,
        "oob_rmse_se": np.nan,
        "oob_mae": np.nan,
        "oob_r2": np.nan,
        "oob_n": 0.0,
    }

    if include_oob and hasattr(fit.model, "oob_prediction_"):
        oob_pred = np.asarray(fit.model.oob_prediction_)
        valid = np.isfinite(oob_pred)
        if np.any(valid):
            oob_rmse, oob_rmse_se = rmse_with_se(y[valid], oob_pred[valid])
            metrics["oob_rmse"] = oob_rmse
            metrics["oob_rmse_se"] = oob_rmse_se
            metrics["oob_mae"] = float(mean_absolute_error(y[valid], oob_pred[valid]))
            metrics["oob_r2"] = float(r2_score(y[valid], oob_pred[valid]))
            metrics["oob_n"] = float(np.sum(valid))

    return metrics


def summarize_fit(
    fit: RandomForestFit,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_metrics: dict[str, float],
    test_metrics: dict[str, float],
    coverage: dict[str, float | int],
    ppd_pte: float,
    test_size: float,
) -> pd.DataFrame:
    """Save forest hyperparameters and diagnostics."""
    params = fit.model.get_params()
    rows = [
        {"quantity": "label", "value": fit.label},
        {"quantity": "split_strategy", "value": "mass_regime_stratified"},
        {
            "quantity": "mass_regime_breakpoints_log_mass",
            "value": ",".join(str(value) for value in MASS_REGIME_BREAKPOINTS),
        },
        {"quantity": "n_train", "value": len(train_df)},
        {"quantity": "n_test", "value": len(test_df)},
        {"quantity": "test_size", "value": test_size},
        {"quantity": "n_estimators", "value": params["n_estimators"]},
        {"quantity": "max_depth", "value": params["max_depth"]},
        {"quantity": "min_samples_leaf", "value": params["min_samples_leaf"]},
        {"quantity": "max_features", "value": params["max_features"]},
        {"quantity": "max_samples", "value": params["max_samples"]},
        {"quantity": "use_sample_weights", "value": fit.use_sample_weights},
        {"quantity": "error_floor", "value": fit.error_floor},
        {"quantity": "ppd_pte", "value": ppd_pte},
    ]
    for name, count in regime_counts(train_df).items():
        rows.append({"quantity": f"train_n_{name}", "value": count})
    for name, count in regime_counts(test_df).items():
        rows.append({"quantity": f"test_n_{name}", "value": count})
    for key, value in train_metrics.items():
        rows.append({"quantity": f"train_{key}", "value": value})
    for key, value in test_metrics.items():
        rows.append({"quantity": f"test_{key}", "value": value})
    for key, value in coverage.items():
        rows.append({"quantity": f"test_{key}", "value": value})
    return pd.DataFrame(rows)


def plot_fit(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fit: RandomForestFit,
    output: Path,
    show: bool = False,
) -> None:
    """Plot the random-forest M-R relation and tree-spread bands."""
    plot_df = pd.concat([train_df, test_df], ignore_index=True)
    train_x = train_df["log_mass"].to_numpy()
    train_y = train_df["log_radius"].to_numpy()
    test_x = test_df["log_mass"].to_numpy()
    test_y = test_df["log_radius"].to_numpy()

    x_grid = np.linspace(plot_df["log_mass"].min(), plot_df["log_mass"].max(), 500)
    y_grid, tree_std = predict_forest(fit, x_grid)

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
        y_grid - 2.0 * tree_std,
        y_grid + 2.0 * tree_std,
        color="C0",
        alpha=0.14,
        linewidth=0,
        label="95% tree-spread band",
    )
    ax.fill_between(
        x_grid,
        y_grid - tree_std,
        y_grid + tree_std,
        color="C0",
        alpha=0.28,
        linewidth=0,
        label="68% tree-spread band",
    )
    ax.plot(x_grid, y_grid, color="black", lw=2, label="forest mean")
    ax.set_xlabel(r"$\log_{10}(M/M_\oplus)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\oplus)$")
    ax.set_title(f"Random-forest radius-mass relation: {fit.label}")
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
    ax.set_title(f"RF predictive check: PTE = {pte:.3f}")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def run_analysis(
    raw_df: pd.DataFrame,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float | int | str | None,
    max_samples: float | int | None,
    seed: int,
    use_sample_weights: bool,
    error_floor: float,
    oob_score: bool,
    test_size: float,
    ppd_samples: int,
    output_prefix: str,
    label: str,
    make_plots: bool,
    show: bool,
) -> dict[str, float | str | int | None]:
    """Run one complete random-forest fit and diagnostic suite."""
    df = MR.prepare_fit_data(raw_df)
    train_df, test_df = split_train_test(df, test_size=test_size, seed=seed)
    print(f"\n=== Random forest: {label} ===")
    print(f"Using {len(train_df)} training planets and {len(test_df)} test planets.")
    print(
        "Mass-regime counts: "
        f"train({format_regime_counts(train_df)}), "
        f"test({format_regime_counts(test_df)})"
    )

    fit = fit_random_forest(
        df=train_df,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        seed=seed,
        use_sample_weights=use_sample_weights,
        error_floor=error_floor,
        oob_score=oob_score,
        label=label,
    )
    train_metrics = fit_metrics(train_df, fit, include_oob=True)
    test_metrics = fit_metrics(test_df, fit)
    coverage, residual_table = residual_coverage_check(test_df, fit)
    ppd_pte, t_rep, t_data = posterior_predictive_pte(test_df, fit, ppd_samples, seed)

    summary_path = PLOT_DIR / f"{output_prefix}_summary.csv"
    residual_path = PLOT_DIR / f"{output_prefix}_residual_coverage.csv"
    summarize_fit(
        fit,
        train_df,
        test_df,
        train_metrics,
        test_metrics,
        coverage,
        ppd_pte,
        test_size,
    ).to_csv(summary_path, index=False)
    residual_table.to_csv(residual_path, index=False)

    if make_plots:
        plot_fit(train_df, test_df, fit, PLOT_DIR / f"{output_prefix}_fit.png", show=show)
        plot_ppd_check(t_rep, t_data, ppd_pte, PLOT_DIR / f"{output_prefix}_ppd_pte.png", show=show)

    print(
        f"train RMSE = {train_metrics['rmse']:.4f}, "
        f"test RMSE = {test_metrics['rmse']:.4f}, "
        f"OOB RMSE = {train_metrics['oob_rmse']:.4f}, "
        f"PPD PTE = {ppd_pte:.3f}"
    )
    print(
        "Residual coverage: "
        f"{coverage['fraction_within_1sigma']:.3f} within 1 sigma, "
        f"{coverage['fraction_within_2sigma']:.3f} within 2 sigma"
    )

    return {
        "label": label,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "train_regime_counts": format_regime_counts(train_df),
        "test_regime_counts": format_regime_counts(test_df),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "max_samples": max_samples,
        "use_sample_weights": use_sample_weights,
        "error_floor": error_floor,
        "test_size": test_size,
        "ppd_pte": ppd_pte,
        "fraction_within_1sigma": coverage["fraction_within_1sigma"],
        "fraction_within_2sigma": coverage["fraction_within_2sigma"],
        "median_distance_over_sigma": coverage["median_distance_over_sigma"],
        "summary_path": str(summary_path),
        "residual_coverage_path": str(residual_path),
        **{f"train_{key}": value for key, value in train_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
    }


def comparison_configs() -> list[dict[str, object]]:
    """Forest smoothness settings to compare by training-set OOB RMSE."""
    return [
        {"label": "ultra_smooth", "max_depth": 2, "min_samples_leaf": 40, "max_samples": 0.6},
        {"label": "very_smooth", "max_depth": 3, "min_samples_leaf": 30, "max_samples": 0.7},
        {"label": "very_smooth_full", "max_depth": 3, "min_samples_leaf": 30, "max_samples": None},
        {"label": "smooth", "max_depth": 4, "min_samples_leaf": 24, "max_samples": 0.7},
        {"label": "smooth_full", "max_depth": 4, "min_samples_leaf": 24, "max_samples": None},
        {"label": "smooth_leaf18", "max_depth": 4, "min_samples_leaf": 18, "max_samples": 0.7},
        {"label": "smooth_leaf18_full", "max_depth": 4, "min_samples_leaf": 18, "max_samples": None},
        {"label": "medium_smooth", "max_depth": 5, "min_samples_leaf": 18, "max_samples": 0.8},
        {"label": "medium", "max_depth": 5, "min_samples_leaf": 12, "max_samples": 0.8},
        {"label": "medium_full", "max_depth": 5, "min_samples_leaf": 12, "max_samples": None},
        {"label": "balanced", "max_depth": 6, "min_samples_leaf": 8, "max_samples": 0.8},
        {"label": "balanced_full", "max_depth": 6, "min_samples_leaf": 8, "max_samples": None},
        {"label": "deeper", "max_depth": 8, "min_samples_leaf": 6, "max_samples": 0.8},
        {"label": "deep", "max_depth": 10, "min_samples_leaf": 4, "max_samples": 0.8},
        {"label": "deep_full", "max_depth": 10, "min_samples_leaf": 4, "max_samples": None},
        {"label": "flexible", "max_depth": None, "min_samples_leaf": 6, "max_samples": 0.8},
        {"label": "flexible_full", "max_depth": None, "min_samples_leaf": 4, "max_samples": None},
    ]


def parse_max_depth(value: str) -> int | None:
    if value.lower() == "none":
        return None
    return int(value)


def smoothness_sort_columns(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add helper columns that prefer smoother forests."""
    ranked = comparison.copy()
    ranked["_depth_rank"] = ranked["max_depth"].fillna(999).astype(float)
    ranked["_leaf_rank"] = -ranked["min_samples_leaf"].astype(float)
    ranked["_sample_rank"] = ranked["max_samples"].fillna(1.0).astype(float)
    return ranked


def run_oob_candidate(
    train_df: pd.DataFrame,
    n_estimators: int,
    max_depth: int | None,
    min_samples_leaf: int,
    max_features: float | int | str | None,
    max_samples: float | int | None,
    seed: int,
    use_sample_weights: bool,
    error_floor: float,
    label: str,
) -> dict[str, float | str | int | None]:
    """Fit one candidate on training data and report OOB diagnostics only."""
    fit = fit_random_forest(
        df=train_df,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        max_samples=max_samples,
        seed=seed,
        use_sample_weights=use_sample_weights,
        error_floor=error_floor,
        oob_score=True,
        label=label,
    )
    train_metrics = fit_metrics(train_df, fit, include_oob=True)
    return {
        "label": label,
        "n_train": len(train_df),
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_leaf": min_samples_leaf,
        "max_features": max_features,
        "max_samples": max_samples,
        "use_sample_weights": use_sample_weights,
        "error_floor": error_floor,
        **{f"train_{key}": value for key, value in train_metrics.items()},
    }


def select_comparison_model(
    comparison: pd.DataFrame,
    selection_rule: str,
    rmse_tolerance: float,
) -> tuple[pd.Series, float]:
    """Select the reported model from training-set OOB diagnostics."""
    metric = "train_oob_rmse"
    metric_se = "train_oob_rmse_se"
    ordered = comparison.sort_values([metric, "train_oob_mae"]).reset_index(drop=True)
    best = ordered.iloc[0]

    if selection_rule == "best-rmse":
        threshold = float(best[metric])
    elif selection_rule == "one-se":
        threshold = float(best[metric] + best[metric_se])
    elif selection_rule == "rmse-tolerance":
        threshold = float(best[metric] + rmse_tolerance)
    else:
        raise ValueError(f"Unknown selection rule: {selection_rule}")

    candidates = ordered[ordered[metric] <= threshold]
    if selection_rule == "best-rmse":
        selected = candidates.iloc[0]
    else:
        selected = (
            smoothness_sort_columns(candidates)
            .sort_values(
                [
                    "_depth_rank",
                    "_leaf_rank",
                    "_sample_rank",
                    metric,
                    "train_oob_mae",
                ]
            )
            .iloc[0]
        )
    return selected, threshold


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--max-depth", type=parse_max_depth, default=6)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=float, default=1.0)
    parser.add_argument("--max-samples", type=float, default=None)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--no-sample-weights", action="store_true")
    parser.add_argument("--error-floor", type=float, default=0.01)
    parser.add_argument("--no-oob", action="store_true")
    parser.add_argument("--ppd-samples", type=int, default=1000)
    parser.add_argument("--compare", action="store_true", help="compare a few forest smoothness settings")
    parser.add_argument(
        "--selection-rule",
        choices=("one-se", "best-rmse", "rmse-tolerance"),
        default="one-se",
        help="model-selection rule used on training-set OOB RMSE",
    )
    parser.add_argument(
        "--rmse-tolerance",
        type=float,
        default=0.003,
        help="extra RMSE allowed when --selection-rule=rmse-tolerance",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=None,
        help="optional random subset for quick smoke tests; default uses all data",
    )
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

    common = {
        "raw_df": raw_df,
        "n_estimators": args.n_estimators,
        "max_features": args.max_features,
        "seed": args.seed,
        "use_sample_weights": not args.no_sample_weights,
        "error_floor": args.error_floor,
        "oob_score": not args.no_oob,
        "test_size": args.test_size,
        "ppd_samples": args.ppd_samples,
    }

    if args.compare:
        if args.no_oob:
            raise ValueError("--compare requires OOB scoring; remove --no-oob.")

        df = MR.prepare_fit_data(raw_df)
        train_df, test_df = split_train_test(df, test_size=args.test_size, seed=args.seed)
        print(
            "OOB model selection uses only the stratified training split: "
            f"{len(train_df)} training planets, {len(test_df)} untouched test planets."
        )
        print(
            "Mass-regime counts: "
            f"train({format_regime_counts(train_df)}), "
            f"test({format_regime_counts(test_df)})"
        )

        rows = []
        for config in comparison_configs():
            rows.append(
                run_oob_candidate(
                    train_df=train_df,
                    n_estimators=args.n_estimators,
                    max_depth=config["max_depth"],
                    min_samples_leaf=config["min_samples_leaf"],
                    max_features=args.max_features,
                    max_samples=config["max_samples"],
                    seed=args.seed,
                    use_sample_weights=not args.no_sample_weights,
                    error_floor=args.error_floor,
                    label=config["label"],
                )
            )

        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values(["train_oob_rmse", "train_oob_mae"]).reset_index(drop=True)
        selected, threshold = select_comparison_model(
            comparison,
            selection_rule=args.selection_rule,
            rmse_tolerance=args.rmse_tolerance,
        )
        comparison["selected"] = comparison["label"].eq(selected["label"])
        comparison_path = PLOT_DIR / "rf_mass_radius_model_comparison.csv"
        selection_path = PLOT_DIR / "rf_mass_radius_model_selection.csv"
        comparison.to_csv(comparison_path, index=False)
        comparison.to_csv(selection_path, index=False)
        best = comparison.iloc[0]
        best_config = next(
            config for config in comparison_configs() if config["label"] == selected["label"]
        )
        print("\nRandom-forest model comparison")
        print(
            comparison[
                [
                    "label",
                    "max_depth",
                    "min_samples_leaf",
                    "max_samples",
                    "train_rmse",
                    "train_oob_rmse",
                    "train_oob_rmse_se",
                    "train_oob_r2",
                    "selected",
                ]
            ].to_string(index=False)
        )
        print(
            "Best training-set OOB RMSE: "
            f"{best['label']} with OOB_RMSE = {best['train_oob_rmse']:.4f}"
        )
        print(
            "Selected model: "
            f"{selected['label']} using {args.selection_rule} "
            f"(OOB_RMSE <= {threshold:.4f})"
        )
        print(f"Saved OOB model-selection tables to {comparison_path} and {selection_path}")
        print("Evaluating the selected model once on the untouched test set.")
        run_analysis(
            **common,
            max_depth=best_config["max_depth"],
            min_samples_leaf=best_config["min_samples_leaf"],
            max_samples=best_config["max_samples"],
            output_prefix="rf_mass_radius_best",
            label=f"selected_{best_config['label']}",
            make_plots=True,
            show=False,
        )
    else:
        run_analysis(
            **common,
            max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf,
            max_samples=args.max_samples,
            output_prefix="rf_mass_radius",
            label="single",
            make_plots=True,
            show=args.show,
        )

    print(f"\nSaved outputs in {PLOT_DIR}")


if __name__ == "__main__":
    main()
