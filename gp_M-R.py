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
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    RBF,
    RationalQuadratic,
    WhiteKernel,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"
RNG = np.random.default_rng(42)


def load_mass_radius_module():
    """Load helpers from emcee_M-R.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("emcee_mr_helpers", EMCEE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["emcee_mr_helpers"] = module
    spec.loader.exec_module(module)
    return module


MR = load_mass_radius_module()


@dataclass
class GPFit:
    model: GaussianProcessRegressor
    y_mean: float
    training_alpha: np.ndarray
    training_slope: np.ndarray
    kernel_name: str


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


def fit_metrics(df: pd.DataFrame, fit: GPFit) -> dict[str, float]:
    """Compute descriptive full-sample residual metrics."""
    y = df["log_radius"].to_numpy()
    y_pred, _ = predict_gp(fit, df["log_mass"].to_numpy())
    residual = y - y_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - ss_res / ss_tot),
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


def plot_fit(df: pd.DataFrame, fit: GPFit, output: Path, show: bool = False) -> None:
    """Plot GP mass-radius relation and predictive bands."""
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    x_grid = np.linspace(x.min(), x.max(), 500)
    y_grid, std_grid = predict_gp(fit, x_grid)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=".", alpha=0.55, label="DACE planets")
    ax.fill_between(
        x_grid,
        y_grid - 2.0 * std_grid,
        y_grid + 2.0 * std_grid,
        color="C0",
        alpha=0.14,
        linewidth=0,
        label="95% GP predictive band",
    )
    ax.fill_between(
        x_grid,
        y_grid - std_grid,
        y_grid + std_grid,
        color="C0",
        alpha=0.28,
        linewidth=0,
        label="68% GP predictive band",
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


def run_analysis(
    raw_df: pd.DataFrame,
    kernel_name: str,
    n_restarts: int,
    propagation_iterations: int,
    ppd_samples: int,
    jitter_floor: float,
    seed: int,
    output_prefix: str,
    make_plots: bool,
    show: bool,
) -> dict[str, float | str | int]:
    """Run one GP fit and diagnostic suite."""
    df = MR.prepare_fit_data(raw_df)
    print(f"\n=== GP kernel={kernel_name} ===")
    print(f"Using {len(df)} planets.")

    fit = fit_gp_model(
        df=df,
        kernel_name=kernel_name,
        n_restarts=n_restarts,
        propagation_iterations=propagation_iterations,
        seed=seed,
        jitter_floor=jitter_floor,
    )

    metrics = fit_metrics(df, fit)
    coverage, residual_table = residual_coverage_check(df, fit)
    ppd_pte, t_rep, t_data = posterior_predictive_pte(df, fit, ppd_samples, seed)

    n_params = len(fit.model.kernel_.theta) + 1
    n_data = len(df)
    log_evidence = float(fit.model.log_marginal_likelihood_value_)
    aic = 2.0 * n_params - 2.0 * log_evidence
    bic = n_params * np.log(n_data) - 2.0 * log_evidence
    white_sigma = float(np.sqrt(kernel_white_noise_variance(fit.model.kernel_)))

    summary_path = PLOT_DIR / f"{output_prefix}_summary.csv"
    residual_path = PLOT_DIR / f"{output_prefix}_residual_coverage.csv"
    summarize_fit(fit, df).to_csv(summary_path, index=False)
    residual_table.to_csv(residual_path, index=False)

    if make_plots:
        plot_fit(df, fit, PLOT_DIR / f"{output_prefix}_fit.png", show=show)
        plot_ppd_check(t_rep, t_data, ppd_pte, PLOT_DIR / f"{output_prefix}_ppd_pte.png", show=show)

    print(f"Optimized kernel: {fit.model.kernel_}")
    print(f"log marginal likelihood = {log_evidence:.3f}")
    print(f"Intrinsic white sigma = {white_sigma:.4f} dex")
    print(f"PPD PTE = {ppd_pte:.3f}")
    print(
        "Residual coverage: "
        f"{coverage['fraction_within_1sigma']:.3f} within 1 sigma, "
        f"{coverage['fraction_within_2sigma']:.3f} within 2 sigma"
    )

    return {
        "kernel": kernel_name,
        "n_planets": n_data,
        "n_params": n_params,
        "log_marginal_likelihood": log_evidence,
        "aic": float(aic),
        "bic": float(bic),
        "ppd_pte": ppd_pte,
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "r2": metrics["r2"],
        "intrinsic_white_sigma_dex": white_sigma,
        "fraction_within_1sigma": coverage["fraction_within_1sigma"],
        "fraction_within_2sigma": coverage["fraction_within_2sigma"],
        "median_distance_over_sigma": coverage["median_distance_over_sigma"],
        "optimized_kernel": str(fit.model.kernel_),
        "summary_path": str(summary_path),
        "residual_coverage_path": str(residual_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--kernel",
        choices=("rbf", "matern32", "matern52", "rq"),
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

    if args.compare:
        rows = []
        for kernel_name in ("rbf", "matern32", "matern52", "rq"):
            prefix = f"gp_mass_radius_{kernel_name}"
            rows.append(
                run_analysis(
                    raw_df=raw_df,
                    kernel_name=kernel_name,
                    n_restarts=args.n_restarts,
                    propagation_iterations=args.propagation_iterations,
                    ppd_samples=args.ppd_samples,
                    jitter_floor=args.jitter_floor,
                    seed=args.seed,
                    output_prefix=prefix,
                    make_plots=True,
                    show=False,
                )
            )

        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values("log_marginal_likelihood", ascending=False)
        comparison["delta_log_evidence"] = (
            comparison["log_marginal_likelihood"] - comparison["log_marginal_likelihood"].max()
        )
        comparison["delta_bic"] = comparison["bic"] - comparison["bic"].min()
        comparison_path = PLOT_DIR / "gp_mass_radius_model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print("\nGP model comparison")
        print(
            comparison[
                [
                    "kernel",
                    "log_marginal_likelihood",
                    "delta_log_evidence",
                    "bic",
                    "delta_bic",
                    "ppd_pte",
                    "fraction_within_1sigma",
                    "fraction_within_2sigma",
                    "intrinsic_white_sigma_dex",
                ]
            ].to_string(index=False)
        )
        print(f"Saved comparison table to {comparison_path}")
    else:
        run_analysis(
            raw_df=raw_df,
            kernel_name=args.kernel,
            n_restarts=args.n_restarts,
            propagation_iterations=args.propagation_iterations,
            ppd_samples=args.ppd_samples,
            jitter_floor=args.jitter_floor,
            seed=args.seed,
            output_prefix="gp_mass_radius",
            make_plots=True,
            show=args.show,
        )

    print(f"\nSaved outputs in {PLOT_DIR}")


if __name__ == "__main__":
    main()
