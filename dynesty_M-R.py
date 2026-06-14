"""
Nested-sampling analysis of the exoplanet radius-mass relation.

This script fits

    log10(R / R_earth) = f(log10(M / M_earth))

with a continuous piecewise-linear relation, one intrinsic scatter per segment,
and the same propagated mass-error term used in emcee_M-R.py. It compares
models with different numbers of breakpoints using the nested-sampling evidence.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import dynesty
    from dynesty.utils import resample_equal
except ImportError:
    dynesty = None
    resample_equal = None


SCRIPT_DIR = Path(__file__).resolve().parent
PLOT_DIR = SCRIPT_DIR / "plots"
EMCEE_SCRIPT = SCRIPT_DIR / "emcee_M-R.py"


def load_mass_radius_module():
    """Load helpers from emcee_M-R.py despite the hyphen in its filename."""
    spec = importlib.util.spec_from_file_location("emcee_mr_helpers", EMCEE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["emcee_mr_helpers"] = module
    spec.loader.exec_module(module)
    return module


MR = load_mass_radius_module()
RNG = np.random.default_rng(42)


def prior_ranges(prior: str) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Return intercept, slope, and log-scatter prior ranges."""
    intercept_bounds = (-2.5, 3.0)
    if prior == "broad":
        slope_bounds = (-2.0, 3.0)
        sigma_bounds = (np.log(1e-3), np.log(1.0))
    elif prior == "positive":
        slope_bounds = (0.0, 2.0)
        sigma_bounds = (np.log(1e-3), np.log(1.0))
    elif prior == "wide-scatter":
        slope_bounds = (-2.0, 3.0)
        sigma_bounds = (np.log(1e-3), 0.5)
    else:
        raise ValueError(f"Unknown prior preset: {prior}")
    return intercept_bounds, slope_bounds, sigma_bounds


def make_prior_transform(x: np.ndarray, prior: str, n_breakpoints: int):
    """Create a dynesty prior transform for a breakpoint model.

    Breakpoints are uniform over ordered positions between the 5th and 95th
    log-mass percentiles, with a small minimum gap to keep regimes distinct.
    """
    n_segments = n_breakpoints + 1
    intercept_bounds, slope_bounds, sigma_bounds = prior_ranges(prior)
    break_low, break_high = np.quantile(x, [0.05, 0.95])
    min_break_gap = 0.03

    def stretch(u: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        low, high = bounds
        return low + (high - low) * u

    def prior_transform(unit_cube: np.ndarray) -> np.ndarray:
        theta = np.empty(1 + n_segments + n_breakpoints + n_segments)
        cursor = 0

        theta[0] = stretch(unit_cube[cursor], intercept_bounds)
        cursor += 1

        theta[1 : 1 + n_segments] = stretch(
            unit_cube[cursor : cursor + n_segments], slope_bounds
        )
        cursor += n_segments

        if n_breakpoints:
            break_span = break_high - break_low - min_break_gap * (n_breakpoints - 1)
            if break_span <= 0:
                raise ValueError("Breakpoint prior range is too small for the requested model.")
            raw_breaks = stretch(
                unit_cube[cursor : cursor + n_breakpoints],
                (0.0, break_span),
            )
            theta[1 + n_segments : 1 + n_segments + n_breakpoints] = np.sort(raw_breaks)
            theta[1 + n_segments : 1 + n_segments + n_breakpoints] += (
                break_low + min_break_gap * np.arange(n_breakpoints)
            )
            cursor += n_breakpoints

        theta[-n_segments:] = stretch(unit_cube[cursor : cursor + n_segments], sigma_bounds)
        return theta

    return prior_transform


def equal_weight_posterior_samples(results, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return normalized nested-sampling weights and equal-weight samples."""
    weights = np.exp(results.logwt - results.logz[-1])
    weights = weights / np.sum(weights)
    samples = resample_equal(results.samples, weights, rstate=np.random.default_rng(seed))
    return samples, weights


def make_log_likelihood(
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
):
    """Create a dynesty log-likelihood callback."""

    def log_likelihood(theta: np.ndarray) -> float:
        return MR.log_likelihood(theta, x, y, xerr, yerr)

    return log_likelihood


def run_nested_model(
    df: pd.DataFrame,
    prior: str,
    n_breakpoints: int,
    nlive: int,
    dlogz: float,
    sample: str,
    bound: str,
    seed: int,
    maxiter: int | None = None,
    maxcall: int | None = None,
    print_progress: bool = True,
) -> dict[str, object]:
    """Run dynesty for one breakpoint model."""
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    ndim = 1 + (n_breakpoints + 1) + n_breakpoints + (n_breakpoints + 1)
    log_likelihood = make_log_likelihood(x, y, xerr, yerr)
    prior_transform = make_prior_transform(x, prior, n_breakpoints)

    sampler = dynesty.NestedSampler(
        log_likelihood,
        prior_transform,
        ndim,
        nlive=nlive,
        bound=bound,
        sample=sample,
        rstate=np.random.default_rng(seed + n_breakpoints),
    )

    run_kwargs = {"dlogz": dlogz, "print_progress": print_progress}
    if maxiter is not None:
        run_kwargs["maxiter"] = maxiter
    if maxcall is not None:
        run_kwargs["maxcall"] = maxcall
    sampler.run_nested(**run_kwargs)

    results = sampler.results
    logz = float(results.logz[-1])
    logzerr = float(results.logzerr[-1])
    posterior_samples, weights = equal_weight_posterior_samples(results, seed)
    best_idx = int(np.argmax(results.logl))
    best_theta = np.asarray(results.samples[best_idx])
    best_log_likelihood = float(results.logl[best_idx])

    ppd_pte, _, _ = MR.posterior_predictive_pte(
        posterior_samples, x, y, xerr, yerr, n_samples=min(1000, len(posterior_samples))
    )
    coverage, residual_table = MR.residual_coverage_check(df, best_theta, x, y, xerr, yerr)

    return {
        "n_breakpoints": n_breakpoints,
        "n_segments": n_breakpoints + 1,
        "prior": prior,
        "ndim": ndim,
        "nlive": nlive,
        "target_dlogz": dlogz,
        "sample": sample,
        "bound": bound,
        "maxiter": maxiter,
        "maxcall": maxcall,
        "niter": len(results.logl),
        "ncall": int(np.sum(results.ncall)),
        "logz": logz,
        "logzerr": logzerr,
        "best_log_likelihood": best_log_likelihood,
        "best_theta": best_theta,
        "posterior_samples": posterior_samples,
        "weights": weights,
        "results": results,
        "ppd_pte": ppd_pte,
        "fraction_within_1sigma": coverage["fraction_within_1sigma"],
        "fraction_within_2sigma": coverage["fraction_within_2sigma"],
        "median_distance_over_sigma": coverage["median_distance_over_sigma"],
        "residual_table": residual_table,
    }


def summarize_posterior(samples: np.ndarray, n_breakpoints: int) -> pd.DataFrame:
    """Summarize equal-weight posterior samples."""
    rows = []
    names = MR.parameter_names(n_breakpoints)
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(samples[:, i], [16, 50, 84])
        rows.append(
            {
                "parameter": name,
                "median": q50,
                "minus_1sigma": q50 - q16,
                "plus_1sigma": q84 - q50,
            }
        )
    return pd.DataFrame(rows)


def posterior_mean_curve(samples: np.ndarray, x_grid: np.ndarray, n_draws: int = 600) -> np.ndarray:
    """Draw posterior mean-relation curves."""
    count = min(n_draws, len(samples))
    indices = RNG.choice(len(samples), size=count, replace=False)
    return np.array([MR.piecewise_log_radius(theta, x_grid) for theta in samples[indices]])


def posterior_predictive_curves(samples: np.ndarray, x_grid: np.ndarray, n_draws: int = 600) -> np.ndarray:
    """Draw posterior predictive curves including intrinsic scatter."""
    count = min(n_draws, len(samples))
    indices = RNG.choice(len(samples), size=count, replace=False)
    draws = []
    for theta in samples[indices]:
        mu = MR.piecewise_log_radius(theta, x_grid)
        sigma = MR.piecewise_intrinsic_scatter(theta, x_grid)
        draws.append(RNG.normal(mu, sigma))
    return np.array(draws)


def breakpoint_intervals(samples: np.ndarray) -> np.ndarray:
    """Return 16th, 50th, and 84th percentiles for each breakpoint."""
    n_segments = MR.n_segments_from_theta(samples[0])
    n_breakpoints = n_segments - 1
    if n_breakpoints == 0:
        return np.empty((3, 0))
    break_start = 1 + n_segments
    break_samples = samples[:, break_start : break_start + n_breakpoints]
    return np.percentile(break_samples, [16, 50, 84], axis=0)


def plot_nested_fit(
    df: pd.DataFrame,
    samples: np.ndarray,
    best_theta: np.ndarray,
    output: Path,
    show: bool = False,
) -> None:
    """Plot nested-sampling fit with mean and predictive bands."""
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()
    x_grid = np.linspace(x.min(), x.max(), 400)

    mean_draws = posterior_mean_curve(samples, x_grid)
    mean_q025, mean_q16, mean_q50, mean_q84, mean_q975 = np.percentile(
        mean_draws, [2.5, 16, 50, 84, 97.5], axis=0
    )
    predictive_draws = posterior_predictive_curves(samples, x_grid)
    pred_q025, pred_q16, pred_q84, pred_q975 = np.percentile(
        predictive_draws, [2.5, 16, 84, 97.5], axis=0
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=".", alpha=0.55, label="DACE planets")
    ax.fill_between(
        x_grid,
        pred_q025,
        pred_q975,
        color="C2",
        alpha=0.12,
        label="95% intrinsic predictive band",
    )
    ax.fill_between(
        x_grid,
        pred_q16,
        pred_q84,
        color="C2",
        alpha=0.22,
        label="68% intrinsic predictive band",
    )
    ax.fill_between(x_grid, mean_q025, mean_q975, color="C0", alpha=0.12, label="95% mean band")
    ax.fill_between(x_grid, mean_q16, mean_q84, color="C0", alpha=0.25, label="68% mean band")
    ax.plot(x_grid, mean_q50, color="black", lw=2, label="posterior median")
    ax.plot(x_grid, MR.piecewise_log_radius(best_theta, x_grid), color="C3", lw=2, label="best logL")

    breakpoint_q16, breakpoint_q50, breakpoint_q84 = breakpoint_intervals(samples)
    breakpoint_colors = [f"C{i}" for i in range(4, 10)]
    for i, (q16, q50, q84) in enumerate(zip(breakpoint_q16, breakpoint_q50, breakpoint_q84)):
        color = breakpoint_colors[i % len(breakpoint_colors)]
        label = "68% breakpoint interval" if i == 0 else None
        ax.axvspan(q16, q84, color=color, alpha=0.22, linewidth=0, label=label)
        ax.axvline(q50, color=color, ls="--", lw=1.4, label=fr"$x_{i + 1}$ median")

    ax.set_xlabel(r"$\log_{10}(M/M_\oplus)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\oplus)$")
    ax.set_title("Nested-sampling radius-mass relation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def save_nested_outputs(
    df: pd.DataFrame,
    result: dict[str, object],
    output_prefix: str,
    show: bool = False,
) -> dict[str, str]:
    """Save posterior summary, residual table, and fit plot for one model."""
    n_breakpoints = int(result["n_breakpoints"])
    samples = np.asarray(result["posterior_samples"])
    best_theta = np.asarray(result["best_theta"])

    summary_path = PLOT_DIR / f"{output_prefix}_summary.csv"
    residual_path = PLOT_DIR / f"{output_prefix}_residual_coverage.csv"
    fit_path = PLOT_DIR / f"{output_prefix}_fit.png"

    summarize_posterior(samples, n_breakpoints).to_csv(summary_path, index=False)
    result["residual_table"].to_csv(residual_path, index=False)
    plot_nested_fit(df, samples, best_theta, fit_path, show=show)

    return {
        "summary_path": str(summary_path),
        "residual_coverage_path": str(residual_path),
        "fit_path": str(fit_path),
    }


def comparison_row(result: dict[str, object]) -> dict[str, object]:
    """Flatten one nested-sampling result for CSV comparison."""
    row = {
        "n_breakpoints": result["n_breakpoints"],
        "n_segments": result["n_segments"],
        "prior": result["prior"],
        "ndim": result["ndim"],
        "nlive": result["nlive"],
        "target_dlogz": result["target_dlogz"],
        "sample": result["sample"],
        "bound": result["bound"],
        "maxiter": result["maxiter"],
        "maxcall": result["maxcall"],
        "niter": result["niter"],
        "ncall": result["ncall"],
        "logz": result["logz"],
        "logzerr": result["logzerr"],
        "best_log_likelihood": result["best_log_likelihood"],
        "ppd_pte": result["ppd_pte"],
        "fraction_within_1sigma": result["fraction_within_1sigma"],
        "fraction_within_2sigma": result["fraction_within_2sigma"],
        "median_distance_over_sigma": result["median_distance_over_sigma"],
    }
    _, _, _, log_sigmas = MR.unpack_theta(np.asarray(result["best_theta"]))
    for i, value in enumerate(np.exp(log_sigmas), start=1):
        row[f"best_sigma_int_{i}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--breakpoints", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--compare", action="store_true", help="compare 1, 2, 3, and 4 breakpoints")
    parser.add_argument(
        "--compare-priors",
        action="store_true",
        help="with --compare, run all prior presets instead of only --prior",
    )
    parser.add_argument(
        "--prior",
        choices=("broad", "positive", "wide-scatter"),
        default="broad",
        help="prior preset for slopes and intrinsic scatters",
    )
    parser.add_argument("--nlive", type=int, default=500, help="number of live points")
    parser.add_argument("--dlogz", type=float, default=0.1, help="nested-sampling stopping criterion")
    parser.add_argument("--sample", default="rwalk", help="dynesty sample method")
    parser.add_argument("--bound", default="multi", help="dynesty bounding method")
    parser.add_argument("--maxiter", type=int, default=None)
    parser.add_argument("--maxcall", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    args = parser.parse_args()

    if dynesty is None:
        raise SystemExit(
            "dynesty is not installed in this environment. Install it with:\n"
            "  ML/ML-venv/bin/python -m pip install dynesty"
        )

    PLOT_DIR.mkdir(exist_ok=True)
    df = MR.prepare_fit_data(MR.load_dace_data())

    if args.compare:
        rows = []
        priors = ("broad", "positive", "wide-scatter") if args.compare_priors else (args.prior,)
        for prior in priors:
            for n_breakpoints in (1, 2, 3, 4):
                print(f"\n=== dynesty: breakpoints={n_breakpoints}, prior={prior} ===")
                result = run_nested_model(
                    df=df,
                    prior=prior,
                    n_breakpoints=n_breakpoints,
                    nlive=args.nlive,
                    dlogz=args.dlogz,
                    sample=args.sample,
                    bound=args.bound,
                    seed=args.seed,
                    maxiter=args.maxiter,
                    maxcall=args.maxcall,
                    print_progress=True,
                )
                prefix = f"dynesty_mass_radius_{n_breakpoints}_breakpoints_{prior}"
                paths = save_nested_outputs(df, result, prefix, show=False)
                row = comparison_row(result)
                row.update(paths)
                rows.append(row)

        comparison = pd.DataFrame(rows)
        comparison = comparison.sort_values("logz", ascending=False).reset_index(drop=True)
        comparison["delta_logz"] = comparison["logz"] - comparison["logz"].iloc[0]
        comparison["log_bayes_factor_vs_next"] = np.nan
        if len(comparison) > 1:
            logz = comparison["logz"].to_numpy()
            comparison.loc[: len(comparison) - 2, "log_bayes_factor_vs_next"] = (
                logz[:-1] - logz[1:]
            )

        comparison_path = PLOT_DIR / "dynesty_mass_radius_evidence_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print("\nEvidence comparison")
        print(
            comparison[
                [
                    "n_breakpoints",
                    "n_segments",
                    "prior",
                    "logz",
                    "logzerr",
                    "delta_logz",
                    "ppd_pte",
                    "fraction_within_1sigma",
                    "fraction_within_2sigma",
                ]
            ].to_string(index=False)
        )
        print(f"Saved comparison table to {comparison_path}")

    else:
        print(f"\n=== dynesty: breakpoints={args.breakpoints}, prior={args.prior} ===")
        result = run_nested_model(
            df=df,
            prior=args.prior,
            n_breakpoints=args.breakpoints,
            nlive=args.nlive,
            dlogz=args.dlogz,
            sample=args.sample,
            bound=args.bound,
            seed=args.seed,
            maxiter=args.maxiter,
            maxcall=args.maxcall,
            print_progress=True,
        )
        prefix = f"dynesty_mass_radius_{args.breakpoints}_breakpoints_{args.prior}"
        paths = save_nested_outputs(df, result, prefix, show=args.show)
        print(
            f"logZ = {result['logz']:.3f} +/- {result['logzerr']:.3f}, "
            f"PPD PTE = {result['ppd_pte']:.3f}"
        )
        print(f"Saved outputs under prefix {prefix} in {PLOT_DIR}")
        for key, path in paths.items():
            print(f"{key}: {path}")


if __name__ == "__main__":
    main()
