"""
Bayesian fit of the exoplanet mass-radius relation.

The model is fitted in log space:

    log10(R / R_earth) = f(log10(M / M_earth))

where f is a continuous piecewise-linear function. The likelihood
uses the measured radius uncertainty, propagates the mass uncertainty through
the local slope of the model, and includes one intrinsic scatter term per
segment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize

try:
    import emcee
except ImportError:  # The project venv currently does not include emcee.
    emcee = None

try:
    import corner
except ImportError:
    corner = None


JUPITER_TO_EARTH_MASS = 317.8284
JUPITER_TO_EARTH_RADIUS = 11.2089
PAPER_RELEASE_TS = 1688252923
RNG = np.random.default_rng(42)


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "DACE_Exo.csv"
PLOT_DIR = SCRIPT_DIR / "plots"


def load_dace_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load DACE data and return clean Earth-unit mass/radius measurements."""
    raw = pd.read_csv(path, delimiter=";", skiprows=[1, 2])

    columns = [
        "planet_mass",
        "planet_mass_lower",
        "planet_mass_upper",
        "planet_radius",
        "planet_radius_lower",
        "planet_radius_upper",
        "discovery_year",
        "last_updated",
    ]
    df = raw.loc[:, columns].copy()

    numeric_columns = columns[:-1]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["last_updated"] = pd.to_datetime(
        df["last_updated"], format="%d.%m.%y", errors="coerce"
    )
    df = df.dropna(subset=columns).copy()

    df["ts"] = (df["last_updated"].astype("int64") // 10**9).astype("int64")
    df = df.drop(
        df[(df["ts"] >= PAPER_RELEASE_TS) & (df["discovery_year"] == 2023)].index
    )

    df = df.rename(
        columns={
            "planet_mass": "mass",
            "planet_mass_lower": "mass_err_minus",
            "planet_mass_upper": "mass_err_plus",
            "planet_radius": "radius",
            "planet_radius_lower": "radius_err_minus",
            "planet_radius_upper": "radius_err_plus",
        }
    )

    df["mass"] *= JUPITER_TO_EARTH_MASS
    df["mass_err_minus"] = df["mass_err_minus"].abs() * JUPITER_TO_EARTH_MASS
    df["mass_err_plus"] = df["mass_err_plus"].abs() * JUPITER_TO_EARTH_MASS

    df["radius"] *= JUPITER_TO_EARTH_RADIUS
    df["radius_err_minus"] = df["radius_err_minus"].abs() * JUPITER_TO_EARTH_RADIUS
    df["radius_err_plus"] = df["radius_err_plus"].abs() * JUPITER_TO_EARTH_RADIUS

    df = df[(df["mass"] > 0) & (df["radius"] > 0)].copy()

    df["mass_err"] = 0.5 * (df["mass_err_minus"] + df["mass_err_plus"])
    df["radius_err"] = 0.5 * (df["radius_err_minus"] + df["radius_err_plus"])
    df = df[(df["mass_err"] > 0) & (df["radius_err"] > 0)].copy()

    df["log_mass"] = np.log10(df["mass"])
    df["log_radius"] = np.log10(df["radius"])
    df["log_mass_err"] = df["mass_err"] / (df["mass"] * np.log(10.0))
    df["log_radius_err"] = df["radius_err"] / (df["radius"] * np.log(10.0))

    return df.sort_values("radius").reset_index(drop=True)


def prepare_fit_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the full cleaned dataset, sorted by the predictor."""
    return df.sort_values("mass").reset_index(drop=True)


def parameter_names(n_breakpoints: int) -> tuple[str, ...]:
    """Parameter names for a model with n_breakpoints."""
    n_segments = n_breakpoints + 1
    return (
        ("a",)
        + tuple(f"b{i}" for i in range(1, n_segments + 1))
        + tuple(f"x{i}" for i in range(1, n_breakpoints + 1))
        + tuple(f"log_sigma_int_{i}" for i in range(1, n_segments + 1))
    )


def n_segments_from_theta(theta: np.ndarray) -> int:
    """Infer segment count from the dynamic parameter vector."""
    if len(theta) % 3 != 0:
        raise ValueError("Invalid theta length for piecewise model.")
    return len(theta) // 3


def unpack_theta(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return intercept, slopes, breakpoints, and log intrinsic scatters."""
    n_segments = n_segments_from_theta(theta)
    n_breakpoints = n_segments - 1
    a = theta[0]
    slopes = theta[1 : 1 + n_segments]
    breakpoints = theta[1 + n_segments : 1 + n_segments + n_breakpoints]
    log_sigmas = theta[1 + n_segments + n_breakpoints :]
    return a, slopes, breakpoints, log_sigmas


def piecewise_log_radius(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Continuous piecewise-linear log-radius model."""
    a, slopes, breakpoints, _ = unpack_theta(theta)
    y = a + slopes[0] * x
    for i, breakpoint in enumerate(breakpoints):
        y += (slopes[i + 1] - slopes[i]) * np.maximum(0.0, x - breakpoint)
    return y


def piecewise_slope(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Derivative d logR / d logM for uncertainty propagation."""
    _, slopes, breakpoints, _ = unpack_theta(theta)
    segment_index = np.searchsorted(breakpoints, x, side="left")
    return slopes[segment_index]


def piecewise_intrinsic_scatter(theta: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Intrinsic scatter in log-radius for the segment containing each point."""
    _, _, breakpoints, log_sigmas = unpack_theta(theta)
    segment_index = np.searchsorted(breakpoints, x, side="left")
    return np.exp(log_sigmas[segment_index])


def make_prior_bounds(
    x: np.ndarray,
    prior: str,
    n_breakpoints: int,
) -> dict[str, tuple[float, float]]:
    """Build broad priors that respect the data range."""
    xmin, xmax = float(np.min(x)), float(np.max(x))

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

    n_segments = n_breakpoints + 1
    bounds = {"a": (-2.5, 3.0)}
    for i in range(1, n_segments + 1):
        bounds[f"b{i}"] = slope_bounds

    # Keep every segment supported by data while allowing breakpoints to move.
    for i in range(1, n_breakpoints + 1):
        low_q = max(0.05, (i - 0.65) / (n_breakpoints + 1))
        high_q = min(0.95, (i + 0.65) / (n_breakpoints + 1))
        low, high = np.quantile(x, [low_q, high_q])
        bounds[f"x{i}"] = (max(xmin, float(low)), min(xmax, float(high)))

    for i in range(1, n_segments + 1):
        bounds[f"log_sigma_int_{i}"] = sigma_bounds

    return bounds


def log_prior(theta: np.ndarray, bounds: dict[str, tuple[float, float]]) -> float:
    """Uniform priors plus ordered breakpoints."""
    n_segments = n_segments_from_theta(theta)
    n_breakpoints = n_segments - 1
    names = parameter_names(n_breakpoints)

    for value, name in zip(theta, names):
        low, high = bounds[name]
        if value < low or value > high:
            return -np.inf

    _, _, breakpoints, _ = unpack_theta(theta)
    if np.any(np.diff(breakpoints) <= 0.03):
        return -np.inf

    return 0.0


def log_likelihood(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> float:
    """Gaussian likelihood with x-error propagation and segment scatter."""
    mu = piecewise_log_radius(theta, x)
    variance = total_y_variance(theta, x, xerr, yerr)

    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        return -np.inf

    residual = y - mu
    return float(-0.5 * np.sum(residual**2 / variance + np.log(2 * np.pi * variance)))


def pointwise_log_likelihood(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> np.ndarray:
    """Log-likelihood contribution for each planet."""
    mu = piecewise_log_radius(theta, x)
    variance = total_y_variance(theta, x, xerr, yerr)
    if np.any(~np.isfinite(variance)) or np.any(variance <= 0):
        return np.full_like(y, -np.inf, dtype=float)
    residual = y - mu
    return -0.5 * (residual**2 / variance + np.log(2 * np.pi * variance))


def information_criteria(
    map_theta: np.ndarray,
    chain: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    n_waic_samples: int = 1000,
) -> dict[str, float]:
    """Compute BIC and WAIC for model comparison."""
    n_data = len(y)
    n_params = len(map_theta)
    map_log_likelihood = log_likelihood(map_theta, x, y, xerr, yerr)
    bic = n_params * np.log(n_data) - 2.0 * map_log_likelihood

    sample_count = min(n_waic_samples, len(chain))
    indices = RNG.choice(len(chain), size=sample_count, replace=False)
    log_lik_samples = np.array(
        [pointwise_log_likelihood(theta, x, y, xerr, yerr) for theta in chain[indices]]
    )

    max_log_lik = np.max(log_lik_samples, axis=0)
    lppd = np.sum(max_log_lik + np.log(np.mean(np.exp(log_lik_samples - max_log_lik), axis=0)))
    p_waic = np.sum(np.var(log_lik_samples, axis=0, ddof=1))
    waic = -2.0 * (lppd - p_waic)

    return {
        "map_log_likelihood": float(map_log_likelihood),
        "bic": float(bic),
        "waic": float(waic),
        "p_waic": float(p_waic),
        "lppd": float(lppd),
    }


def total_y_variance(
    theta: np.ndarray,
    x: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> np.ndarray:
    """Total sigma_y^2 used by the likelihood."""
    slope = piecewise_slope(theta, x)
    sigma_int = piecewise_intrinsic_scatter(theta, x)
    return yerr**2 + (slope * xerr) ** 2 + sigma_int**2


def chi_square_statistic(
    y: np.ndarray,
    theta: np.ndarray,
    x: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> float:
    """Chi-square under the full model variance."""
    mu = piecewise_log_radius(theta, x)
    variance = total_y_variance(theta, x, xerr, yerr)
    return float(np.sum((y - mu) ** 2 / variance))


def posterior_predictive_pte(
    chain: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    n_samples: int = 1000,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Posterior predictive PTE using chi-square as the test statistic."""
    sample_count = min(n_samples, len(chain))
    indices = RNG.choice(len(chain), size=sample_count, replace=False)
    t_data = np.empty(sample_count)
    t_rep = np.empty(sample_count)

    for i, theta in enumerate(chain[indices]):
        mu = piecewise_log_radius(theta, x)
        variance = total_y_variance(theta, x, xerr, yerr)
        y_rep = RNG.normal(mu, np.sqrt(variance))
        t_data[i] = chi_square_statistic(y, theta, x, xerr, yerr)
        t_rep[i] = chi_square_statistic(y_rep, theta, x, xerr, yerr)

    return float(np.mean(t_rep >= t_data)), t_rep, t_data


def residual_coverage_check(
    df: pd.DataFrame,
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    """Check how many planets sit within 1 and 2 total y-sigma of the MAP line."""
    fitted_y = piecewise_log_radius(theta, x)
    sigma_y = np.sqrt(total_y_variance(theta, x, xerr, yerr))
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


def log_probability(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> float:
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, x, y, xerr, yerr)


def negative_log_probability_for_optimizer(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    bounds: dict[str, tuple[float, float]],
) -> float:
    """Finite objective for scipy; MCMC still uses -inf outside support."""
    logp = log_probability(theta, x, y, xerr, yerr, bounds)
    if not np.isfinite(logp):
        return 1e100
    return -logp


def segment_indices(x: np.ndarray, breakpoints: np.ndarray) -> np.ndarray:
    """Return the segment index for each x."""
    return np.searchsorted(breakpoints, x, side="left")


def initial_guess(x: np.ndarray, y: np.ndarray, n_breakpoints: int) -> np.ndarray:
    """Use a single straight-line fit as a stable center for optimization."""
    n_segments = n_breakpoints + 1
    slope, intercept = np.polyfit(x, y, deg=1)
    breakpoints = np.quantile(
        x, np.linspace(1.0 / (n_segments + 1), n_segments / (n_segments + 1), n_breakpoints)
    )
    scatter = np.std(y - (intercept + slope * x))
    log_scatter = np.log(max(scatter, 0.03))
    return np.concatenate(
        [
            np.array([intercept]),
            np.full(n_segments, slope),
            breakpoints,
            np.full(n_segments, log_scatter),
        ]
    )


def segmented_initial_guess(
    x: np.ndarray,
    y: np.ndarray,
    break_quantiles: np.ndarray,
) -> np.ndarray:
    """Least-squares continuous piecewise-linear start for fixed break quantiles."""
    breakpoints = np.quantile(x, break_quantiles)
    design = np.column_stack(
        [np.ones_like(x), x]
        + [np.maximum(0.0, x - breakpoint) for breakpoint in breakpoints]
    )
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    a = coefficients[0]
    slopes = np.empty(len(breakpoints) + 1)
    slopes[0] = coefficients[1]
    for i, delta in enumerate(coefficients[2:], start=1):
        slopes[i] = slopes[i - 1] + delta

    theta = np.concatenate([np.array([a]), slopes, breakpoints, np.zeros(len(slopes))])
    residual = y - piecewise_log_radius(theta, x)
    scatter_floor = 0.02
    indices = segment_indices(x, breakpoints)
    sigmas = []
    for i in range(len(slopes)):
        segment_residuals = residual[indices == i]
        sigmas.append(max(np.std(segment_residuals), scatter_floor))
    theta[-len(slopes) :] = np.log(sigmas)
    return theta


def deterministic_initial_guesses(
    x: np.ndarray,
    y: np.ndarray,
    n_breakpoints: int,
) -> list[np.ndarray]:
    """Initial points designed to cover plausible three-regime break locations."""
    starts = [initial_guess(x, y, n_breakpoints)]
    candidate_quantiles = [
        np.linspace(0.15, 0.85, n_breakpoints),
        np.linspace(0.20, 0.80, n_breakpoints),
        np.linspace(0.10, 0.90, n_breakpoints),
        np.linspace(0.25, 0.75, n_breakpoints),
    ]
    if n_breakpoints == 1:
        candidate_quantiles.extend([np.array([0.33]), np.array([0.50]), np.array([0.67])])
    for quantiles in candidate_quantiles:
        starts.append(segmented_initial_guess(x, y, np.asarray(quantiles)))
    return starts


def random_initial_guess(
    x: np.ndarray,
    y: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    n_breakpoints: int,
) -> np.ndarray:
    """Build a valid randomized start within the prior support."""
    n_segments = n_breakpoints + 1
    slope, intercept = np.polyfit(x, y, deg=1)
    names = parameter_names(n_breakpoints)
    start = np.empty(len(names))
    start[0] = intercept + RNG.normal(0.0, 0.25)
    start[1 : 1 + n_segments] = slope + RNG.normal(0.0, 0.4, size=n_segments)

    breakpoints = []
    last = -np.inf
    for i in range(1, n_breakpoints + 1):
        low, high = bounds[f"x{i}"]
        low = max(low, last + 0.04)
        if low >= high:
            low, high = np.quantile(x, [i / (n_breakpoints + 2), (i + 1) / (n_breakpoints + 2)])
            low = max(low, last + 0.04)
        value = RNG.uniform(low, high)
        breakpoints.append(value)
        last = value
    break_start = 1 + n_segments
    start[break_start : break_start + n_breakpoints] = breakpoints

    residual_scatter = max(np.std(y - (intercept + slope * x)), 0.03)
    start[-n_segments:] = np.log(residual_scatter) + RNG.normal(0.0, 0.35, size=n_segments)

    for i, name in enumerate(names):
        low, high = bounds[name]
        start[i] = np.clip(start[i], low + 1e-4, high - 1e-4)
    return start


def find_map(
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    n_starts: int,
    n_breakpoints: int,
) -> scipy.optimize.OptimizeResult:
    """Find a posterior mode from several bounded scipy optimizations."""
    names = parameter_names(n_breakpoints)
    scipy_bounds = [bounds[name] for name in names]
    starts = deterministic_initial_guesses(x, y, n_breakpoints)
    starts.extend(
        random_initial_guess(x, y, bounds, n_breakpoints)
        for _ in range(max(0, n_starts - len(starts)))
    )

    best_result = None
    for start in starts:
        for i, name in enumerate(names):
            low, high = bounds[name]
            start[i] = np.clip(start[i], low + 1e-4, high - 1e-4)

        result = scipy.optimize.minimize(
            negative_log_probability_for_optimizer,
            x0=start,
            args=(x, y, xerr, yerr, bounds),
            bounds=scipy_bounds,
            method="L-BFGS-B",
            options={"maxiter": 20_000},
        )

        if not np.isfinite(result.fun):
            continue
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    if best_result is None:
        raise RuntimeError("MAP optimization failed for all initial points.")
    return best_result


def sample_starting_points(
    center: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    n_walkers: int,
) -> np.ndarray:
    """Generate valid initial walker positions around the MAP."""
    ndim = len(center)
    n_segments = n_segments_from_theta(center)
    n_breakpoints = n_segments - 1
    names = parameter_names(n_breakpoints)
    scales = np.concatenate(
        [
            np.array([0.03]),
            np.full(n_segments, 0.08),
            np.full(n_breakpoints, 0.02),
            np.full(n_segments, 0.08),
        ]
    )
    walkers = []
    attempts = 0

    while len(walkers) < n_walkers and attempts < 10_000:
        attempts += 1
        candidate = center + scales * RNG.normal(size=ndim)
        for i, name in enumerate(names):
            low, high = bounds[name]
            candidate[i] = np.clip(candidate[i], low + 1e-6, high - 1e-6)
        if np.isfinite(log_prior(candidate, bounds)):
            walkers.append(candidate)

    if len(walkers) != n_walkers:
        raise RuntimeError("Could not initialize valid MCMC walkers.")

    return np.array(walkers)


def run_emcee_sampler(
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    map_theta: np.ndarray,
    n_walkers: int,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, float]:
    """Run emcee when available."""
    ndim = len(map_theta)
    n_walkers = max(n_walkers, 2 * ndim + 2)
    p0 = sample_starting_points(map_theta, bounds, n_walkers)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_probability,
        args=(x, y, xerr, yerr, bounds),
        moves=emcee.moves.StretchMove(a=2.0),
    )
    sampler.run_mcmc(p0, n_steps, progress=True)

    try:
        tau = sampler.get_autocorr_time(tol=0)
        burn = int(max(2 * np.max(tau), 0.2 * n_steps))
        thin = max(1, int(0.5 * np.min(tau)))
    except Exception:
        tau = None
        burn = int(0.3 * n_steps)
        thin = 10

    flat_chain = sampler.get_chain(discard=burn, thin=thin, flat=True)
    walker_chain = sampler.get_chain()
    acceptance = float(np.mean(sampler.acceptance_fraction))
    return flat_chain, walker_chain, tau, acceptance


def run_metropolis_sampler(
    x: np.ndarray,
    y: np.ndarray,
    xerr: np.ndarray,
    yerr: np.ndarray,
    bounds: dict[str, tuple[float, float]],
    map_theta: np.ndarray,
    n_steps: int,
) -> tuple[np.ndarray, np.ndarray, None, float]:
    """Small fallback sampler so the script remains runnable without emcee."""
    n_segments = n_segments_from_theta(map_theta)
    n_breakpoints = n_segments - 1
    proposal_scale = np.concatenate(
        [
            np.array([0.02]),
            np.full(n_segments, 0.05),
            np.full(n_breakpoints, 0.01),
            np.full(n_segments, 0.05),
        ]
    )
    chain = np.zeros((n_steps, len(map_theta)))
    theta = map_theta.copy()
    logp = log_probability(theta, x, y, xerr, yerr, bounds)
    accepted = 0

    for i in range(n_steps):
        proposal = theta + proposal_scale * RNG.normal(size=len(theta))
        proposal_logp = log_probability(proposal, x, y, xerr, yerr, bounds)
        if np.log(RNG.uniform()) < proposal_logp - logp:
            theta = proposal
            logp = proposal_logp
            accepted += 1
        chain[i] = theta

    burn = int(0.3 * n_steps)
    thin = 5
    return chain[burn::thin], chain[:, None, :], None, accepted / n_steps


def summarize_chain(chain: np.ndarray) -> pd.DataFrame:
    rows = []
    names = parameter_names(n_segments_from_theta(chain[0]) - 1)
    for i, name in enumerate(names):
        q16, q50, q84 = np.percentile(chain[:, i], [16, 50, 84])
        rows.append(
            {
                "parameter": name,
                "median": q50,
                "minus_1sigma": q50 - q16,
                "plus_1sigma": q84 - q50,
            }
        )
    return pd.DataFrame(rows)


def posterior_predictive(
    chain: np.ndarray,
    x_grid: np.ndarray,
    n_samples: int = 500,
) -> np.ndarray:
    sample_count = min(n_samples, len(chain))
    indices = RNG.choice(len(chain), size=sample_count, replace=False)
    return np.array([piecewise_log_radius(theta, x_grid) for theta in chain[indices]])


def posterior_predictive_observations(
    chain: np.ndarray,
    x_grid: np.ndarray,
    n_samples: int = 500,
) -> np.ndarray:
    """Draw noiseless-x posterior predictive radii including intrinsic scatter."""
    sample_count = min(n_samples, len(chain))
    indices = RNG.choice(len(chain), size=sample_count, replace=False)
    draws = []
    for theta in chain[indices]:
        mu = piecewise_log_radius(theta, x_grid)
        sigma_int = piecewise_intrinsic_scatter(theta, x_grid)
        draws.append(RNG.normal(mu, sigma_int))
    return np.array(draws)


def plot_fit(
    df: pd.DataFrame,
    map_theta: np.ndarray,
    chain: np.ndarray,
    output: Path,
    show: bool = False,
) -> None:
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()

    x_grid = np.linspace(x.min(), x.max(), 300)
    mean_draws = posterior_predictive(chain, x_grid)
    mean_q025, mean_q16, mean_q50, mean_q84, mean_q975 = np.percentile(
        mean_draws, [2.5, 16, 50, 84, 97.5], axis=0
    )
    predictive_draws = posterior_predictive_observations(chain, x_grid)
    pred_q025, pred_q16, pred_q84, pred_q975 = np.percentile(
        predictive_draws, [2.5, 16, 84, 97.5], axis=0
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt=".", alpha=0.65, label="DACE planets")
    ax.fill_between(
        x_grid,
        pred_q025,
        pred_q975,
        color="C2",
        alpha=0.12,
        linewidth=0,
        label="95% predictive band",
    )
    ax.fill_between(
        x_grid,
        pred_q16,
        pred_q84,
        color="C2",
        alpha=0.22,
        linewidth=0,
        label="68% predictive band",
    )
    ax.plot(x_grid, piecewise_log_radius(map_theta, x_grid), color="C3", lw=2, label="MAP")
    ax.plot(x_grid, mean_q50, color="black", lw=2, label="posterior median")
    ax.fill_between(
        x_grid,
        mean_q16,
        mean_q84,
        color="C0",
        alpha=0.25,
        linewidth=0,
        label="68% mean-relation band",
    )
    ax.fill_between(
        x_grid,
        mean_q025,
        mean_q975,
        color="C0",
        alpha=0.12,
        linewidth=0,
        label="95% mean-relation band",
    )
    _, _, breakpoints, _ = unpack_theta(map_theta)
    for breakpoint in breakpoints:
        ax.axvline(breakpoint, color="0.5", ls=":", lw=1)
    ax.set_xlabel(r"$\log_{10}(M/M_\oplus)$")
    ax.set_ylabel(r"$\log_{10}(R/R_\oplus)$")
    ax.set_title("Bayesian radius-mass relation")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_trace(walker_chain: np.ndarray, output: Path, show: bool = False) -> None:
    names = parameter_names(n_segments_from_theta(walker_chain[0, 0]) - 1)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 8), sharex=True)
    for i, name in enumerate(names):
        axes[i].plot(walker_chain[:, :, i], color="black", alpha=0.20, lw=0.6)
        axes[i].set_ylabel(name)
    axes[-1].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def plot_corner(chain: np.ndarray, output: Path, show: bool = False) -> None:
    if corner is None:
        print("corner is not installed; skipping corner plot.")
        return

    names = parameter_names(n_segments_from_theta(chain[0]) - 1)
    fig = corner.corner(
        chain,
        labels=names,
        quantiles=[0.16, 0.50, 0.84],
        show_titles=True,
        title_fmt=".3f",
    )
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
    ax.set_xlabel(r"$T(y^\mathrm{rep}, \theta) - T(y, \theta)$")
    ax.set_ylabel("density")
    ax.set_title(f"Posterior predictive check: PTE = {pte:.3f}")
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    if show:
        plt.show()
    plt.close(fig)


def run_analysis(
    raw_df: pd.DataFrame,
    prior: str,
    n_breakpoints: int,
    n_starts: int,
    n_steps: int,
    n_walkers: int,
    ppd_samples: int,
    waic_samples: int,
    output_prefix: str,
    make_plots: bool,
    show: bool,
) -> dict[str, float | str | int]:
    """Run one complete model fit and posterior predictive check."""
    df = prepare_fit_data(raw_df)
    x = df["log_mass"].to_numpy()
    y = df["log_radius"].to_numpy()
    xerr = df["log_mass_err"].to_numpy()
    yerr = df["log_radius_err"].to_numpy()
    bounds = make_prior_bounds(x, prior, n_breakpoints)

    print(f"\n=== breakpoints={n_breakpoints}, prior={prior} ===")
    print(f"Using {len(df)} planets.")
    print(f"Radius range: {df['radius'].min():.3f} - {df['radius'].max():.3f} R_earth")
    print(f"Mass range: {df['mass'].min():.3f} - {df['mass'].max():.3f} M_earth")

    map_result = find_map(x, y, xerr, yerr, bounds, n_starts, n_breakpoints)
    if not np.isfinite(-map_result.fun):
        raise RuntimeError("MAP optimization failed to find a finite posterior.")

    map_theta = np.asarray(map_result.x)
    names = parameter_names(n_breakpoints)
    _, _, breakpoints, log_sigmas = unpack_theta(map_theta)
    print("\nMAP estimate")
    for name, value in zip(names, map_theta):
        print(f"{name:>15s} = {value: .4f}")
    print("Intrinsic scatter by segment:")
    for i, value in enumerate(np.exp(log_sigmas), start=1):
        print(f"{f'sigma_int_{i}':>15s} = {value: .4f} dex")

    if emcee is not None:
        chain, walker_chain, tau, acceptance = run_emcee_sampler(
            x, y, xerr, yerr, bounds, map_theta, n_walkers, n_steps
        )
    else:
        print("\nemcee is not installed; using a simple Metropolis fallback sampler.")
        print("Install emcee to use the intended ensemble sampler.")
        chain, walker_chain, tau, acceptance = run_metropolis_sampler(
            x, y, xerr, yerr, bounds, map_theta, max(n_steps * n_walkers, 5000)
        )

    print(f"\nMean acceptance fraction: {acceptance:.3f}")
    if tau is not None:
        print("Autocorrelation time:")
        for name, value in zip(names, tau):
            print(f"{name:>15s} = {value:.1f}")

    summary = summarize_chain(chain)
    summary_path = PLOT_DIR / f"{output_prefix}_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nPosterior summary: median -/+ 1 sigma")
    for row in summary.itertuples(index=False):
        print(
            f"{row.parameter:>15s} = {row.median: .4f} "
            f"-{row.minus_1sigma:.4f} +{row.plus_1sigma:.4f}"
        )

    sigma_summary = []
    print("\nIntrinsic scatter summary in dex: median -/+ 1 sigma")
    sigma_start = len(names) - (n_breakpoints + 1)
    for i, column in enumerate(range(sigma_start, len(names)), start=1):
        sigma_samples = np.exp(chain[:, column])
        q16, q50, q84 = np.percentile(sigma_samples, [16, 50, 84])
        sigma_summary.append(q50)
        print(f"{f'sigma_int_{i}':>15s} = {q50: .4f} -{q50 - q16:.4f} +{q84 - q50:.4f}")

    ppd_pte, t_rep, t_data = posterior_predictive_pte(
        chain, x, y, xerr, yerr, n_samples=ppd_samples
    )
    print(f"\nPPD PTE = {ppd_pte:.3f}")

    criteria = information_criteria(
        map_theta, chain, x, y, xerr, yerr, n_waic_samples=waic_samples
    )
    print(
        "Information criteria: "
        f"BIC = {criteria['bic']:.2f}, "
        f"WAIC = {criteria['waic']:.2f}, "
        f"p_WAIC = {criteria['p_waic']:.2f}"
    )

    coverage, residual_table = residual_coverage_check(df, map_theta, x, y, xerr, yerr)
    residual_path = PLOT_DIR / f"{output_prefix}_residual_coverage.csv"
    residual_table.to_csv(residual_path, index=False)
    print(
        "MAP residual coverage: "
        f"{coverage['n_within_1sigma']}/{coverage['n_planets']} "
        f"({coverage['fraction_within_1sigma']:.3f}) within 1 sigma, "
        f"{coverage['n_within_2sigma']}/{coverage['n_planets']} "
        f"({coverage['fraction_within_2sigma']:.3f}) within 2 sigma"
    )

    if make_plots:
        plot_fit(df, map_theta, chain, PLOT_DIR / f"{output_prefix}_fit.png", show)
        plot_trace(walker_chain, PLOT_DIR / f"{output_prefix}_trace.png", show)
        plot_corner(chain, PLOT_DIR / f"{output_prefix}_corner.png", show)
        plot_ppd_check(t_rep, t_data, ppd_pte, PLOT_DIR / f"{output_prefix}_ppd_pte.png", show)

    return {
        "n_breakpoints": n_breakpoints,
        "n_segments": n_breakpoints + 1,
        "prior": prior,
        "n_planets": len(df),
        "map_log_posterior": -float(map_result.fun),
        "map_log_likelihood": criteria["map_log_likelihood"],
        "bic": criteria["bic"],
        "waic": criteria["waic"],
        "p_waic": criteria["p_waic"],
        "lppd": criteria["lppd"],
        "acceptance": acceptance,
        "ppd_pte": ppd_pte,
        "fraction_within_1sigma": coverage["fraction_within_1sigma"],
        "fraction_within_2sigma": coverage["fraction_within_2sigma"],
        "median_distance_over_sigma": coverage["median_distance_over_sigma"],
        **{f"sigma_int_{i}_median": value for i, value in enumerate(sigma_summary, start=1)},
        "summary_path": str(summary_path),
        "residual_coverage_path": str(residual_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4000, help="MCMC steps")
    parser.add_argument("--walkers", type=int, default=32, help="emcee walkers")
    parser.add_argument("--n-starts", type=int, default=24, help="MAP initial points")
    parser.add_argument(
        "--breakpoints",
        type=int,
        choices=(1, 2, 3, 4),
        default=2,
        help="number of breakpoints in the piecewise-linear relation",
    )
    parser.add_argument("--ppd-samples", type=int, default=1000, help="posterior samples for PPD PTE")
    parser.add_argument("--waic-samples", type=int, default=1000, help="posterior samples for WAIC")
    parser.add_argument(
        "--prior",
        choices=("broad", "positive", "wide-scatter"),
        default="broad",
        help="prior preset for slopes and intrinsic scatters",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="compare models with 1, 2, 3, and 4 breakpoints for the selected prior",
    )
    parser.add_argument(
        "--compare-steps",
        type=int,
        default=800,
        help="MCMC steps for each comparison-grid run",
    )
    parser.add_argument("--show", action="store_true", help="show plots interactively")
    args = parser.parse_args()

    PLOT_DIR.mkdir(exist_ok=True)

    raw_df = load_dace_data()

    if args.compare:
        comparison_rows = []
        for n_breakpoints in (1, 2, 3, 4):
            prefix = f"emcee_mass_radius_compare_{n_breakpoints}_breakpoints_{args.prior}"
            comparison_rows.append(
                run_analysis(
                    raw_df=raw_df,
                    prior=args.prior,
                    n_breakpoints=n_breakpoints,
                    n_starts=max(args.n_starts, 24),
                    n_steps=args.compare_steps,
                    n_walkers=args.walkers,
                    ppd_samples=args.ppd_samples,
                    waic_samples=args.waic_samples,
                    output_prefix=prefix,
                    make_plots=False,
                    show=False,
                )
            )

        comparison = pd.DataFrame(comparison_rows)
        comparison["abs_pte_minus_half"] = (comparison["ppd_pte"] - 0.5).abs()
        comparison["delta_bic"] = comparison["bic"] - comparison["bic"].min()
        comparison["delta_waic"] = comparison["waic"] - comparison["waic"].min()
        comparison = comparison.sort_values(["waic", "bic"])
        comparison_path = PLOT_DIR / "emcee_mass_radius_model_comparison.csv"
        comparison.to_csv(comparison_path, index=False)
        print("\nModel comparison")
        print(
            comparison[
                [
                    "prior",
                    "n_breakpoints",
                    "n_segments",
                    "n_planets",
                    "bic",
                    "delta_bic",
                    "waic",
                    "delta_waic",
                    "p_waic",
                    "ppd_pte",
                    "fraction_within_1sigma",
                    "fraction_within_2sigma",
                    "abs_pte_minus_half",
                    "acceptance",
                ]
            ].to_string(index=False)
        )
        print(f"Saved comparison table to {comparison_path}")

    run_analysis(
        raw_df=raw_df,
        prior=args.prior,
        n_breakpoints=args.breakpoints,
        n_starts=args.n_starts,
        n_steps=args.steps,
        n_walkers=args.walkers,
        ppd_samples=args.ppd_samples,
        waic_samples=args.waic_samples,
        output_prefix="emcee_mass_radius",
        make_plots=True,
        show=args.show,
    )

    print(f"\nSaved outputs in {PLOT_DIR}")


if __name__ == "__main__":
    main()
