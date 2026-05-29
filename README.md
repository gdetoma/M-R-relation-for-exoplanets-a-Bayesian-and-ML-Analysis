# Exoplanet Mass-Radius Relation: Bayesian and ML Analysis

This repository contains a comparative analysis of the exoplanet mass-radius
relation using Bayesian inference and machine-learning methods.

The models fit the relation in log space:

```text
log10(R / R_earth) = f(log10(M / M_earth))
```

The analysis uses cleaned DACE exoplanet data with measured masses, radii, and
measurement uncertainties. The main goal is to compare physically interpretable
Bayesian broken-power-law models with more flexible Gaussian Process and Random
Forest regressions.

## Main Scripts

| File | Purpose |
| --- | --- |
| `emcee_M-R.py` | Bayesian piecewise-linear mass-radius model sampled with `emcee`; compares 1-4 breakpoints using BIC, WAIC, and posterior predictive checks. |
| `dynesty_M-R.py` | Nested-sampling version of the same piecewise-linear model using `dynesty`; compares breakpoint models with Bayesian evidence. |
| `gp_M-R.py` | Gaussian Process regression of the mass-radius relation with heteroscedastic radius errors and propagated mass errors. |
| `rf_M-R.py` | Random Forest regression with train/test hyperparameter comparison and tree-to-tree spread as an uncertainty proxy. |

## Data

The scripts expect the DACE catalogue file to be available as:

```text
DACE_Exo.csv
```

in the same directory as the Python scripts. The data file is not required to be
tracked in git; if it is not present, download or place it manually before
running the analyses.

## Environment

A typical Python environment needs:

```bash
pip install numpy pandas matplotlib scipy emcee corner dynesty scikit-learn
```

If you use a virtual environment:

```bash
python -m venv ML-venv
source ML-venv/bin/activate
pip install numpy pandas matplotlib scipy emcee corner dynesty scikit-learn
```

## Usage

Run the MCMC Bayesian analysis:

```bash
python emcee_M-R.py --compare --breakpoints 2 --prior broad
```

Run the nested-sampling evidence comparison:

```bash
python dynesty_M-R.py --compare --prior broad
```

Run the Gaussian Process kernel comparison:

```bash
python gp_M-R.py --compare
```

Run the Random Forest hyperparameter comparison:

```bash
python rf_M-R.py --compare
```

All scripts write summaries, diagnostic CSV files, and figures to:

```text
plots/
```

## Outputs

The analyses produce:

- Fit figures in log mass-log radius space.
- Posterior or predictive uncertainty bands.
- Posterior predictive checks.
- Residual coverage tables.
- Model-comparison tables for breakpoint, kernel, or hyperparameter choices.

## Notes

- The Bayesian models include radius measurement uncertainty, propagated mass
  uncertainty, and intrinsic scatter.
- The Gaussian Process uses a global white-noise term as an intrinsic-scatter
  proxy.
- The Random Forest tree spread is only a practical uncertainty proxy and is not
  a calibrated Bayesian predictive interval.
