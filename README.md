# Exoplanet Mass-Radius Relation: Bayesian and ML Analysis

This project compares Bayesian inference and machine-learning approaches for
modelling the exoplanet mass-radius relation. Models are fitted in log space:

```text
log10(R / R_earth) = f(log10(M / M_earth))
```

The analyses use measured masses, radii, and their uncertainties from a cleaned
DACE exoplanet catalogue. The project includes interpretable broken-power-law
models alongside flexible regression methods and evaluates both predictive
accuracy and uncertainty coverage.

## Methods

| Script | Method |
| --- | --- |
| `emcee_M-R.py` | Bayesian continuous broken-power-law model sampled with `emcee`; compares 1-4 breakpoints using BIC, WAIC, and posterior predictive checks. |
| `dynesty_M-R.py` | Nested-sampling version of the broken-power-law model; compares breakpoint models using Bayesian evidence. |
| `gp_M-R.py` | Gaussian Process regression with heteroscedastic radius errors and propagated mass errors. |
| `rf_M-R.py` | Random Forest regression with configurable smoothness and tree-to-tree spread as an uncertainty proxy. |
| `qboost_svr_M-R.py` | Quantile Gradient Boosting and Support Vector Regression comparison. |
| `nn_bnn_M-R.py` | Neural Network and variational Bayesian Neural Network comparison. |
| `compare_ML_methods_M-R.py` | Runs the main ML methods on one shared train/test split and compares their metrics. |

The accompanying report source is in `MR_methods_report.tex`. Earlier
exploratory implementations are retained in `old_files/`.

## Data

Place the cleaned DACE catalogue at:

```text
DACE_Exo.csv
```

in the repository root. Catalogue files and generated outputs are intentionally
not tracked in Git.

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/gdetoma/M-R-relation-for-exoplanets-a-Bayesian-and-ML-Analysis.git
cd M-R-relation-for-exoplanets-a-Bayesian-and-ML-Analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Each script supports `--help` for its full set of options. Common runs are:

```bash
# Compare Bayesian broken-power-law models
python emcee_M-R.py --compare --prior broad
python dynesty_M-R.py --compare --prior broad

# Compare Gaussian Process kernels and Random Forest configurations
python gp_M-R.py --compare
python rf_M-R.py --compare

# Run the additional ML models
python qboost_svr_M-R.py
python nn_bnn_M-R.py

# Evaluate the main ML methods on a shared split
python compare_ML_methods_M-R.py
```

The Bayesian and neural-network analyses can take considerably longer than the
other methods. Use each script's sampling, epoch, or `--max-points` options for
quicker exploratory runs.

## Outputs

Scripts write figures, summary tables, predictions, and diagnostics to
`plots/`. Depending on the method, outputs include:

- fits and predictive uncertainty bands;
- posterior predictive checks and residual coverage tables;
- model-comparison tables for breakpoints, kernels, and hyperparameters;
- train/test metrics and predicted-versus-observed figures;
- MCMC trace and corner plots.

## Notes

- The Bayesian models include radius measurement uncertainty, propagated mass
  uncertainty, and intrinsic scatter.
- The Gaussian Process uses a global white-noise term as an intrinsic-scatter
  proxy.
- Random Forest tree spread is a practical uncertainty proxy, not a calibrated
  Bayesian predictive interval.
- Reproducible random seeds are exposed through the command-line interfaces.
