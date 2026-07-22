# Statistical Analysis of the Exoplanet Mass–Radius Relation

**ETH Zürich, Switzerland**<br>
**Team Project — supervisor: Dr. T. Tröster**<br>
**September 2024 – January 2025**

This project investigates the exoplanet mass–radius relation using observational
data from the DACE Exoplanet Catalog. Bayesian inference and machine-learning
methods are combined to characterize distinct planetary regimes and compare
physically interpretable models with flexible predictive alternatives.

The analysis implements continuous broken power laws that explicitly account
for observational uncertainties and intrinsic scatter. Posterior distributions
are explored with MCMC using `emcee`, while Dynamic Nested Sampling with
`dynesty` provides Bayesian evidence for model comparison. BIC, WAIC, posterior
predictive checks, and evidence support a description with three broad
planetary regimes. Gaussian Processes and Random Forests provide flexible
statistical comparisons, evaluated against the Bayesian baseline on the same
held-out planets.

The full methodology and results are in
[ML_Project_Report.pdf](ML_Project_Report.pdf).

## Analyses

All models use

```text
log10(R / R_earth) = f(log10(M / M_earth)).
```

| Script | Analysis |
| --- | --- |
| `emcee_M-R.py` | Continuous broken-power-law inference with measurement errors in mass and radius, segment-dependent intrinsic scatter, MCMC posterior sampling, BIC, WAIC, and posterior predictive checks. |
| `dynesty_M-R.py` | The same broken-power-law family fitted with Dynamic Nested Sampling, allowing one-to-four-breakpoint models to be compared through Bayesian evidence. |
| `gp_M-R.py` | Gaussian Process regression with training-only kernel selection, heteroscedastic radius errors, propagated mass errors, and an optional MCMC check of kernel-hyperparameter uncertainty. |
| `rf_M-R.py` | Random Forest regression with training-only out-of-bag hyperparameter selection and tree-to-tree spread as an explicitly uncalibrated uncertainty proxy. |
| `compare_MR_fit_performance.py` | Leakage-free comparison of the two-breakpoint Bayesian model, selected GP, and pre-declared Random Forest on one mass-regime-stratified train/test split using RMSE, NLPD, and interval coverage. |

## Data

The analyses expect the cleaned DACE catalogue at `DACE_Exo.csv` in the
repository root. It must contain measured planetary masses, radii, and their
uncertainties; the loaders in `emcee_M-R.py` document the accepted column names
and unit conversion.

The DACE catalogue is not redistributed in this repository. Generated CSV
tables, figures, and the local `plots/` directory are also excluded from Git.
The report PDF is the sole tracked generated document.

## Installation

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/gdetoma/M-R-relation-for-exoplanets-a-Bayesian-and-ML-Analysis.git
cd M-R-relation-for-exoplanets-a-Bayesian-and-ML-Analysis
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproducing the analyses

Each script exposes its complete command-line interface through `--help`.
Representative full runs are:

```bash
# Bayesian broken-power-law model comparison
python emcee_M-R.py --compare --prior broad
python dynesty_M-R.py --compare --prior broad

# Training-only GP and RF model selection followed by held-out evaluation
python gp_M-R.py --compare
python rf_M-R.py --compare

# All three predictive methods on exactly the same held-out planets
python compare_MR_fit_performance.py
```

For a quick integration check of the comparison pipeline:

```bash
python compare_MR_fit_performance.py --smoke-test
```

The Bayesian calculations can take substantially longer than the empirical ML
fits. Sampling controls, random seeds, output paths, and reduced-data options
are documented in each script's help output.

## Outputs and interpretation

The scripts write fit figures, diagnostic plots, posterior or hyperparameter
summaries, residual-coverage tables, predictions, and model-comparison tables
to `plots/`.

- `emcee` and `dynesty` provide interpretable slopes, breakpoint masses, and
  explicit intrinsic-scatter estimates for planetary regimes.
- BIC, WAIC, and Bayesian evidence answer different model-selection questions;
  their breakpoint rankings should be interpreted together rather than treated
  as interchangeable scores.
- The GP is a flexible probabilistic regression, but its saved held-out check
  shows some tail under-dispersion.
- Random Forest tree spread measures ensemble disagreement. It is not a
  calibrated Bayesian or astrophysical predictive interval.
- The shared comparison reserves the test set until all method-specific choices
  have been fixed, preventing test-set leakage.

## Report source

`MR_methods_report.tex` and `references.bib` are included so the report can be
rebuilt when the analysis outputs in `plots/` are available. The bibliography
uses the BibTeX backend and can be built with Tectonic or a conventional LaTeX
toolchain.
