# ============================================================
# Gaussian Process for the Exoplanet Mass-Radius Relation
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv("nasa_exoplanets_mass_radius.csv")

features = [
    "log_mass",
    "st_teff",
    "pl_orbper",
    "pl_orbeccen"
]

target = "log_radius"

# Compute radius uncertainty
df["radius_err"] = 0.5 * (
    np.abs(df["pl_radeerr1"]) + np.abs(df["pl_radeerr2"])
)

df["log_radius_err"] = df["radius_err"] / (df["pl_rade"] * np.log(10))

# Remove invalid rows
df = df.dropna(subset=features + [target, "log_radius_err"]).copy()
df = df[df["log_radius_err"] > 0].copy()

X = df[features].values
y = df[target].values
y_err = df["log_radius_err"].values


# ============================================================
# 2. Train-validation-test split
# ============================================================

X_train, X_temp, y_train, y_temp, yerr_train, yerr_temp = train_test_split(
    X,
    y,
    y_err,
    test_size=0.3,
    random_state=42
)

X_val, X_test, y_val, y_test, yerr_val, yerr_test = train_test_split(
    X_temp,
    y_temp,
    yerr_temp,
    test_size=0.5,
    random_state=42
)


# ============================================================
# 3. Standardize inputs and target
# ============================================================

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
y_val_scaled = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
y_test_scaled = y_scaler.transform(y_test.reshape(-1, 1)).ravel()

# Scale observational uncertainties consistently with y
yerr_train_scaled = yerr_train / y_scaler.scale_[0]
yerr_val_scaled = yerr_val / y_scaler.scale_[0]
yerr_test_scaled = yerr_test / y_scaler.scale_[0]

# Add uncertainty floor
error_floor = 0.03
error_floor_scaled = error_floor / y_scaler.scale_[0]

yerr_train_scaled = np.maximum(yerr_train_scaled, error_floor_scaled)
yerr_val_scaled = np.maximum(yerr_val_scaled, error_floor_scaled)
yerr_test_scaled = np.maximum(yerr_test_scaled, error_floor_scaled)


# ============================================================
# 4. Try different length-scale bounds and compare test RMSE
# ============================================================

os.makedirs("plots", exist_ok=True)

length_scale_ranges = [
    (1e-2, 1e-1),
    (1e-1, 1),
    (1, 10),
    (10, 100),
    (100, 1000),
]

results = []
models = []

for bounds in length_scale_ranges:

    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2))
        * RBF(
            length_scale=np.ones(len(features)),
            length_scale_bounds=bounds
        )
        + WhiteKernel(
            noise_level=1e-3,
            noise_level_bounds=(1e-6, 1e1)
        )
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=yerr_train_scaled**2,
        n_restarts_optimizer=10,
        normalize_y=False,
        random_state=42
    )

    gp.fit(X_train_scaled, y_train_scaled)

    y_pred_test_scaled, y_std_test_scaled = gp.predict(
        X_test_scaled,
        return_std=True
    )

    y_pred_test = y_scaler.inverse_transform(
        y_pred_test_scaled.reshape(-1, 1)
    ).ravel()

    y_std_test = y_std_test_scaled * y_scaler.scale_[0]

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    results.append({
        "length_scale_bounds": bounds,
        "optimized_kernel": gp.kernel_,
        "test_RMSE": rmse,
        "test_MAE": mae,
        "test_R2": r2
    })

    models.append(gp)

results_df = pd.DataFrame(results)

print("\nComparison of GP length-scale ranges")
print("------------------------------------")
print(results_df[["length_scale_bounds", "test_RMSE", "test_MAE", "test_R2"]])

best_idx = results_df["test_RMSE"].idxmin()
best_gp = models[best_idx]

print("\nBest model")
print("----------")
print("Length-scale bounds:", results_df.loc[best_idx, "length_scale_bounds"])
print("Optimized kernel:", results_df.loc[best_idx, "optimized_kernel"])
print("Best test RMSE:", results_df.loc[best_idx, "test_RMSE"])


# ============================================================
# 5. Plot RMSE comparison
# ============================================================

plt.figure(figsize=(7, 4))

labels = [str(b) for b in results_df["length_scale_bounds"]]

plt.plot(
    labels,
    results_df["test_RMSE"],
    marker="o"
)

plt.xlabel("Allowed length-scale range")
plt.ylabel("Test RMSE")
plt.title("Test RMSE for different GP length-scale ranges")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("plots/gp_length_scale_range_rmse.png", dpi=300)
plt.show()


# ============================================================
# 6. Final prediction using best model
# ============================================================

y_pred_test_scaled, y_std_test_scaled = best_gp.predict(
    X_test_scaled,
    return_std=True
)

y_pred_test = y_scaler.inverse_transform(
    y_pred_test_scaled.reshape(-1, 1)
).ravel()

y_std_test = y_std_test_scaled * y_scaler.scale_[0]


# ============================================================
# 7. Plot predicted vs observed test radii
# ============================================================

plt.figure(figsize=(6, 6))

plt.errorbar(
    y_test,
    y_pred_test,
    yerr=y_std_test,
    fmt="o",
    alpha=0.6,
    label="Test planets"
)

min_val = min(y_test.min(), y_pred_test.min())
max_val = max(y_test.max(), y_pred_test.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--",
    label="Perfect prediction"
)

plt.xlabel(r"Observed $\log_{10}(R/R_\oplus)$")
plt.ylabel(r"Predicted $\log_{10}(R/R_\oplus)$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/gp_predicted_vs_observed_best_model.png", dpi=300)
plt.show()


# ============================================================
# 8. Plot learned mass-radius relation using best model
# ============================================================

mass_grid = np.linspace(df["log_mass"].min(), df["log_mass"].max(), 500)

median_teff = df["st_teff"].median()
median_period = df["pl_orbper"].median()
median_ecc = df["pl_orbeccen"].median()

X_grid = pd.DataFrame({
    "log_mass": mass_grid,
    "st_teff": median_teff,
    "pl_orbper": median_period,
    "pl_orbeccen": median_ecc
})

X_grid_scaled = x_scaler.transform(X_grid[features].values)

radius_pred_scaled, radius_std_scaled = best_gp.predict(
    X_grid_scaled,
    return_std=True
)

radius_pred_log = y_scaler.inverse_transform(
    radius_pred_scaled.reshape(-1, 1)
).ravel()

radius_std_log = radius_std_scaled * y_scaler.scale_[0]

plt.figure(figsize=(7, 5))

plt.scatter(
    df["log_mass"],
    df["log_radius"],
    alpha=0.4,
    label="Observed planets"
)

plt.plot(
    mass_grid,
    radius_pred_log,
    linewidth=3,
    label="Best GP mean prediction"
)

plt.fill_between(
    mass_grid,
    radius_pred_log - 2 * radius_std_log,
    radius_pred_log + 2 * radius_std_log,
    alpha=0.25,
    label=r"Best GP $2\sigma$ uncertainty"
)

plt.xlabel(r"$\log_{10}(M/M_\oplus)$")
plt.ylabel(r"$\log_{10}(R/R_\oplus)$")
plt.title("Best Gaussian Process mass-radius relation")
plt.legend()
plt.tight_layout()
plt.savefig("plots/gp_mass_radius_relation_best_model.png", dpi=300)
plt.show()