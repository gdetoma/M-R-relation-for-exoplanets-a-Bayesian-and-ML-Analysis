import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from urllib.parse import quote

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# # ============================================================
# # 1. Download data from the NASA Exoplanet Archive
# # ============================================================

# # We use the PSCompPars table, which contains one row per planet
# # and is useful for population-level studies.
# # NASA provides access through the TAP service.
# # See: NASA Exoplanet Archive TAP / PSCompPars documentation.
# # Sources: NASA Exoplanet Archive TAP and PSCompPars docs.
# # ============================================================

# query = """
# SELECT
#     pl_name,
#     hostname,
#     pl_bmasse,
#     pl_bmasseerr1,
#     pl_bmasseerr2,
#     pl_rade,
#     pl_radeerr1,
#     pl_radeerr2,
#     pl_orbper,
#     pl_orbeccen,
#     pl_insol,
#     pl_eqt,
#     st_teff,
#     st_mass,
#     st_rad,
#     sy_dist,
#     discoverymethod,
#     disc_year
# FROM pscomppars
# WHERE pl_bmasse IS NOT NULL
# AND pl_rade IS NOT NULL
# """

# url = (
#     "https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
#     "query=" + quote(query) +
#     "&format=csv"
# )

# df = pd.read_csv(url)


# # ============================================================
# # 2. Basic cleaning
# # ============================================================

# # Keep only physically meaningful positive masses and radii.
# df = df[(df["pl_bmasse"] > 0) & (df["pl_rade"] > 0)]

# # Define logarithmic mass and radius, as in your Bayesian analysis.
# df["log_mass"] = np.log10(df["pl_bmasse"])
# df["log_radius"] = np.log10(df["pl_rade"])

# # Save the cleaned dataset.
# df.to_csv("nasa_exoplanets_mass_radius.csv", index=False)

# print("Number of planets with mass and radius:", len(df))
# print(df.head())



df = pd.read_csv("nasa_exoplanets_mass_radius.csv")

print("Number of planets:", len(df))
print(df.head())
# ============================================================
# 3. First ML model: radius predicted only from mass
# ============================================================

# This is the closest ML analogue of your Bayesian mass-radius model:
# log10(R/R_earth) = f(log10(M/M_earth))

X = df[["log_mass"]]
y = df["log_radius"]


# Split data into training and testing sets.
# The test set is kept aside to evaluate predictive performance.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Define the Random Forest regressor.
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=3,
    random_state=42
)

# Train the model.
rf.fit(X_train, y_train)


# ============================================================
# 4. Evaluate the model
# ============================================================

y_pred = rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel performance on test set:")
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)


# ============================================================
# 5. Plot the learned mass-radius relation
# ============================================================

mass_grid = np.linspace(df["log_mass"].min(), df["log_mass"].max(), 500)
X_grid = pd.DataFrame({"log_mass": mass_grid})

radius_pred = rf.predict(X_grid)

plt.figure(figsize=(7, 5))
plt.scatter(
    df["log_mass"],
    df["log_radius"],
    alpha=0.5,
    label="Observed planets"
)
plt.plot(
    mass_grid,
    radius_pred,
    linewidth=2,
    label="Random Forest",
    color="red"
)

plt.xlabel(r"$\log_{10}(M/M_\oplus)$")
plt.ylabel(r"$\log_{10}(R/R_\oplus)$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/random_forest_M-R.png")
plt.show()


# ============================================================
# 6. Hyperparameter tuning
# ============================================================

param_grid = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 3, 5, 10]
}

grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(grid.best_params_)

print("\nBest cross-validated RMSE:")
print(-grid.best_score_)

best_rf = grid.best_estimator_



import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv("nasa_exoplanets_mass_radius.csv")


# ============================================================
# 2. Select features
# ============================================================

features = [
    "pl_bmasse",    # planet mass
    "pl_orbper",    # orbital period
    "pl_orbeccen",  # eccentricity
    "st_teff"       # stellar effective temperature
]

target = "pl_rade"  # planet radius


# ============================================================
# 3. Keep only planets with all required values
# ============================================================

df_ml = df.dropna(subset=features + [target]).copy()

df_ml = df_ml[
    (df_ml["pl_bmasse"] > 0) &
    (df_ml["pl_orbper"] > 0) &
    (df_ml["pl_rade"] > 0) &
    (df_ml["st_teff"] > 0) &
    (df_ml["pl_orbeccen"] >= 0)
]

print("Number of planets usable for ML:", len(df_ml))


# ============================================================
# 4. Log-transform positive features
# ============================================================

df_ml["log_mass"] = np.log10(df_ml["pl_bmasse"])
df_ml["log_orbper"] = np.log10(df_ml["pl_orbper"])
df_ml["log_teff"] = np.log10(df_ml["st_teff"])
df_ml["log_radius"] = np.log10(df_ml["pl_rade"])

ml_features = [
    "log_mass",
    "log_orbper",
    "pl_orbeccen",
    "log_teff"
]

ml_target = "log_radius"


# ============================================================
# 7. Hyperparameter tuning for Random Forest
# ============================================================

param_grid = {
    "n_estimators": [200, 500, 1000],
    "max_depth": [3, 5, 10, None],
    "min_samples_leaf": [1, 3, 5, 10],
    "max_features": ["sqrt", None]
}

grid = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\nBest hyperparameters:")
print(grid.best_params_)

print("\nBest cross-validated RMSE:")
print(-grid.best_score_)

# Best Random Forest model
best_rf = grid.best_estimator_


# ============================================================
# 8. Evaluate best model on test set
# ============================================================

y_pred = best_rf.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nBest Random Forest with physical features")
print("-----------------------------------------")
print("Test RMSE:", rmse)
print("Test MAE:", mae)
print("Test R²:", r2)


# ============================================================
# 9. Estimate uncertainty from different trees
# ============================================================

# Each tree in the forest gives its own prediction.
# The spread among trees gives an approximate uncertainty.
tree_predictions = np.array([
    tree.predict(X_test)
    for tree in best_rf.estimators_
])

# Mean prediction across trees
y_pred_mean = tree_predictions.mean(axis=0)

# Standard deviation across trees
y_pred_std = tree_predictions.std(axis=0)

# Approximate 68% interval
y_pred_lower_1sigma = y_pred_mean - y_pred_std
y_pred_upper_1sigma = y_pred_mean + y_pred_std

# Approximate 95% interval
y_pred_lower_2sigma = y_pred_mean - 2 * y_pred_std
y_pred_upper_2sigma = y_pred_mean + 2 * y_pred_std


# Store results in a dataframe
prediction_results = X_test.copy()
prediction_results["y_true"] = y_test
prediction_results["y_pred"] = y_pred_mean
prediction_results["y_pred_std"] = y_pred_std
prediction_results["lower_1sigma"] = y_pred_lower_1sigma
prediction_results["upper_1sigma"] = y_pred_upper_1sigma
prediction_results["lower_2sigma"] = y_pred_lower_2sigma
prediction_results["upper_2sigma"] = y_pred_upper_2sigma

print("\nPredictions with tree-based uncertainty:")
print(prediction_results.head())


# Save predictions
prediction_results.to_csv(
    "random_forest_predictions_with_uncertainty.csv",
    index=False
)

# ============================================================
# 9. Feature importance
# ============================================================

importance = pd.DataFrame({
    "feature": ml_features,
    "importance": best_rf.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature importance:")
print(importance)


# ============================================================
# 10. Save feature importance plot
# ============================================================

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(7, 5))
plt.barh(importance["feature"], importance["importance"])
plt.gca().invert_yaxis()
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.title("Random Forest: physics features")
plt.tight_layout()
plt.savefig("plots/random_forest_feature_importance.png", dpi=300)
plt.show()


# ============================================================
# 11. Predicted vs observed plot
# ============================================================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.6)

min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

plt.xlabel(r"Observed $\log_{10}(R/R_\oplus)$")
plt.ylabel(r"Predicted $\log_{10}(R/R_\oplus)$")
plt.title("Predicted vs observed radius")
plt.tight_layout()
plt.savefig("plots/random_forest_predicted_vs_observed.png", dpi=300)
plt.show()

# ============================================================
# Predicted R(P_orb) relation
# ============================================================

# Create a grid in orbital period.
orbper_grid = np.linspace(
    df_ml["log_orbper"].min(),
    df_ml["log_orbper"].max(),
    500
)

# Fix the other variables to their median values.
median_mass = df_ml["log_mass"].median()
median_ecc = df_ml["pl_orbeccen"].median()
median_teff = df_ml["log_teff"].median()

# Build prediction dataframe.
X_orbper = pd.DataFrame({
    "log_mass": median_mass,
    "log_orbper": orbper_grid,
    "pl_orbeccen": median_ecc,
    "log_teff": median_teff
})

# Predict radius.
radius_pred = best_rf.predict(X_orbper)

# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))

# Observed planets
plt.scatter(
    df_ml["pl_orbper"],
    df_ml["pl_rade"],
    alpha=0.3,
    label="Observed planets"
)

# Predicted relation
plt.plot(
    10**orbper_grid,
    10**radius_pred,
    color="red",
    linewidth=3,
    label="Random Forest prediction"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$P_{\rm orb}$ [days]")
plt.ylabel(r"$R/R_\oplus$")

plt.title(r"Predicted $R(P_{\rm orb})$ relation")

plt.legend()
plt.tight_layout()

plt.savefig(
    "plots/predicted_radius_vs_orbital_period.png",
    dpi=300
)

plt.show()





# ============================================================
# 12. Predicted R(M) relation with tree-based uncertainty
# ============================================================

# Create a grid in planet mass.
mass_grid = np.linspace(
    df_ml["log_mass"].min(),
    df_ml["log_mass"].max(),
    500
)

# Fix the other physical variables to their median values.
median_orbper = df_ml["log_orbper"].median()
median_ecc = df_ml["pl_orbeccen"].median()
median_teff = df_ml["log_teff"].median()

# Build prediction dataframe.
X_mass = pd.DataFrame({
    "log_mass": mass_grid,
    "log_orbper": median_orbper,
    "pl_orbeccen": median_ecc,
    "log_teff": median_teff
})

# Predict with every tree separately.
tree_predictions_mass = np.array([
    tree.predict(X_mass)
    for tree in best_rf.estimators_
])

# Mean and standard deviation in log-radius.
radius_pred_mean = tree_predictions_mass.mean(axis=0)
radius_pred_std = tree_predictions_mass.std(axis=0)

# Convert from log space to physical units.
mass_physical = 10**mass_grid
radius_mean_physical = 10**radius_pred_mean

radius_lower_1sigma = 10**(radius_pred_mean - radius_pred_std)
radius_upper_1sigma = 10**(radius_pred_mean + radius_pred_std)

radius_lower_2sigma = 10**(radius_pred_mean - 2 * radius_pred_std)
radius_upper_2sigma = 10**(radius_pred_mean + 2 * radius_pred_std)


# ============================================================
# Plot
# ============================================================

plt.figure(figsize=(7, 5))

# Observed planets
plt.scatter(
    df_ml["pl_bmasse"],
    df_ml["pl_rade"],
    alpha=0.3,
    label="Observed planets"
)

# 95% tree-spread interval
plt.fill_between(
    mass_physical,
    radius_lower_2sigma,
    radius_upper_2sigma,
    alpha=0.15,
    label=r"Approx. $2\sigma$ tree spread"
)

# 68% tree-spread interval
plt.fill_between(
    mass_physical,
    radius_lower_1sigma,
    radius_upper_1sigma,
    alpha=0.25,
    label=r"Approx. $1\sigma$ tree spread"
)

# Mean prediction
plt.plot(
    mass_physical,
    radius_mean_physical,
    linewidth=3,
    label="Random Forest prediction"
)

plt.xscale("log")
plt.yscale("log")

plt.xlabel(r"$M/M_\oplus$")
plt.ylabel(r"$R/R_\oplus$")

plt.title(r"Predicted mass-radius relation with tree uncertainty")

plt.legend()
plt.tight_layout()

plt.savefig(
    "plots/predicted_mass_radius_with_uncertainty.png",
    dpi=300
)

plt.show()