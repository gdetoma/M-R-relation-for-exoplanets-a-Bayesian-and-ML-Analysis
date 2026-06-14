# ============================================================
# Neural Network for the Exoplanet Mass-Radius Relation
# ============================================================

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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

# Now drop invalid rows
df = df.dropna(subset=features + [target, "log_radius_err"]).copy()
df = df[df["log_radius_err"] > 0].copy()

X = df[features].values
y = df[[target]].values
y_err = df[["log_radius_err"]].values


# ============================================================
# 2. Train-test split
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
# 3. Standardize the data
# ============================================================

# Neural networks usually train better when inputs and outputs
# have mean 0 and standard deviation 1.

x_scaler = StandardScaler()
y_scaler = StandardScaler()

X_train_scaled = x_scaler.fit_transform(X_train)
X_val_scaled = x_scaler.transform(X_val)
X_test_scaled = x_scaler.transform(X_test)

y_train_scaled = y_scaler.fit_transform(y_train)
y_val_scaled = y_scaler.transform(y_val)
y_test_scaled = y_scaler.transform(y_test)

yerr_train_scaled = yerr_train / y_scaler.scale_[0]
yerr_val_scaled = yerr_val / y_scaler.scale_[0]
yerr_test_scaled = yerr_test / y_scaler.scale_[0]

error_floor = 0.03
error_floor_scaled = error_floor / y_scaler.scale_[0]

yerr_train_scaled = np.maximum(yerr_train_scaled, error_floor_scaled)
yerr_val_scaled = np.maximum(yerr_val_scaled, error_floor_scaled)
yerr_test_scaled = np.maximum(yerr_test_scaled, error_floor_scaled)


# Convert numpy arrays to PyTorch tensors.
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32)
yerr_val_tensor = torch.tensor(yerr_val_scaled, dtype=torch.float32)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32)

yerr_train_tensor = torch.tensor(yerr_train_scaled, dtype=torch.float32)
yerr_test_tensor = torch.tensor(yerr_test_scaled, dtype=torch.float32)


# ============================================================
# 4. Define the neural network
# ============================================================

class MassRadiusNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.network(x)


model = MassRadiusNN()


# ============================================================
# 5. Define loss function and optimizer
# ============================================================

def weighted_mse_loss(y_pred, y_true, y_err):
    return torch.mean(((y_pred - y_true) / y_err) ** 2)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ============================================================
# 6. Train the model with validation and early stopping
# ============================================================

n_epochs = 3000

train_losses = []
val_losses = []

best_val_loss = np.inf
patience = 200
patience_counter = 0

best_model_state = None

for epoch in range(n_epochs):

    # -----------------------
    # Training step
    # -----------------------
    model.train()

    y_pred_train = model(X_train_tensor)

    train_loss = weighted_mse_loss(
        y_pred_train,
        y_train_tensor,
        yerr_train_tensor
    )

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()

    train_losses.append(train_loss.item())

    # -----------------------
    # Validation step
    # -----------------------
    model.eval()

    with torch.no_grad():
        y_pred_val = model(X_val_tensor)

        val_loss = weighted_mse_loss(
            y_pred_val,
            y_val_tensor,
            yerr_val_tensor
        )

    val_losses.append(val_loss.item())

    # -----------------------
    # Early stopping
    # -----------------------
    if val_loss.item() < best_val_loss:
        best_val_loss = val_loss.item()
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1

    if epoch % 300 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"Train loss = {train_loss.item():.6f} | "
            f"Val loss = {val_loss.item():.6f}"
        )

    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# Load best model
model.load_state_dict(best_model_state)


# ============================================================
# 7. Evaluate the model
# ============================================================

model.eval()

with torch.no_grad():
    y_pred_test_scaled = model(X_test_tensor).numpy()

# Convert predictions back to original log-radius scale.
y_pred_test = y_scaler.inverse_transform(y_pred_test_scaled)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
mae = mean_absolute_error(y_test, y_pred_test)
r2 = r2_score(y_test, y_pred_test)

print("\nNeural Network performance on test set")
print("--------------------------------------")
print("RMSE:", rmse)
print("MAE:", mae)
print("R²:", r2)


# ============================================================
# 8. Plot training loss
# ============================================================

os.makedirs("plots", exist_ok=True)

plt.figure(figsize=(7, 5))
plt.plot(train_losses, label="Training loss")
plt.plot(val_losses, label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Weighted MSE loss")
plt.title("Neural Network training and validation loss")
plt.legend()
plt.tight_layout()
plt.savefig("plots/nn_training_validation_loss.png", dpi=300)
plt.show()


# ============================================================
# 9. Plot learned mass-radius relation
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
X_grid_tensor = torch.tensor(X_grid_scaled, dtype=torch.float32)

model.eval()

with torch.no_grad():
    radius_pred_scaled = model(X_grid_tensor).numpy()

radius_pred_log = y_scaler.inverse_transform(radius_pred_scaled).flatten()

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
    label="NN at median Teff, period, eccentricity",
    color="red"
)

plt.xlabel(r"$\log_{10}(M/M_\oplus)$")
plt.ylabel(r"$\log_{10}(R/R_\oplus)$")
plt.legend()
plt.tight_layout()
plt.savefig("plots/neural_network_mass_radius_multifeature.png", dpi=300)
plt.show()