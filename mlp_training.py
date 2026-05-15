import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

import joblib
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# =========================
# 1. LOAD DATA
# =========================
df = pd.read_csv("./data/rocketpy_dataset1.csv")

X = df[[
    "tburn_s",
    "thrust_N",
    "empty_rocket_mass_kg",
    "motor_dry_mass_without_tank_kg"
]].values

y = df[[
    "apogee_altitude_AGL_m",
    "rail_exit_velocity_mps"
]].values


# =========================
# 2. TRAIN / TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# =========================
# 3. SCALING
# =========================
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train = scaler_X.fit_transform(X_train)
X_test = scaler_X.transform(X_test)

y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)


# =========================
# 4. MODEL (MLP)
# =========================
model = Sequential([
    Dense(32, activation='relu', input_shape=(4,)),
    Dense(16, activation='relu'),
    Dense(2)  # 2 outputs
])

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)


# =========================
# 5. EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)


# =========================
# 6. TRAINING
# =========================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)


# =========================
# 7. EVALUATION
# =========================
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)

print("\n===== TEST RESULTS (SCALED) =====")
print("Test Loss (MSE):", test_loss)
print("Test MAE:", test_mae)


# =========================
# 8. PREDICTIONS
# =========================
y_pred = model.predict(X_test)

# inverse scaling
y_pred_real = scaler_y.inverse_transform(y_pred)
y_test_real = scaler_y.inverse_transform(y_test)


# =========================
# 9. METRICS (REAL SCALE)
# =========================
print("\n===== FINAL METRICS =====")

print("Apogee R²:",
      r2_score(y_test_real[:, 0], y_pred_real[:, 0]))

print("Velocity R²:",
      r2_score(y_test_real[:, 1], y_pred_real[:, 1]))

print("Apogee MAE:",
      mean_absolute_error(y_test_real[:, 0], y_pred_real[:, 0]))

print("Velocity MAE:",
      mean_absolute_error(y_test_real[:, 1], y_pred_real[:, 1]))


# =========================
# 10. PARITY PLOTS
# =========================
plt.figure()
plt.scatter(y_test_real[:, 0], y_pred_real[:, 0])
plt.xlabel("True Apogee")
plt.ylabel("Predicted Apogee")
plt.title("Apogee Parity Plot")
plt.plot([y_test_real[:, 0].min(), y_test_real[:, 0].max()],
         [y_test_real[:, 0].min(), y_test_real[:, 0].max()],
         'r--')
plt.savefig("apogee_parity.png")

plt.figure()
plt.scatter(y_test_real[:, 1], y_pred_real[:, 1])
plt.xlabel("True Rail Exit Velocity")
plt.ylabel("Predicted Rail Exit Velocity")
plt.title("Velocity Parity Plot")
plt.plot([y_test_real[:, 1].min(), y_test_real[:, 1].max()],
         [y_test_real[:, 1].min(), y_test_real[:, 1].max()],
         'r--')
plt.savefig("velocity_parity.png")

# =========================
# 11. SAVE MODEL + SCALERS
# =========================
model.save("mlp_rocket_surrogate.keras")

joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

print("\nModel and scalers saved successfully.")