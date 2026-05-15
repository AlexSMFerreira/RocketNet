import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

# =========================
# LOAD MODEL + SCALERS
# =========================
model = load_model("mlp_rocket_surrogate.keras")

scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")


# =========================
# INPUT FUNCTION
# =========================
print("\n=== Rocket Surrogate Model Prediction ===")

tburn_s = float(input("Enter burn time (tburn_s): "))
thrust_N = float(input("Enter thrust (N): "))
empty_mass = float(input("Enter empty rocket mass (kg): "))
motor_mass = float(input("Enter motor dry mass (kg): "))


# =========================
# PREPARE INPUT
# =========================
X_input = np.array([[tburn_s, thrust_N, empty_mass, motor_mass]])

X_scaled = scaler_X.transform(X_input)


# =========================
# PREDICTION
# =========================

start = time.time()
y_scaled_pred = model(X_scaled, training=False)
end = time.time()

y_pred = scaler_y.inverse_transform(y_scaled_pred)

apogee = y_pred[0, 0]
velocity = y_pred[0, 1]


# =========================
# OUTPUT
# =========================
print("\n===== PREDICTION RESULT =====")
print(f"Apogee altitude (AGL): {apogee:.2f} m")
print(f"Rail exit velocity:   {velocity:.2f} m/s")
print(f"\nPrediction time: {end - start:.4f} seconds")