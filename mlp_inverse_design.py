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
# TARGETS
# =========================
TARGET_APOGEE = 9300
TARGET_VELOCITY = 33


# =========================
# WEIGHTS
# =========================
W_APOGEE = 0.25
W_VELOCITY = 0.50
W_THRUST = 0.25


# =========================
# FIXED PARAMETERS
# =========================
EMPTY_MASS = 20.0
MOTOR_MASS = 20.0


# =========================
# CONSTRAINTS
# =========================
MIN_THRUST = 2000
MAX_THRUST = 4000

MIN_BURN = 1.5
MAX_BURN = 20.0


# =========================
# SEARCH SPACE
# =========================
thrust_range = np.linspace(MIN_THRUST, MAX_THRUST, 80)
burn_range = np.linspace(MIN_BURN, MAX_BURN, 80)

results = []

start_time = time.time()


# =========================
# SEARCH LOOP
# =========================
for thrust in thrust_range:
    for burn in burn_range:

        X = np.array([[burn, thrust, EMPTY_MASS, MOTOR_MASS]])
        X_scaled = scaler_X.transform(X)

        y_scaled = model(X_scaled)
        y = scaler_y.inverse_transform(y_scaled)

        apogee = y[0, 0]
        velocity = y[0, 1]

        # =========================
        # PERCENTAGE ERRORS
        # =========================
        apogee_error = (
            abs(apogee - TARGET_APOGEE)
            / TARGET_APOGEE
            * 100
        )

        velocity_error = (
            abs(velocity - TARGET_VELOCITY)
            / TARGET_VELOCITY
            * 100
        )

        # =========================
        # THRUST MINIMIZATION TERM
        # =========================
        thrust_penalty = (
            thrust / MAX_THRUST
        ) * 100

        # =========================
        # FINAL MULTI-OBJECTIVE SCORE
        # =========================
        error = (
            (W_APOGEE * apogee_error)
            + (W_VELOCITY * velocity_error)
            + (W_THRUST * thrust_penalty)
        )

        results.append((
            error,
            thrust,
            burn,
            apogee,
            velocity,
            apogee_error,
            velocity_error,
            thrust_penalty
        ))


end_time = time.time()


# =========================
# SORT RESULTS
# =========================
results.sort(key=lambda x: x[0])


# =========================
# TOP 10
# =========================
print("\n===== TOP 10 SOLUTIONS =====")

for i in range(10):

    (
        e,
        t,
        b,
        a,
        v,
        ae,
        ve,
        tp
    ) = results[i]

    print(f"\n#{i+1}")
    print(f"Thrust: {t:.1f} N")
    print(f"Burn: {b:.2f} s")

    print(f"Apogee: {a:.1f} m")
    print(f"Velocity: {v:.2f} m/s")

    print(f"Apogee Error: {ae:.2f}%")
    print(f"Velocity Error: {ve:.2f}%")

    print(f"Thrust Penalty: {tp:.2f}%")

    print(f"Final Score: {e:.2f}")


# =========================
# TIME
# =========================
print("\n===== SEARCH TIME =====")
print(f"Total time: {end_time - start_time:.2f} seconds")

print(
    f"Weights -> "
    f"Apogee: {W_APOGEE}, "
    f"Velocity: {W_VELOCITY}, "
    f"Thrust: {W_THRUST}"
)

print(f"Max thrust constraint: {MAX_THRUST} N")


# =========================
# BEST SOLUTION
# =========================
best = results[0]

print("\n===== BEST CONFIGURATION =====")

print(f"Thrust (N):         {best[1]:.2f}")
print(f"Burn time (s):      {best[2]:.2f}")

print(f"Predicted apogee:   {best[3]:.2f} m")
print(f"Rail exit velocity: {best[4]:.2f} m/s")

print(f"Final score:        {best[0]:.2f}")