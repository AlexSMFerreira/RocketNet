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
# TARGET RANGES
# =========================
MIN_APOGEE = 9800
MAX_APOGEE = 10000

MIN_VELOCITY = 30


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

MIN_BURN = 1.0
MAX_BURN = 20.0


# =========================
# SEARCH SPACE
# =========================
thrust_range = np.linspace(MIN_THRUST, MAX_THRUST, 80)
burn_range = np.linspace(MIN_BURN, MAX_BURN, 80)

valid_results = []

start_time = time.time()


# =========================
# SEARCH LOOP
# =========================
for thrust in thrust_range:
    for burn in burn_range:

        X = np.array([
            [burn, thrust, EMPTY_MASS, MOTOR_MASS]
        ])

        X_scaled = scaler_X.transform(X)

        # FAST INFERENCE
        y_scaled = model(X_scaled, training=False)

        y = scaler_y.inverse_transform(y_scaled)

        apogee = y[0, 0]
        velocity = y[0, 1]

        # =========================
        # RANGE FILTER
        # =========================
        if (
            MIN_APOGEE <= apogee <= MAX_APOGEE
            and velocity >= MIN_VELOCITY
        ):

            # =========================
            # RANKING SCORE
            # LOWER = BETTER
            # =========================

            # Prefer:
            # - higher velocity
            
            score = MIN_VELOCITY - velocity

            valid_results.append((
                score,
                thrust,
                burn,
                apogee,
                velocity
            ))


end_time = time.time()


# =========================
# SORT RESULTS
# =========================
valid_results.sort(key=lambda x: x[0])


# =========================
# RESULTS
# =========================
print("\n===== VALID CONFIGURATIONS =====")

if len(valid_results) == 0:

    print("\nNo solutions found.")

else:

    max_results = min(10, len(valid_results))

    for i in range(max_results):

        (
            score,
            thrust,
            burn,
            apogee,
            velocity
        ) = valid_results[i]

        print(f"\n#{i+1}")

        print(f"Thrust: {thrust:.1f} N")
        print(f"Burn time: {burn:.2f} s")

        print(f"Apogee: {apogee:.2f} m")
        print(f"Rail Exit Velocity: {velocity:.2f} m/s")

        print(f"Score: {score:.2f}")


# =========================
# TIMING
# =========================
total_time = end_time - start_time

num_predictions = (
    len(thrust_range)
    * len(burn_range)
)

avg_time = total_time / num_predictions

print("\n===== SEARCH TIME =====")

print(f"Total time: {total_time:.2f} s")

print(f"Predictions evaluated: {num_predictions}")

print(f"Average inference time: {avg_time*1000:.3f} ms")


# =========================
# SUMMARY
# =========================
print("\n===== SEARCH CONSTRAINTS =====")

print(
    f"Apogee range: "
    f"{MIN_APOGEE} - {MAX_APOGEE} m"
)

print(
    f"Minimum velocity: "
    f"{MIN_VELOCITY} m/s"
)

print(
    f"Solutions found: "
    f"{len(valid_results)}"
)


# =========================
# BEST SOLUTION
# =========================
if len(valid_results) > 0:

    best = valid_results[0]

    print("\n===== BEST CONFIGURATION =====")

    print(f"Thrust (N):         {best[1]:.2f}")
    print(f"Burn time (s):      {best[2]:.2f}")

    print(f"Predicted apogee:   {best[3]:.2f} m")
    print(f"Rail exit velocity: {best[4]:.2f} m/s")

    print(f"Score:              {best[0]:.2f}")

else:

    print("\nNo valid solution found.")