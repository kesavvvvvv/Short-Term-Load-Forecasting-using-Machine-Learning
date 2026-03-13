import pandas as pd

# ===============================
# Step 3: Lag Feature Generation
# ===============================

# Load Step 2 output
input_path = "step2_time_features.csv"
df = pd.read_csv(input_path, parse_dates=["time"])

# ---- Load lag features (hours) ----
df["load_lag_1"] = df["load"].shift(1)
df["load_lag_2"] = df["load"].shift(2)
df["load_lag_3"] = df["load"].shift(3)
df["load_lag_24"] = df["load"].shift(24)
df["load_lag_48"] = df["load"].shift(48)
df["load_lag_168"] = df["load"].shift(168)

# ---- Temperature lag features ----
df["temp_lag_1"] = df["temperature"].shift(1)
df["temp_lag_24"] = df["temperature"].shift(24)

# ---- Save Step 3 output ----
output_path = "step3_lag_features.csv"
df.to_csv(output_path, index=False)

# Verification
print("Step 3 completed successfully")
print(df.head(30))
print(df.info())
