import pandas as pd
import numpy as np

# ===============================
# Step 5: Cyclical Encoding
# ===============================

# Load Step 4 output
input_path = "step4_rolling_features.csv"
df = pd.read_csv(input_path, parse_dates=["time"])

# ---- Cyclical encoding for hour of day ----
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

# ---- Cyclical encoding for day of week ----
df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

# ---- Save Step 5 output ----
output_path = "step5_cyclical_features.csv"
df.to_csv(output_path, index=False)

# Verification
print("Step 5 completed successfully")
print(df[["hour", "hour_sin", "hour_cos", "day_of_week", "dow_sin", "dow_cos"]].head())
print(df.info())
