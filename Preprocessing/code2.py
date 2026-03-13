import pandas as pd

# ===============================
# Step 2: Time-Based Feature Extraction
# ===============================

# Load Step 1 output
input_path = "step1_datetime_processed.csv"
df = pd.read_csv(input_path, parse_dates=["time"])

# ---- Extract time-based features ----
df["hour"] = df["time"].dt.hour
df["day"] = df["time"].dt.day
df["month"] = df["time"].dt.month
df["day_of_week"] = df["time"].dt.dayofweek  # Monday=0, Sunday=6

# Weekend indicator
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# ---- Save Step 2 output ----
output_path = "step2_time_features.csv"
df.to_csv(output_path, index=False)

# Verification
print("Step 2 completed successfully")
print(df.head())
print(df.info())
