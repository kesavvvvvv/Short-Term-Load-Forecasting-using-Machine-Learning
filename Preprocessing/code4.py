import pandas as pd

# ===============================
# Step 4: Rolling Statistics
# ===============================

# Load Step 3 output
input_path = "step3_lag_features.csv"
df = pd.read_csv(input_path, parse_dates=["time"])

# ---- Rolling mean features on load ----
df["rolling_mean_3"] = df["load"].rolling(window=3).mean()
df["rolling_mean_6"] = df["load"].rolling(window=6).mean()
df["rolling_mean_24"] = df["load"].rolling(window=24).mean()

# ---- Rolling standard deviation (volatility) ----
df["rolling_std_24"] = df["load"].rolling(window=24).std()

# ---- Save Step 4 output ----
output_path = "step4_rolling_features.csv"
df.to_csv(output_path, index=False)

# Verification
print("Step 4 completed successfully")
print(df.head(30))
print(df.info())
