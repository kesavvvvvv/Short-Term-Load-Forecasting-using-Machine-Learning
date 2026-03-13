import pandas as pd

# ===============================
# Step 6: Data Cleaning (NaN Removal)
# ===============================

# Load Step 5 output
input_path = "step5_cyclical_features.csv"
df = pd.read_csv(input_path, parse_dates=["time"])

# ---- Drop rows with NaN values ----
df_clean = df.dropna().reset_index(drop=True)

# ---- Save cleaned dataset ----
output_path = "step6_cleaned_data.csv"
df_clean.to_csv(output_path, index=False)

# Verification
print("Step 6 completed successfully")
print("Original rows:", df.shape[0])
print("After cleaning:", df_clean.shape[0])
print(df_clean.head())
print(df_clean.info())
