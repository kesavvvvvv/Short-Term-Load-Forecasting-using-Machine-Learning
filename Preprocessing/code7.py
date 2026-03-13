import pandas as pd

# ===============================
# Step 7: Time-Aware Train–Test Split
# ===============================

# Load cleaned dataset
input_path = "step6_cleaned_data.csv"
df = pd.read_csv(input_path, parse_dates=["time"])

# ---- Define target and features ----
target = "load"

# Drop time and target from features
X = df.drop(columns=["time", target])
y = df[target]

# ---- Time-based split (80% train, 20% test) ----
split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# ---- Save split datasets ----
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)

# Verification
print("Step 7 completed successfully")
print("Train samples:", X_train.shape[0])
print("Test samples :", X_test.shape[0])
print("Train features:", X_train.shape[1])
