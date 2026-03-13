import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# ===============================
# Step 8: Feature Scaling (SVR only)
# ===============================

# Load train-test splits
X_train = pd.read_csv("X_train.csv")
X_test  = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv")
y_test  = pd.read_csv("y_test.csv")

# ---- Initialize scaler ----
scaler = StandardScaler()

# ---- Fit ONLY on training data ----
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ---- Convert back to DataFrame ----
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled  = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# ---- Save scaled datasets ----
X_train_scaled.to_csv("X_train_scaled_svr.csv", index=False)
X_test_scaled.to_csv("X_test_scaled_svr.csv", index=False)

# ---- Save scaler for reproducibility ----
joblib.dump(scaler, "svr_scaler.pkl")

# Verification
print("Step 8 completed successfully")
print("Scaled train shape:", X_train_scaled.shape)
print("Scaled test shape :", X_test_scaled.shape)
