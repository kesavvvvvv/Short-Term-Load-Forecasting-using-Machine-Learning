import pandas as pd

# ===============================
# Step 1: Data Loading & Date-Time Processing
# ===============================

# Load semicolon-separated CSV
input_path = "malaysia_all_data_for_paper.csv"
df = pd.read_csv(input_path, sep=";")

# Convert time column (MM/DD/YY HH:MM AM/PM)
df["time"] = pd.to_datetime(
    df["time"],
    format="%m/%d/%y %I:%M %p"
)

# Sort chronologically
df = df.sort_values("time").reset_index(drop=True)

# Save processed dataset
output_path = "step1_datetime_processed.csv"
df.to_csv(output_path, index=False)

# Verification
print("Step 1 completed successfully")
print(df.head())
print(df.info())
