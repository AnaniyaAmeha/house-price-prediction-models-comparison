"""
Ridge + LabelEncoding + NoScaling
=================================
Combination 12/42 in the algorithm x encoding x scaling comparison.

Trains a Ridge model on houses_improved_data.csv using LabelEncoding
for categorical features and NoScaling for numeric scaling, evaluates it,
saves the trained model, and appends its results to results_comparison.csv
(shared across all 42 combinations) so you can compare them afterward.

Run with:
    python "Ridge__LabelEncoding__NoScaling.py"
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ------------------------------------------------------------------
# Paths: this script lives in combinations/<algo_folder>/, so the
# project root (where the dataset and shared results file live) is
# two directories up.
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_PATH = os.path.join(ROOT_DIR, "houses_improved_data.csv")
RESULTS_PATH = os.path.join(ROOT_DIR, "results_comparison.csv")
MODEL_DIR = os.path.join(ROOT_DIR, "saved_models")
os.makedirs(MODEL_DIR, exist_ok=True)

ALGORITHM = "Ridge"
ENCODING = "LabelEncoding"
SCALING = "NoScaling"

print(f"Running: {ALGORITHM} + {ENCODING} + {SCALING}")

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# --- Encoding: Label Encoding ---
# Converts each categorical column's text values into integer codes (0,1,2,...).
categorical_cols = ['Construction_Materials', 'Housing_Typology', 'Land_Value_Grading', 'Type_of_Nearest_Road']
encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    encoders[col] = le

numeric_cols = ['Number_of_Rooms', 'Site_Area_sqm', 'Built_Area_sqm', 'Property_Years', 'Proximity_to_CBD_km', 'Proximity_to_Bus_Station_km', 'Proximity_to_Schools_km']
feature_cols = numeric_cols + [c + "_encoded" for c in categorical_cols]

X = df[feature_cols]
y = df["Price_ETB"]

print(f"Feature matrix shape: {X.shape}")

# ------------------------------------------------------------------
# 2. Train/test split (same split settings across all 42 combinations
#    for a fair comparison)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Scaling: None ---
# Features are used exactly as they are, with no rescaling at all.
scaler = None
X_train_scaled = X_train.values
X_test_scaled = X_test.values

# ------------------------------------------------------------------
# 3. Train the model
# ------------------------------------------------------------------
model = Ridge(alpha=1.0, random_state=42)
model.fit(X_train_scaled, y_train)

# ------------------------------------------------------------------
# 4. Evaluate
# ------------------------------------------------------------------
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

print(f"Train RMSE: {train_rmse:,.2f}   Test RMSE: {test_rmse:,.2f}")
print(f"Train MAE:  {train_mae:,.2f}   Test MAE:  {test_mae:,.2f}")
print(f"Train R2:   {train_r2:.4f}       Test R2:   {test_r2:.4f}")

# ------------------------------------------------------------------
# 5. Save the trained model (+ scaler/encoders if used)
# ------------------------------------------------------------------
model_filename = f"{ALGORITHM}__{ENCODING}__{SCALING}.pkl"
joblib.dump(model, os.path.join(MODEL_DIR, model_filename))

# ------------------------------------------------------------------
# 6. Append this run's results to the shared comparison file
# ------------------------------------------------------------------
result_row = pd.DataFrame([{
    "Algorithm": ALGORITHM,
    "Encoding": ENCODING,
    "Scaling": SCALING,
    "Train_RMSE": train_rmse,
    "Test_RMSE": test_rmse,
    "Train_MAE": train_mae,
    "Test_MAE": test_mae,
    "Train_R2": train_r2,
    "Test_R2": test_r2,
}])

if os.path.exists(RESULTS_PATH):
    existing = pd.read_csv(RESULTS_PATH)
    # Replace a prior run of this exact combination, if present, instead
    # of appending a duplicate row.
    existing = existing[
        ~((existing["Algorithm"] == ALGORITHM) &
          (existing["Encoding"] == ENCODING) &
          (existing["Scaling"] == SCALING))
    ]
    combined = pd.concat([existing, result_row], ignore_index=True)
else:
    combined = result_row

combined.to_csv(RESULTS_PATH, index=False)
print(f"Result appended to {RESULTS_PATH}")
