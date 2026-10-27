"""
generate_scripts.py
--------------------
Generates 42 standalone, runnable Python scripts:
    7 algorithms x 2 encodings x 3 scaling methods = 42 combinations

Organized as:
    combinations/
        01_LinearRegression/
            LinearRegression__OHE__StandardScaler.py
            LinearRegression__OHE__MinMaxScaler.py
            LinearRegression__OHE__NoScaling.py
            LinearRegression__LabelEncoding__StandardScaler.py
            LinearRegression__LabelEncoding__MinMaxScaler.py
            LinearRegression__LabelEncoding__NoScaling.py
        02_Ridge/
            ... (6 files)
        ...
        07_GradientBoostingRegressor/
            ... (6 files)

Every script is fully self-contained (loads data, encodes, splits, scales,
trains, evaluates, saves its own model, and appends one row to a shared
results_comparison.csv at the project root) so any single file can be run
on its own, in any order.

Run this once to (re)generate all 42 files:
    python generate_scripts.py
"""

import os

ALGORITHMS = [
    ("LinearRegression",
     "from sklearn.linear_model import LinearRegression",
     "LinearRegression()"),
    ("Ridge",
     "from sklearn.linear_model import Ridge",
     "Ridge(alpha=1.0, random_state=42)"),
    ("Lasso",
     "from sklearn.linear_model import Lasso",
     "Lasso(alpha=0.01, max_iter=10000, random_state=42)"),
    ("SVR",
     "from sklearn.svm import SVR",
     "SVR(kernel='rbf', C=1.0, epsilon=0.1)"),
    ("RandomForestRegressor",
     "from sklearn.ensemble import RandomForestRegressor",
     "RandomForestRegressor(n_estimators=200, random_state=42)"),
    ("MLPRegressor",
     "from sklearn.neural_network import MLPRegressor",
     "MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=3000, random_state=42)"),
    ("GradientBoostingRegressor",
     "from sklearn.ensemble import GradientBoostingRegressor",
     "GradientBoostingRegressor(n_estimators=200, random_state=42)"),
]

ENCODINGS = ["OHE", "LabelEncoding"]
SCALINGS = ["StandardScaler", "MinMaxScaler", "NoScaling"]

CATEGORICAL_COLS = [
    "Construction_Materials",
    "Housing_Typology",
    "Land_Value_Grading",
    "Type_of_Nearest_Road",
]

NUMERIC_COLS = [
    "Number_of_Rooms",
    "Site_Area_sqm",
    "Built_Area_sqm",
    "Property_Years",
    "Proximity_to_CBD_km",
    "Proximity_to_Bus_Station_km",
    "Proximity_to_Schools_km",
]

TARGET = "Price_ETB"


def encoding_block(encoding):
    if encoding == "OHE":
        return f"""
# --- Encoding: One-Hot Encoding ---
# Converts each categorical column into multiple 0/1 dummy columns.
# drop_first=True avoids the "dummy variable trap" (perfect multicollinearity).
categorical_cols = {CATEGORICAL_COLS!r}
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

numeric_cols = {NUMERIC_COLS!r}
dummy_cols = [c for c in df_encoded.columns if c not in numeric_cols + ["{TARGET}"]]
feature_cols = numeric_cols + dummy_cols

X = df_encoded[feature_cols]
y = df_encoded["{TARGET}"]
"""
    else:  # LabelEncoding
        return f"""
# --- Encoding: Label Encoding ---
# Converts each categorical column's text values into integer codes (0,1,2,...).
categorical_cols = {CATEGORICAL_COLS!r}
encoders = {{}}
for col in categorical_cols:
    le = LabelEncoder()
    df[col + "_encoded"] = le.fit_transform(df[col])
    encoders[col] = le

numeric_cols = {NUMERIC_COLS!r}
feature_cols = numeric_cols + [c + "_encoded" for c in categorical_cols]

X = df[feature_cols]
y = df["{TARGET}"]
"""


def scaling_block(scaling):
    if scaling == "StandardScaler":
        return """
# --- Scaling: StandardScaler ---
# Rescales every feature to mean 0, standard deviation 1.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
    elif scaling == "MinMaxScaler":
        return """
# --- Scaling: MinMaxScaler ---
# Rescales every feature into the range [0, 1].
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
"""
    else:  # NoScaling
        return """
# --- Scaling: None ---
# Features are used exactly as they are, with no rescaling at all.
scaler = None
X_train_scaled = X_train.values
X_test_scaled = X_test.values
"""


def build_script(algo_name, algo_import, algo_init, encoding, scaling, index):
    scaler_import = {
        "StandardScaler": "from sklearn.preprocessing import StandardScaler",
        "MinMaxScaler": "from sklearn.preprocessing import MinMaxScaler",
        "NoScaling": "",
    }[scaling]

    encoding_import = (
        "" if encoding == "OHE" else
        "from sklearn.preprocessing import LabelEncoder"
    )

    combo_label = f"{algo_name} + {encoding} + {scaling}"

    script = f'''"""
{combo_label}
{"=" * len(combo_label)}
Combination {index}/42 in the algorithm x encoding x scaling comparison.

Trains a {algo_name} model on houses_improved_data.csv using {encoding}
for categorical features and {scaling} for numeric scaling, evaluates it,
saves the trained model, and appends its results to results_comparison.csv
(shared across all 42 combinations) so you can compare them afterward.

Run with:
    python "{algo_name}__{encoding}__{scaling}.py"
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
{encoding_import}
{scaler_import}
{algo_import}
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

ALGORITHM = "{algo_name}"
ENCODING = "{encoding}"
SCALING = "{scaling}"

print(f"Running: {{ALGORITHM}} + {{ENCODING}} + {{SCALING}}")

# ------------------------------------------------------------------
# 1. Load data
# ------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
{encoding_block(encoding)}
print(f"Feature matrix shape: {{X.shape}}")

# ------------------------------------------------------------------
# 2. Train/test split (same split settings across all 42 combinations
#    for a fair comparison)
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
{scaling_block(scaling)}
# ------------------------------------------------------------------
# 3. Train the model
# ------------------------------------------------------------------
model = {algo_init}
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

print(f"Train RMSE: {{train_rmse:,.2f}}   Test RMSE: {{test_rmse:,.2f}}")
print(f"Train MAE:  {{train_mae:,.2f}}   Test MAE:  {{test_mae:,.2f}}")
print(f"Train R2:   {{train_r2:.4f}}       Test R2:   {{test_r2:.4f}}")

# ------------------------------------------------------------------
# 5. Save the trained model (+ scaler/encoders if used)
# ------------------------------------------------------------------
model_filename = f"{{ALGORITHM}}__{{ENCODING}}__{{SCALING}}.pkl"
joblib.dump(model, os.path.join(MODEL_DIR, model_filename))

# ------------------------------------------------------------------
# 6. Append this run's results to the shared comparison file
# ------------------------------------------------------------------
result_row = pd.DataFrame([{{
    "Algorithm": ALGORITHM,
    "Encoding": ENCODING,
    "Scaling": SCALING,
    "Train_RMSE": train_rmse,
    "Test_RMSE": test_rmse,
    "Train_MAE": train_mae,
    "Test_MAE": test_mae,
    "Train_R2": train_r2,
    "Test_R2": test_r2,
}}])

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
print(f"Result appended to {{RESULTS_PATH}}")
'''
    return script


def main():
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combinations")
    os.makedirs(base_dir, exist_ok=True)

    count = 0
    for i, (algo_name, algo_import, algo_init) in enumerate(ALGORITHMS, start=1):
        algo_folder = os.path.join(base_dir, f"{i:02d}_{algo_name}")
        os.makedirs(algo_folder, exist_ok=True)

        for encoding in ENCODINGS:
            for scaling in SCALINGS:
                count += 1
                script_text = build_script(
                    algo_name, algo_import, algo_init, encoding, scaling, count
                )
                filename = f"{algo_name}__{encoding}__{scaling}.py"
                filepath = os.path.join(algo_folder, filename)
                with open(filepath, "w") as f:
                    f.write(script_text)

    print(f"Generated {count} scripts across {len(ALGORITHMS)} algorithm folders.")


if __name__ == "__main__":
    main()
