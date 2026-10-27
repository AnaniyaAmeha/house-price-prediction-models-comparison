# House Price Prediction — 42 Algorithm × Encoding × Scaling Comparison

## What this is
7 regression algorithms × 2 encoding methods × 3 scaling methods = **42 combinations**,
each trained and evaluated separately on `houses_improved_data.csv`, so you can compare
which combination predicts house price (`Price_ETB`) best.

## The 42 combinations

**Algorithms (7):** Linear Regression, Ridge, Lasso, SVR, Random Forest Regressor,
MLP Regressor, Gradient Boosting Regressor

**Encodings (2):** One-Hot Encoding (OHE), Label Encoding

**Scaling (3):** StandardScaler, MinMaxScaler, No Scaling

## Folder structure

```
combinations/
    01_LinearRegression/
        LinearRegression__OHE__StandardScaler.py
        LinearRegression__OHE__MinMaxScaler.py
        LinearRegression__OHE__NoScaling.py
        LinearRegression__LabelEncoding__StandardScaler.py
        LinearRegression__LabelEncoding__MinMaxScaler.py
        LinearRegression__LabelEncoding__NoScaling.py
    02_Ridge/            (same 6-file pattern)
    03_Lasso/            (same 6-file pattern)
    04_SVR/              (same 6-file pattern)
    05_RandomForestRegressor/   (same 6-file pattern)
    06_MLPRegressor/     (same 6-file pattern)
    07_GradientBoostingRegressor/   (same 6-file pattern)

houses_improved_data.csv     <- the dataset
generate_scripts.py          <- generates all 42 scripts above (already run for you)
compare_results.py           <- prints a ranked comparison table
results_comparison.csv       <- created/updated as you run the 42 scripts
saved_models/                <- created as you run the 42 scripts (one .pkl per combo)
```

Each of the 42 scripts is **fully self-contained**: it loads the data, encodes,
splits, scales, trains, evaluates, saves its own trained model into `saved_models/`,
and appends one row of results to the shared `results_comparison.csv` at the project
root. You can run any single script on its own, in any order, as many times as you like.

## How to run

1. Install dependencies:
   ```
   pip install pandas numpy scikit-learn joblib
   ```

2. Run any individual combination, e.g.:
   ```
   python combinations/01_LinearRegression/LinearRegression__OHE__StandardScaler.py
   ```
   This prints its own RMSE/MAE/R² and adds a row to `results_comparison.csv`.

3. To run **all 42** and build the full comparison table, from the project root run
   each script in turn (Windows example — adjust for Mac/Linux with a `.sh` loop if
   needed):
   ```
   for /r combinations %f in (*.py) do python "%f"
   ```
   (On Mac/Linux: `find combinations -name "*.py" -exec python3 {} \;`)

4. Once you've run some or all combinations, view the ranked comparison:
   ```
   python compare_results.py
   ```
   This prints:
   - The full table of all combinations run so far, ranked by Test R²
   - The single best combination overall
   - The best encoding+scaling choice for each algorithm
   - Average R² broken down by encoding, and by scaling

## Results from this run (all 42 executed)

**Best overall:** Gradient Boosting Regressor + One-Hot Encoding + StandardScaler
— Test R² ≈ 0.906, Test RMSE ≈ ETB 394,920

**Ranking pattern observed:**
1. Gradient Boosting Regressor (R² ≈ 0.90–0.91 across all 6 variants) — best and most
   consistent, barely affected by encoding/scaling choice.
2. MLP Regressor — best with OHE + StandardScaler (R² ≈ 0.88), but drops sharply
   without scaling (R² as low as ≈0.38–0.47) — neural networks are very sensitive to
   feature scale.
3. Random Forest Regressor (R² ≈ 0.83–0.85) — like Gradient Boosting, tree-based, so
   scaling barely matters, but encoding does (OHE outperforms Label Encoding here).
4. Linear Regression / Ridge / Lasso (R² ≈ 0.68–0.81) — all three land in almost the
   same place per encoding, since they're all fundamentally linear models; OHE clearly
   beats Label Encoding for these.
5. SVR — performed poorly in every combination (R² slightly negative), which points to
   its default hyperparameters (`C=1.0`, `epsilon=0.1`) not suiting this data, not a
   flaw in the setup — SVR generally needs hyperparameter tuning to perform well.

**General patterns worth including in your write-up:**
- **One-Hot Encoding beat Label Encoding on average** (mean R² 0.674 vs 0.600) —
  makes sense, since Label Encoding imposes a fake numeric order on categories that
  aren't actually ordered (e.g. "Concrete" isn't inherently "more" than "Mud&Wood").
- **Tree-based models (Random Forest, Gradient Boosting) barely care about scaling**
  — their R² is nearly identical whether you scale or not, because they split on
  raw feature values rather than computing distances or gradients directly from
  magnitude.
- **Scale-sensitive models (SVR, MLP, and to a lesser extent linear models) benefit
  more from StandardScaler** — confirmed by StandardScaler having the highest
  average R² (0.667) across all combinations, versus 0.622 for both MinMaxScaler and
  no scaling at all.
