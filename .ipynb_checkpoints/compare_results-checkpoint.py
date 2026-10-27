"""
compare_results.py
-------------------
Reads results_comparison.csv (built up as you run the 42 combination
scripts) and prints a ranked comparison table, plus a couple of summary
breakdowns (best combination overall, best per algorithm, effect of
encoding/scaling on average).

Run any time after running some or all of the 42 combination scripts:
    python compare_results.py
"""

import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(SCRIPT_DIR, "results_comparison.csv")

if not os.path.exists(RESULTS_PATH):
    print("No results_comparison.csv found yet. Run some of the 42 "
          "combination scripts first (see combinations/ folder).")
    raise SystemExit

df = pd.read_csv(RESULTS_PATH)
pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)

print(f"Loaded {len(df)} / 42 combination results.\n")

print("=" * 70)
print("FULL COMPARISON (ranked by Test R2, best first)")
print("=" * 70)
ranked = df.sort_values("Test_R2", ascending=False).reset_index(drop=True)
ranked.index += 1
print(ranked[["Algorithm", "Encoding", "Scaling", "Test_RMSE", "Test_MAE", "Test_R2"]]
      .to_string())

print("\n" + "=" * 70)
print("BEST COMBINATION OVERALL")
print("=" * 70)
best = ranked.iloc[0]
print(f"{best['Algorithm']} + {best['Encoding']} + {best['Scaling']}")
print(f"Test R2 = {best['Test_R2']:.4f}   Test RMSE = {best['Test_RMSE']:,.2f}")

print("\n" + "=" * 70)
print("BEST ENCODING + SCALING PER ALGORITHM")
print("=" * 70)
best_per_algo = (
    df.sort_values("Test_R2", ascending=False)
    .groupby("Algorithm")
    .first()
    .sort_values("Test_R2", ascending=False)
    [["Encoding", "Scaling", "Test_RMSE", "Test_R2"]]
)
print(best_per_algo.to_string())

print("\n" + "=" * 70)
print("AVERAGE TEST R2 BY ENCODING (across all algorithms/scalings)")
print("=" * 70)
print(df.groupby("Encoding")["Test_R2"].mean().sort_values(ascending=False).to_string())

print("\n" + "=" * 70)
print("AVERAGE TEST R2 BY SCALING (across all algorithms/encodings)")
print("=" * 70)
print(df.groupby("Scaling")["Test_R2"].mean().sort_values(ascending=False).to_string())
