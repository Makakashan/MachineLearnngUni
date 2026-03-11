import csv
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ROOT / "iris_big.csv"
PLOT_PATH = OUT_DIR / "normalization_sepal_scatter.png"
SUMMARY_JSON = OUT_DIR / "task3_summary.json"

# Load data
sepal = []
labels = []
with open(INPUT_CSV, newline="") as f:
    reader = csv.reader(f)
    _ = next(reader)
    for row in reader:
        sepal.append([float(row[0]), float(row[1])])
        labels.append(row[4].strip().lower())

X = np.array(sepal, dtype=float)

# Normalizations
min_vals = X.min(axis=0)
max_vals = X.max(axis=0)
mean_vals = X.mean(axis=0)
std_vals = X.std(axis=0)

minmax = (X - min_vals) / (max_vals - min_vals)
zs = (X - mean_vals) / std_vals

# Stats
stats = {}
for name, data in [
    ("original", X),
    ("minmax", minmax),
    ("zscore", zs),
]:
    stats[name] = {
        "min": data.min(axis=0).tolist(),
        "max": data.max(axis=0).tolist(),
        "mean": data.mean(axis=0).tolist(),
        "std": data.std(axis=0).tolist(),
    }

# Plot
species_order = ["setosa", "versicolor", "virginica"]
colors = {"setosa": "#1f77b4", "versicolor": "#2ca02c", "virginica": "#d62728"}

fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=False, sharey=False)

for ax, (title, data) in zip(
    axes,
    [
        ("Original", X),
        ("Min-Max", minmax),
        ("Z-Score", zs),
    ],
):
    for sp in species_order:
        idxs = [i for i, label in enumerate(labels) if label == sp]
        ax.scatter(data[idxs, 0], data[idxs, 1], s=10, alpha=0.7, label=sp, c=colors[sp])
    ax.set_title(title)
    ax.set_xlabel("sepal length")
    ax.set_ylabel("sepal width")

# Single legend
handles, legend_labels = axes[0].get_legend_handles_labels()
fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False)
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig(PLOT_PATH, dpi=160)

with open(SUMMARY_JSON, "w") as f:
    json.dump({"stats": stats, "plot": str(PLOT_PATH)}, f, indent=2)

print("Task 3 summary:")
print(json.dumps({"stats": stats}, indent=2))
print(f"Plot saved to: {PLOT_PATH}")
