import csv
import json
from pathlib import Path
from typing import cast

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ROOT / "iris_big.csv"
PCA_PLOT = OUT_DIR / "pca_scatter.png"
PCA_CSV = OUT_DIR / "iris_pca.csv"
SUMMARY_JSON = OUT_DIR / "task2_summary.json"

# Load data
X = []
labels = []
with open(INPUT_CSV, newline="") as f:
    reader = csv.reader(f)
    _ = next(reader)
    for row in reader:
        X.append([float(row[i]) for i in range(4)])
        labels.append(row[4].strip().lower())

X = np.array(X, dtype=float)

# PCA: center data, compute covariance, find eigenvectors
X_mean = X.mean(axis=0)
X_centered = X - X_mean
cov = np.cov(X_centered, rowvar=False)

eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]  # Sort by importance

eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

# Calculate how much variance each component explains
explained_ratio = eigvals / eigvals.sum()
explained_cum = explained_ratio.cumsum()

# Find components needed for 95% variance
n_components = int(np.searchsorted(explained_cum, 0.95) + 1)
X_pca = X_centered @ eigvecs[:, :n_components]  # Transform data

info_loss = float(1.0 - explained_cum[n_components - 1])

# Save PCA-transformed data
with open(PCA_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([f"pc{i + 1}" for i in range(n_components)] + ["species"])
    for i in range(len(labels)):
        writer.writerow([f"{x:.6f}" for x in X_pca[i]] + [labels[i]])

# Plot
species_order = ["setosa", "versicolor", "virginica"]
colors = {"setosa": "#1f77b4", "versicolor": "#2ca02c", "virginica": "#d62728"}

plt.figure(figsize=(8, 6))
if n_components == 2:
    for sp in species_order:
        idxs = [i for i, s in enumerate(labels) if s == sp]
        plt.scatter(
            X_pca[idxs, 0], X_pca[idxs, 1], s=12, alpha=0.7, label=sp, c=colors[sp]
        )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
else:
    from mpl_toolkits.mplot3d import Axes3D

    ax = cast(Axes3D, plt.axes(projection="3d"))
    for sp in species_order:
        idxs = [i for i, s in enumerate(labels) if s == sp]
        ax.scatter(
            X_pca[idxs, 0],
            X_pca[idxs, 1],
            zs=X_pca[idxs, 2],
            s=10,
            alpha=0.7,
            label=sp,
            c=colors[sp],
        )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

plt.title(f"PCA on iris_big.csv (components={n_components})")
plt.legend()
plt.tight_layout()
plt.savefig(PCA_PLOT, dpi=160)

summary = {
    "explained_variance_ratio": explained_ratio.tolist(),
    "explained_variance_cumulative": explained_cum.tolist(),
    "n_components_95": n_components,
    "information_loss": info_loss,
    "pca_plot": str(PCA_PLOT),
    "pca_csv": str(PCA_CSV),
}

with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("Task 2 summary:")
print(json.dumps(summary, indent=2))
print(f"PCA plot saved to: {PCA_PLOT}")
