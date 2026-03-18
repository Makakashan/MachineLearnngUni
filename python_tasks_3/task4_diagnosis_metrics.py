import json
from typing import Any, cast

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.axes3d import Axes3D
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from common import OUT_DIR, load_diagnosis_df, split_xy

SUMMARY_JSON = OUT_DIR / "task4_summary.json"
SCATTER_PNG = OUT_DIR / "task4_diagnosis_3d.png"
HEATMAP_PNG = OUT_DIR / "task4_confusion_matrices.png"
RESULTS_CSV = OUT_DIR / "task4_metrics.csv"

df = load_diagnosis_df()
x_train, x_test, y_train, y_test = split_xy(df, "diagnosis", random_state=13)

fig = plt.figure(figsize=(8, 6))
ax = cast(Axes3D, fig.add_subplot(111, projection="3d"))
color_map = {0: "#1f77b4", 1: "#d62728"}
diagnosis_values = df["diagnosis"].to_numpy()
colors = [color_map[int(value)] for value in diagnosis_values]
param1 = df["param1"].to_numpy()
param2 = df["param2"].to_numpy()
param3 = df["param3"].to_numpy()
cast(Any, ax).scatter(param1, param2, zs=param3, c=colors, s=20, alpha=0.7)
ax.set_xlabel("param1")
ax.set_ylabel("param2")
ax.set_zlabel("param3")
ax.set_title("Diagnosis dataset")
legend_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="healthy (0)",
        markerfacecolor="#1f77b4",
        markersize=8,
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="w",
        label="ill (1)",
        markerfacecolor="#d62728",
        markersize=8,
    ),
]
ax.legend(handles=legend_handles, loc="upper left")
plt.tight_layout()
plt.savefig(SCATTER_PNG, dpi=160)
plt.close(fig)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=13),
    "kNN_3": KNeighborsClassifier(n_neighbors=3),
    "kNN_5": KNeighborsClassifier(n_neighbors=5),
    "kNN_11": KNeighborsClassifier(n_neighbors=11),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=13),
}

results: list[dict[str, Any]] = []
summary: dict[str, Any] = {}
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
axes = axes.flatten()

for index, (name, model) in enumerate(models.items()):
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)

    results.append(
        {
            "classifier": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        }
    )
    summary[name] = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "confusion_matrix": matrix.tolist(),
    }

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["pred 0", "pred 1"],
        yticklabels=["true 0", "true 1"],
        ax=axes[index],
    )
    axes[index].set_title(name)

fig.tight_layout()
plt.savefig(HEATMAP_PNG, dpi=160)
plt.close(fig)

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
results_df.to_csv(RESULTS_CSV, index=False)

summary["answers"] = {
    "accuracy_definition": "Accuracy to odsetek wszystkich poprawnych klasyfikacji.",
    "precision_definition": "Precision to odsetek poprawnych diagnoz pozytywnych wsrod wszystkich przewidzianych jako pozytywne.",
    "recall_definition": "Recall (sensitivity) to odsetek wykrytych osob chorych wsrod wszystkich faktycznie chorych.",
    "important_for_avoiding_false_positive": "Precision",
    "important_for_avoiding_false_negative": "Recall",
    "is_accuracy_safe_for_imbalanced_data": "Nie. Przy niezbalansowanym zbiorze accuracy moze byc mylace.",
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Zadanie 4 - diagnoza choroby")
print(results_df.to_string(index=False))
print()
print("Odpowiedzi teoretyczne:")
for key, value in summary["answers"].items():
    print(f"{key}: {value}")
print()
print(f"Summary zapisany do: {SUMMARY_JSON}")
print(f"Wykres 3D zapisany do: {SCATTER_PNG}")
print(f"Heatmapy zapisane do: {HEATMAP_PNG}")
