import json

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix

from common import OUT_DIR, load_iris_df, split_xy

SUMMARY_JSON = OUT_DIR / "task1_summary.json"


def classify_iris(sl, sw, pl, pw):
    if pl <= 2.5:
        return "setosa"
    elif pw > 1.75:
        return "virginica"
    else:
        return "versicolor"


df = load_iris_df()
x_train, x_test, y_train, y_test = split_xy(df, "target_name", random_state=13)
train_df = pd.concat(
    [x_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
)
test_df = pd.concat(
    [x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
)

train_set = train_df.values
test_set = test_df.values

predictions = [
    classify_iris(row[0], row[1], row[2], row[3]) for row in test_set
]
true_labels = [row[4] for row in test_set]

accuracy = accuracy_score(true_labels, predictions)
matrix = confusion_matrix(
    true_labels, predictions, labels=["setosa", "versicolor", "virginica"]
)

summary = {
    "train_size": int(train_set.shape[0]),
    "test_size": int(test_set.shape[0]),
    "good_predictions": int((matrix.diagonal()).sum()),
    "accuracy": float(accuracy),
    "accuracy_percent": float(accuracy * 100),
    "confusion_matrix": matrix.tolist(),
    "labels": ["setosa", "versicolor", "virginica"],
    "rules": [
        "if petal length <= 2.5 -> setosa",
        "elif petal width > 1.75 -> virginica",
        "else -> versicolor",
    ],
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Zadanie 1 - klasyfikacja przez czlowieka")
print("Train set:")
print(train_set)
print()
print("Test set:")
print(test_set)
print()
print("Confusion matrix:")
print(matrix)
print()
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Summary zapisany do: {SUMMARY_JSON}")
