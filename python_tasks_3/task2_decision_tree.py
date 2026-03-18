import json

from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, export_text

from common import OUT_DIR, load_iris_df, split_xy

SUMMARY_JSON = OUT_DIR / "task2_summary.json"
TREE_TXT = OUT_DIR / "task2_tree.txt"

df = load_iris_df()
x_train, x_test, y_train, y_test = split_xy(df, "target_name", random_state=13)

model = DecisionTreeClassifier(random_state=13)
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
predictions = model.predict(x_test)
labels = ["setosa", "versicolor", "virginica"]
matrix = confusion_matrix(y_test, predictions, labels=labels)
tree_text = export_text(model, feature_names=list(x_train.columns))

TREE_TXT.write_text(tree_text, encoding="utf-8")

summary = {
    "train_size": int(len(x_train)),
    "test_size": int(len(x_test)),
    "accuracy": float(accuracy),
    "accuracy_percent": float(accuracy * 100),
    "confusion_matrix": matrix.tolist(),
    "labels": labels,
    "tree_text_file": str(TREE_TXT),
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Zadanie 2 - drzewo decyzyjne")
print("Train inputs:")
print(x_train)
print()
print("Test inputs:")
print(x_test)
print()
print("Tree:")
print(tree_text)
print()
print("Confusion matrix:")
print(matrix)
print()
print(f"Accuracy: {accuracy * 100:.2f}%")
print(f"Summary zapisany do: {SUMMARY_JSON}")
