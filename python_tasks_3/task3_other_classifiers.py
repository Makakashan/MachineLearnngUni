import json

import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from common import OUT_DIR, load_iris_df, split_xy

SUMMARY_JSON = OUT_DIR / "task3_summary.json"
RESULTS_CSV = OUT_DIR / "task3_results.csv"

df = load_iris_df()
x_train, x_test, y_train, y_test = split_xy(df, "target_name", random_state=13)

models = {
    "DecisionTree": DecisionTreeClassifier(random_state=13),
    "kNN_3": KNeighborsClassifier(n_neighbors=3),
    "kNN_5": KNeighborsClassifier(n_neighbors=5),
    "kNN_11": KNeighborsClassifier(n_neighbors=11),
    "NaiveBayes": GaussianNB(),
    "MLP": MLPClassifier(hidden_layer_sizes=(8,), max_iter=2000, random_state=13),
}

labels = ["setosa", "versicolor", "virginica"]
results = []
summary = {}

for name, model in models.items():
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    matrix = confusion_matrix(y_test, predictions, labels=labels)

    results.append({"classifier": name, "accuracy": accuracy})
    summary[name] = {
        "accuracy": float(accuracy),
        "accuracy_percent": float(accuracy * 100),
        "confusion_matrix": matrix.tolist(),
        "labels": labels,
    }

results_df = pd.DataFrame(results).sort_values("accuracy", ascending=False)
results_df.to_csv(RESULTS_CSV, index=False)

best_row = results_df.iloc[0]
summary["best_classifier"] = {
    "classifier": str(best_row["classifier"]),
    "accuracy": float(best_row["accuracy"]),
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("Zadanie 3 - inne klasyfikatory")
print(results_df.to_string(index=False))
print()
for name in results_df["classifier"]:
    print(name)
    print(confusion_matrix(y_test, models[name].predict(x_test), labels=labels))
    print()
print(f"Summary zapisany do: {SUMMARY_JSON}")
print(f"Tabela wynikow zapisana do: {RESULTS_CSV}")
