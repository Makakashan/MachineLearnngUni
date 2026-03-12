import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_CSV = ROOT / "iris_big_with_errors.csv"
OUTPUT_CSV = OUT_DIR / "iris_big_cleaned.csv"
SUMMARY_JSON = OUT_DIR / "task1_summary.json"

ALLOWED_SPECIES = {"setosa", "versicolor", "virginica"}


def fix_row_structure(row):
    """Fix rows where decimal comma split into 2 columns: "4,75" -> ["4", "75"] -> "4.75" """
    if len(row) == 6 and row[0].isdigit() and row[1].isdigit():
        return [row[0] + "." + row[1]] + row[2:]
    return row


def parse_float(value):
    """Parse string to float, return (number, error_type)"""
    raw = value.strip()
    if raw == "":
        return None, "empty"
    if "," in raw and "." not in raw:
        raw = raw.replace(",", ".")
    try:
        num = float(raw)
    except ValueError:
        return None, "non_numeric"
    if not (0 < num < 15):
        return None, "out_of_range"
    return num, None


def normalize_species(value):
    """Clean species names: remove separators, prefixes, punctuation"""
    s = value.strip().lower()
    s = s.replace("_", " ").replace("-", " ")
    s = " ".join(s.split())
    if s.startswith("iris "):
        s = s[5:]
    s = s.strip(" .?")
    return s


rows: List[List[float]] = []
raw_species: List[str | None] = []

struct_fixes = 0
struct_bad = 0
numeric_issues = Counter()
invalid_species_raw = Counter()

# Read CSV and detect errors
with open(INPUT_CSV, newline="") as f:
    reader = csv.reader(f)
    header = next(reader)

    for row in reader:
        fixed = fix_row_structure(row)
        if fixed != row:
            struct_fixes += 1
        row = fixed

        if len(row) != 5:
            struct_bad += 1
            continue

        numeric = []
        for j in range(4):
            num, issue = parse_float(row[j])
            if issue:
                numeric_issues[issue] += 1
                numeric.append(np.nan)
            else:
                numeric.append(num)

        species_raw = row[4]
        species_norm = normalize_species(species_raw)
        if species_norm not in ALLOWED_SPECIES:
            invalid_species_raw[species_raw] += 1
            species_norm = None

        rows.append(numeric)
        raw_species.append(species_norm)

X = np.array(rows, dtype=float)

# Calculate stats before fixing
missing_mask = np.isnan(X)
missing_total = int(missing_mask.sum())
missing_by_col = missing_mask.sum(axis=0).astype(int).tolist()

stats_with_errors = []
for col in range(X.shape[1]):
    col_data = X[:, col]
    stats_with_errors.append(
        {
            "min": float(np.nanmin(col_data)),
            "max": float(np.nanmax(col_data)),
            "mean": float(np.nanmean(col_data)),
            "std": float(np.nanstd(col_data)),
        }
    )

# Fix numeric: fill NaN with column median
col_median = np.nanmedian(X, axis=0)
X_filled = X.copy()
for j in range(X_filled.shape[1]):
    nan_mask = np.isnan(X_filled[:, j])
    X_filled[nan_mask, j] = col_median[j]

# Fix species: assign to nearest centroid
species: List[str | None] = list(raw_species)
valid_indices = [i for i, s in enumerate(species) if s in ALLOWED_SPECIES]
centroids: Dict[str, np.ndarray] = {}
for sp in ALLOWED_SPECIES:
    idx = [i for i in valid_indices if species[i] == sp]
    centroids[sp] = X_filled[idx].mean(axis=0)

fixed_species_count = 0
for i, sp in enumerate(species):
    if sp in ALLOWED_SPECIES:
        continue
    distances = {
        k: float(np.linalg.norm(X_filled[i] - v)) for k, v in centroids.items()
    }
    best = min(distances, key=lambda k: distances[k])
    species[i] = best
    fixed_species_count += 1

# Write cleaned CSV
with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for i in range(len(species)):
        writer.writerow([f"{x:.2f}" for x in X_filled[i]] + [species[i]])

summary = {
    "rows_total": int(len(rows)),
    "struct_fixes": int(struct_fixes),
    "struct_bad_rows": int(struct_bad),
    "numeric_issues": dict(numeric_issues),
    "missing_total": missing_total,
    "missing_by_col": missing_by_col,
    "stats_with_errors": stats_with_errors,
    "invalid_species_raw": dict(invalid_species_raw),
    "invalid_species_fixed": int(fixed_species_count),
}

with open(SUMMARY_JSON, "w") as f:
    json.dump(summary, f, indent=2)

print("Task 1 summary:")
print(json.dumps(summary, indent=2))
print(f"Cleaned file saved to: {OUTPUT_CSV}")
