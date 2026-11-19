"""
Educational decision tree implementations:
- ID3: entropy + information gain (categorical splits)
- C4.5-style: gain ratio (categorical splits)
- CART-style: Gini impurity (categorical multiway split)

To keep runtime practical, the experiment samples encounters and uses a small,
interpretable set of categorical attributes. These are educational implementations,
not replacements for mature ML libraries.
"""
import math
import json
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report
from src.common import load_clean, TABLES, MODELS

FEATURES = ["race", "gender", "age", "admission_type_id", "discharge_disposition_id",
            "admission_source_id", "A1Cresult", "max_glu_serum", "insulin", "metformin",
            "diabetesMed"]
TARGET = "readmitted"

def entropy(y):
    counts = np.bincount(np.asarray(y, dtype=int))
    probs = counts[counts > 0] / len(y)
    return float(-np.sum(probs * np.log2(probs)))

def gini(y):
    counts = np.bincount(np.asarray(y, dtype=int))
    probs = counts[counts > 0] / len(y)
    return float(1 - np.sum(probs ** 2))

def split_score(y, groups, criterion):
    parent = entropy(y) if criterion in ("gain", "gain_ratio") else gini(y)
    weighted = 0.0
    split_info = 0.0
    for idx in groups.values():
        weight = len(idx) / len(y)
        child = y[idx]
        weighted += weight * (entropy(child) if criterion in ("gain", "gain_ratio") else gini(child))
        if weight > 0:
            split_info -= weight * math.log2(weight)
    gain = parent - weighted
    if criterion == "gain_ratio":
        return gain / split_info if split_info > 0 else 0.0
    return gain

def build_tree(X, y, feature_names, criterion="gain", depth=0, max_depth=4, min_samples=100):
    majority = int(Counter(y).most_common(1)[0][0])
    if len(set(y)) == 1 or depth >= max_depth or len(y) < min_samples or not feature_names:
        return {"leaf": majority, "n": int(len(y))}
    best_feature, best_score, best_groups = None, -1.0, None
    for feature in feature_names:
        groups = {}
        for value in pd.unique(X[feature]):
            groups[value] = np.flatnonzero((X[feature].values == value))
        score = split_score(y, groups, criterion)
        if score > best_score:
            best_feature, best_score, best_groups = feature, score, groups
    if best_feature is None or best_score <= 1e-12 or not best_groups:
        return {"leaf": majority, "n": int(len(y))}
    children = {}
    remaining = [f for f in feature_names if f != best_feature]
    for value, idx in best_groups.items():
        children[str(value)] = build_tree(X.iloc[idx], y[idx], remaining, criterion,
                                          depth + 1, max_depth, min_samples)
    return {"feature": best_feature, "children": children, "default": majority,
            "score": float(best_score), "n": int(len(y))}

def predict_one(tree, row):
    if "leaf" in tree:
        return tree["leaf"]
    value = str(row.get(tree["feature"], "Unknown"))
    child = tree["children"].get(value)
    return predict_one(child, row) if child is not None else tree["default"]

def run_trees(sample_size=12000):
    df = load_clean()
    cols = [c for c in FEATURES if c in df.columns] + [TARGET]
    data = df[cols + (["patient_nbr"] if "patient_nbr" in df.columns else [])].copy()
    data = data.sample(min(sample_size, len(data)), random_state=42)
    data = data.dropna(subset=[TARGET])
    for c in cols:
        data[c] = data[c].fillna("Unknown").astype(str)
    labels = sorted(data[TARGET].unique())
    label_to_int = {v: i for i, v in enumerate(labels)}
    y = data[TARGET].map(label_to_int).to_numpy()
    X = data[[c for c in cols if c != TARGET]]
    if "patient_nbr" in data.columns and data["patient_nbr"].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups=data["patient_nbr"]))
    else:
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(data))
        cut = int(len(idx) * 0.75)
        train_idx, test_idx = idx[:cut], idx[cut:]
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    rows = []
    for name, criterion in [("ID3_information_gain", "gain"),
                            ("C4.5_gain_ratio", "gain_ratio"),
                            ("CART_gini", "gini")]:
        tree = build_tree(X_train, y_train, list(X.columns), criterion=criterion,
                          max_depth=4, min_samples=80)
        preds = np.array([predict_one(tree, row) for _, row in X_test.iterrows()])
        rows.append({"algorithm": name, "accuracy": accuracy_score(y_test, preds),
                     "test_encounters": len(y_test)})
        with open(MODELS / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(tree, f, indent=2)
    result = pd.DataFrame(rows)
    result.to_csv(TABLES / "decision_tree_comparison.csv", index=False)
    print(result.to_string(index=False))
    return result

if __name__ == "__main__":
    run_trees()
