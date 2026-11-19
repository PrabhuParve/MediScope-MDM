"""Correlation and multiple linear regression trained with NumPy gradient descent."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.common import load_clean, TABLES, MODELS

NUMERIC = ["num_lab_procedures", "num_procedures", "num_medications",
           "number_outpatient", "number_emergency", "number_inpatient",
           "number_diagnoses", "age"]

def run_regression(sample_size=20000, epochs=1800, learning_rate=0.03):
    df = load_clean()
    cols = [c for c in NUMERIC if c in df.columns] + ["time_in_hospital"]
    data = df[cols + (["patient_nbr"] if "patient_nbr" in df.columns else [])].copy()
    data = data.sample(min(sample_size, len(data)), random_state=13)
    # Convert age labels to an ordinal band index.
    if "age" in data:
        data["age"] = data["age"].astype(str).str.extract(r"(\d+)").astype(float)
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
        data[c] = data[c].fillna(data[c].median())
    X = data[[c for c in cols if c != "time_in_hospital"]].to_numpy(dtype=float)
    y = data["time_in_hospital"].to_numpy(dtype=float).reshape(-1, 1)
    if "patient_nbr" in data and data["patient_nbr"].nunique() > 1:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=13)
        train_idx, test_idx = next(splitter.split(X, y, groups=data["patient_nbr"]))
    else:
        train_idx, test_idx = train_test_split(np.arange(len(data)), test_size=0.25, random_state=13)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Standardize using training-set statistics only.
    mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
    sigma[sigma == 0] = 1
    X_train = (X_train - mu) / sigma
    X_test = (X_test - mu) / sigma
    X_train = np.c_[np.ones((len(X_train), 1)), X_train]
    X_test = np.c_[np.ones((len(X_test), 1)), X_test]

    weights = np.zeros((X_train.shape[1], 1))
    for _ in range(epochs):
        pred = X_train @ weights
        grad = (2 / len(X_train)) * (X_train.T @ (pred - y_train))
        weights -= learning_rate * grad

    predictions = X_test @ weights
    metrics = pd.DataFrame([{
        "target": "time_in_hospital",
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(y_test, predictions) ** 0.5,
        "R2": r2_score(y_test, predictions),
        "test_encounters": len(y_test)
    }])
    metrics.to_csv(TABLES / "regression_metrics.csv", index=False)
    np.savez(MODELS / "linear_regression_numpy.npz", weights=weights, mean=mu, std=sigma,
             features=np.array([c for c in cols if c != "time_in_hospital"]))
    print(metrics.to_string(index=False))
    return metrics

if __name__ == "__main__":
    run_regression()
