import pandas as pd
import numpy as np
from src.common import load_raw, CLEAN_PATH, TABLES

def prepare():
    df = load_raw()
    original_shape = df.shape

    # Standardize blank strings and source placeholder values.
    df = df.replace(r"^\s*$", np.nan, regex=True)

    # Remove exact duplicate rows only; keep repeated patients/encounters.
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    # Keep the original clinical categories. Do not infer that missing means "normal".
    # Drop columns with extreme missingness from the modelling copy only.
    high_missing = ["weight", "payer_code", "medical_specialty"]
    model_df = df.drop(columns=[c for c in high_missing if c in df.columns]).copy()

    # Missing diagnoses are kept as an explicit category for categorical analysis.
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in model_df:
            model_df[col] = model_df[col].fillna("Unknown")

    # Replace missing values in selected categorical fields with an explicit label.
    categorical = model_df.select_dtypes(include=["object"]).columns
    for col in categorical:
        model_df[col] = model_df[col].fillna("Unknown")

    # Numeric columns are not blanket-imputed here. The selected modelling scripts
    # perform feature-specific imputation on their training samples.
    model_df.to_csv(CLEAN_PATH, index=False)

    missing = df.isna().sum().rename("missing_count").to_frame()
    missing["missing_percent"] = (missing["missing_count"] / len(df) * 100).round(2)
    missing.sort_values("missing_percent", ascending=False).to_csv(TABLES / "missing_values.csv")

    summary = pd.DataFrame({
        "metric": ["original_rows", "original_columns", "exact_duplicate_rows_removed",
                   "clean_rows", "unique_patients"],
        "value": [original_shape[0], original_shape[1], exact_duplicates,
                  len(model_df), df["patient_nbr"].nunique() if "patient_nbr" in df else np.nan]
    })
    summary.to_csv(TABLES / "dataset_summary.csv", index=False)
    print("Prepared dataset:", model_df.shape)
    print("Saved:", CLEAN_PATH)
    return model_df

if __name__ == "__main__":
    prepare()
