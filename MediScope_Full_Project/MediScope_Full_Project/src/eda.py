import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.common import load_clean, FIGURES, TABLES

sns.set_theme(style="whitegrid")

def run_eda():
    df = load_clean()
    df.describe(include="all").T.to_csv(TABLES / "descriptive_statistics.csv")

    # Readmission distribution
    readmit = df["readmitted"].value_counts(dropna=False).rename_axis("readmitted").reset_index(name="encounters")
    readmit["percent"] = (readmit["encounters"] / len(df) * 100).round(2)
    readmit.to_csv(TABLES / "readmission_distribution.csv", index=False)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="readmitted", order=["NO", ">30", "<30"])
    plt.title("Hospital encounters by readmission status")
    plt.xlabel("Readmission status")
    plt.ylabel("Number of encounters")
    plt.tight_layout()
    plt.savefig(FIGURES / "readmission_distribution.png", dpi=160)
    plt.close()

    # Length of stay
    if "time_in_hospital" in df:
        plt.figure(figsize=(9, 5))
        sns.histplot(df["time_in_hospital"], bins=14, discrete=True)
        plt.title("Distribution of length of hospital stay")
        plt.xlabel("Days in hospital")
        plt.tight_layout()
        plt.savefig(FIGURES / "length_of_stay.png", dpi=160)
        plt.close()

    # Age band
    if "age" in df:
        age_order = [x for x in ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
                                 "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]
                     if x in set(df["age"])]
        plt.figure(figsize=(11, 5))
        sns.countplot(data=df, x="age", order=age_order, color="steelblue")
        plt.title("Encounter distribution by age band")
        plt.xticks(rotation=35)
        plt.tight_layout()
        plt.savefig(FIGURES / "age_distribution.png", dpi=160)
        plt.close()

    # Admission type
    if "admission_type_id" in df:
        plt.figure(figsize=(9, 5))
        sns.countplot(data=df, x="admission_type_id", hue="readmitted")
        plt.title("Readmission status by admission type")
        plt.tight_layout()
        plt.savefig(FIGURES / "admission_type_readmission.png", dpi=160)
        plt.close()

    # Correlation for numeric fields
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > 1:
        corr = numeric.corr(numeric_only=True)
        corr.to_csv(TABLES / "correlation_matrix.csv")
        plt.figure(figsize=(13, 10))
        sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation matrix — numeric variables")
        plt.tight_layout()
        plt.savefig(FIGURES / "correlation_heatmap.png", dpi=160)
        plt.close()

    print("EDA outputs saved under outputs/")
    return df

if __name__ == "__main__":
    run_eda()
