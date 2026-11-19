from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "diabetic_data.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "diabetic_data_cleaned.csv"
OUTPUT = ROOT / "outputs"
FIGURES = OUTPUT / "figures"
TABLES = OUTPUT / "tables"
MODELS = ROOT / "models"

for folder in [OUTPUT, FIGURES, TABLES, MODELS, CLEAN_PATH.parent]:
    folder.mkdir(parents=True, exist_ok=True)

def load_raw():
    if not RAW_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW_PATH}")
    df = pd.read_csv(RAW_PATH, low_memory=False)
    # The source uses '?' as a missing-value marker.
    return df.replace("?", np.nan)

def load_clean():
    if CLEAN_PATH.exists():
        return pd.read_csv(CLEAN_PATH, low_memory=False)
    from src.prepare_data import prepare
    return prepare()
