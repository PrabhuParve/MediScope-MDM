# MediScope — Diabetes Patient Analytics & Hospital Readmission Analysis

An educational MDM Data Analytics project using the UCI diabetes 130-US hospitals encounter dataset. It explores encounter-level patient records, readmission patterns, medication patterns, classification trees, association rules, correlation and regression, and a Streamlit dashboard.

> **Important:** This is an academic analytics project, not a clinical decision-support system. It must not be used to diagnose, treat, or make care decisions.

## 1. Project structure

```text
MediScope_Full_Project/
├── data/
│   ├── raw/diabetic_data.csv
│   └── processed/
├── src/
│   ├── common.py
│   ├── prepare_data.py
│   ├── eda.py
│   ├── decision_trees.py
│   ├── association_rules.py
│   ├── regression.py
│   └── run_all.py
├── outputs/
│   ├── figures/
│   └── tables/
├── models/
├── docs/
├── app.py
├── requirements.txt
└── README.md
```

## 2. Setup on Windows PowerShell

Open the **MediScope_Full_Project** folder in VS Code. Confirm that the terminal is in the folder containing `requirements.txt`:

```powershell
Get-Location
dir
```

Create and activate a virtual environment (recommended):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Otherwise install with:

```powershell
py -3.11 -m pip install -r requirements.txt
```

## 3. Run the analysis

Run all data preparation, EDA, decision-tree, association-rule, and regression tasks:

```powershell
py -3.11 -m src.run_all
```

Run one module at a time if preferred:

```powershell
py -3.11 -m src.prepare_data
py -3.11 -m src.eda
py -3.11 -m src.decision_trees
py -3.11 -m src.association_rules
py -3.11 -m src.regression
```

## 4. Launch the dashboard

```powershell
py -3.11 -m streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`.

## 5. Algorithms and concepts covered

- Dataset profiling, missing-value and duplicate analysis
- Cleaning placeholder `?` values, handling high-missingness columns transparently
- EDA: readmission, age, length of stay, admission type, medication patterns
- Decision trees: ID3 (information gain), C4.5-style gain ratio, and CART-style Gini split
- Association analysis: Apriori, FP-Growth, vertical transaction/TID representation, and a simple two-level category hierarchy
- Correlation matrix and multiple linear regression using NumPy gradient descent
- Interactive Streamlit dashboard

## 6. Methodology and limitations

- Each row is a hospital encounter, not necessarily a unique person. Repeated encounters exist.
- `patient_nbr` and `encounter_id` are identifiers and are excluded from predictive features.
- Readmission labels are `NO`, `>30`, and `<30`. The dataset's original labels are retained.
- Model evaluation uses a patient-group split where supported to reduce leakage from repeat encounters.
- The tree implementations are educational, compact implementations; they are not intended to outperform production libraries.
- For speed, association rules use a configurable sample and selected categorical medication/diagnosis features.
- Some original fields have extensive missingness. `weight`, `payer_code`, `medical_specialty`, `A1Cresult`, and `max_glu_serum` are not automatically imputed as if missing meant a known clinical value.
- Regression estimates length of stay from selected encounter attributes. Associations are not causal effects.

## 7. Suggested presentation sections

1. Problem statement and objectives
2. Dataset overview and data dictionary
3. Data quality and preprocessing decisions
4. Exploratory analysis and visual findings
5. Decision-tree algorithm comparison
6. Association rules and transaction mining
7. Correlation and regression
8. Dashboard walkthrough
9. Limitations, ethics, and conclusion
