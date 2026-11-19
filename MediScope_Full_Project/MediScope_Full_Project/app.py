from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw" / "diabetic_data.csv"
CLEAN = ROOT / "data" / "processed" / "diabetic_data_cleaned.csv"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

st.set_page_config(page_title="MediScope | Diabetes Analytics", page_icon="🩺",
                   layout="wide", initial_sidebar_state="expanded")

@st.cache_data
def load_data():
    path = CLEAN if CLEAN.exists() else RAW
    df = pd.read_csv(path, low_memory=False)
    return df.replace("?", "Unknown")

df = load_data()
st.sidebar.title("🩺 MediScope")
st.sidebar.caption("Diabetes Patient Analytics")
page = st.sidebar.radio("Navigate", ["Overview", "Patient & Encounter Trends",
                                    "Clinical & Medication Patterns", "Readmission Analysis",
                                    "Algorithms & Models", "Data Quality & Methodology"])

st.title("MediScope")
st.caption("Diabetes Patient Analytics & Hospital Readmission Analysis")
st.info("Academic analytics only. This dashboard is not a clinical decision-support tool.")

if page == "Overview":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hospital encounters", f"{len(df):,}")
    c2.metric("Unique patients", f"{df['patient_nbr'].nunique():,}" if "patient_nbr" in df else "N/A")
    c3.metric("Average stay", f"{df['time_in_hospital'].mean():.2f} days" if "time_in_hospital" in df else "N/A")
    c4.metric("Readmitted within 30 days", f"{(df['readmitted']=='<30').mean()*100:.1f}%" if "readmitted" in df else "N/A")
    st.subheader("Readmission distribution")
    if "readmitted" in df:
        counts = df["readmitted"].value_counts().reindex(["NO", ">30", "<30"]).fillna(0).reset_index()
        counts.columns = ["Status", "Encounters"]
        st.plotly_chart(px.bar(counts, x="Status", y="Encounters", color="Status",
                               text_auto=True), use_container_width=True)
    st.subheader("Dataset preview")
    st.dataframe(df.head(20), use_container_width=True)

elif page == "Patient & Encounter Trends":
    st.subheader("Encounter trends")
    if "age" in df:
        age = df["age"].value_counts().reindex(["[0-10)", "[10-20)", "[20-30)", "[30-40)",
            "[40-50)", "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"]).dropna().reset_index()
        age.columns = ["Age band", "Encounters"]
        st.plotly_chart(px.bar(age, x="Age band", y="Encounters", title="Encounters by age band"),
                        use_container_width=True)
    if "time_in_hospital" in df:
        st.plotly_chart(px.histogram(df, x="time_in_hospital", nbins=14,
                                     title="Length of stay distribution"),
                        use_container_width=True)
    if "admission_type_id" in df and "readmitted" in df:
        cross = pd.crosstab(df["admission_type_id"], df["readmitted"]).reset_index()
        st.plotly_chart(px.bar(cross, x="admission_type_id", y=[c for c in cross.columns if c != "admission_type_id"],
                               barmode="group", title="Readmission by admission type"),
                        use_container_width=True)

elif page == "Clinical & Medication Patterns":
    st.subheader("Medication and clinical indicators")
    meds = [c for c in ["metformin", "insulin", "glipizide", "glyburide",
                        "pioglitazone", "rosiglitazone", "diabetesMed"] if c in df]
    if meds:
        selected = st.selectbox("Choose a medication field", meds)
        st.plotly_chart(px.histogram(df, x=selected, color="readmitted" if "readmitted" in df else None,
                                     barmode="group", title=f"{selected} categories"),
                        use_container_width=True)
    for col in ["A1Cresult", "max_glu_serum"]:
        if col in df:
            st.plotly_chart(px.histogram(df, x=col, color="readmitted" if "readmitted" in df else None,
                                         barmode="group", title=f"{col} distribution"),
                            use_container_width=True)

elif page == "Readmission Analysis":
    st.subheader("Readmission exploration")
    status = st.multiselect("Readmission status", ["NO", ">30", "<30"],
                            default=["NO", ">30", "<30"])
    view = df[df["readmitted"].isin(status)] if "readmitted" in df else df
    if "age" in view and "readmitted" in view:
        cross = pd.crosstab(view["age"], view["readmitted"], normalize="index").reset_index()
        st.plotly_chart(px.bar(cross, x="age", y=[c for c in cross.columns if c != "age"],
                               barmode="group", title="Readmission proportions by age band"),
                        use_container_width=True)
    if "time_in_hospital" in view and "readmitted" in view:
        st.plotly_chart(px.box(view, x="readmitted", y="time_in_hospital",
                               category_orders={"readmitted": ["NO", ">30", "<30"]},
                               title="Length of stay by readmission status"),
                        use_container_width=True)
    st.dataframe(view[["readmitted", "age", "time_in_hospital"]].head(100)
                 if all(c in view for c in ["readmitted", "age", "time_in_hospital"]) else view.head(100),
                 use_container_width=True)

elif page == "Algorithms & Models":
    st.subheader("Model outputs")
    for filename, title in [
        ("decision_tree_comparison.csv", "Decision-tree comparison"),
        ("regression_metrics.csv", "Linear regression metrics"),
        ("apriori_rules.csv", "Apriori association rules"),
        ("fp_growth_itemsets.csv", "FP-Growth frequent itemsets"),
        ("multilevel_itemsets.csv", "Multilevel association itemsets")
    ]:
        st.markdown(f"**{title}**")
        path = TABLES / filename
        if path.exists():
            st.dataframe(pd.read_csv(path).head(100), use_container_width=True)
        else:
            st.warning(f"{filename} not generated yet. Run `py -3.11 src/run_all.py` first.")
    path = FIGURES / "correlation_heatmap.png"
    if path.exists():
        st.image(str(path), caption="Numeric correlation heatmap", use_container_width=True)

elif page == "Data Quality & Methodology":
    st.subheader("Data quality")
    missing_path = TABLES / "missing_values.csv"
    if missing_path.exists():
        miss = pd.read_csv(missing_path)
        st.dataframe(miss.sort_values("missing_percent", ascending=False).head(20),
                     use_container_width=True)
    else:
        missing = df.isna().sum().sort_values(ascending=False).head(20).reset_index()
        missing.columns = ["Column", "Missing values"]
        st.dataframe(missing, use_container_width=True)
    st.markdown("""
    **Methodology notes**
    - One row represents an encounter; a patient can appear more than once.
    - Patient and encounter identifiers are not predictive features.
    - `?` placeholders are treated as missing values.
    - High-missingness fields are excluded from the modelling copy rather than blindly filled.
    - Patient-group splitting is used for model experiments where applicable.
    - Readmission association is not proof of causation.
    """)

st.sidebar.divider()
st.sidebar.caption("MediScope · MDM Data Analytics Project")
