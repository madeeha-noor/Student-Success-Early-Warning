"""Student Success Early-Warning: advisor dashboard (M3 skeleton).

Page 1 (working): ranked list of students by predicted dropout risk.
Page 2 (working): what-if sliders for one student.
Pages 3 and 4: placeholders for clustering profiles and association rules.

The model here is a simple stand-in (Random Forest on admission + first-semester features).
Swap it for the team's final classifier in train_final_model() and scored_students().
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

st.set_page_config(page_title="Student Success Early-Warning", page_icon="🎓", layout="wide")
SEED = 42


# ---------------------------------------------------------------- data and model
@st.cache_data(show_spinner="Loading dataset...")
def load_data():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=697)
    X = ds.data.features.copy()
    y = ds.data.targets["Target"].copy()
    return X, y


def early_columns(X):
    """Features known by the end of the first semester (drop all 2nd-semester columns)."""
    return [c for c in X.columns if "2nd sem" not in c]


def make_model():
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)


@st.cache_data(show_spinner="Scoring students (first run only, about 20 seconds)...")
def scored_students():
    """Out-of-fold dropout probability for every student (each student is scored by a model that never saw them)."""
    X, y = load_data()
    Xe = X[early_columns(X)]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(make_model(), Xe, y, cv=cv, method="predict_proba")
    classes = sorted(y.unique())
    out = Xe.copy()
    out["risk"] = proba[:, classes.index("Dropout")] * 100
    out["actual"] = y.values
    out.index.name = "student_id"
    return out


@st.cache_resource(show_spinner="Training model...")
def train_final_model():
    X, y = load_data()
    cols = early_columns(X)
    model = make_model().fit(X[cols], y)
    return model, cols


def reasons(row):
    r = []
    if row.get("Tuition fees up to date") == 0:
        r.append("tuition not up to date")
    if row.get("Debtor") == 1:
        r.append("debtor")
    if row.get("Curricular units 1st sem (approved)") == 0:
        r.append("no units approved in 1st sem")
    g = row.get("Curricular units 1st sem (grade)")
    if g is not None and 0 < g < 10:
        r.append("low 1st sem grade")
    return ", ".join(r) if r else "-"


# ---------------------------------------------------------------- pages
def page_risk_list():
    st.header("At-risk students")
    st.caption("Students ranked by predicted dropout risk, using admission data and first-semester results.")
    df = scored_students()

    c1, c2 = st.columns([2, 1])
    threshold = c1.slider("Flag students with risk above (%)", 10, 90, 50, step=5)
    top_n = c2.number_input("Rows to show", 10, 500, 50, step=10)

    flagged = df[df["risk"] >= threshold]
    m1, m2, m3 = st.columns(3)
    m1.metric("Students", f"{len(df):,}")
    m2.metric("Flagged", f"{len(flagged):,}")
    if len(flagged):
        m3.metric("Flagged who actually dropped out", f"{(flagged['actual'] == 'Dropout').mean() * 100:.0f}%",
                  help="Check against known outcomes in this dataset. A real advisor would not have this column.")

    view = flagged.sort_values("risk", ascending=False).head(int(top_n)).copy()
    view["Reasons (simple flags)"] = view.apply(reasons, axis=1)
    show = view[["risk", "Reasons (simple flags)", "Debtor", "Tuition fees up to date",
                 "Scholarship holder", "Curricular units 1st sem (approved)", "Curricular units 1st sem (grade)"]]
    show = show.rename(columns={"risk": "Dropout risk (%)"})
    st.dataframe(show, width="stretch",
                 column_config={"Dropout risk (%)": st.column_config.ProgressColumn(
                     "Dropout risk (%)", min_value=0, max_value=100, format="%.0f")})
    st.caption("Reasons are simple rule-based flags for now. Model-based explanations come in a later milestone.")


def page_what_if():
    st.header("What-if for one student")
    st.caption("Start from a typical student and change a few things to see how the predicted risk moves.")
    X, _ = load_data()
    model, cols = train_final_model()
    base = X[cols].median()

    a, b = st.columns(2)
    approved = a.slider("Units approved in 1st semester", 0, int(X["Curricular units 1st sem (approved)"].max()),
                        int(base["Curricular units 1st sem (approved)"]))
    grade = a.slider("Average grade in 1st semester", 0.0, 20.0,
                     float(round(base["Curricular units 1st sem (grade)"], 1)), step=0.5)
    age = a.slider("Age at enrollment", 17, 70, int(base["Age at enrollment"]))
    tuition = b.toggle("Tuition fees up to date", value=bool(base["Tuition fees up to date"]))
    debtor = b.toggle("Debtor", value=bool(base["Debtor"]))
    scholar = b.toggle("Scholarship holder", value=bool(base["Scholarship holder"]))

    row = base.copy()
    row["Curricular units 1st sem (approved)"] = approved
    row["Curricular units 1st sem (grade)"] = grade
    row["Age at enrollment"] = age
    row["Tuition fees up to date"] = int(tuition)
    row["Debtor"] = int(debtor)
    row["Scholarship holder"] = int(scholar)

    proba = model.predict_proba(pd.DataFrame([row[cols]]))[0]
    p = dict(zip(model.classes_, proba))
    st.subheader(f"Predicted dropout risk: {p['Dropout'] * 100:.0f}%")
    st.progress(float(p["Dropout"]))
    st.write({k: f"{v * 100:.0f}%" for k, v in p.items()})
    st.caption("Other features are held at the typical (median) student.")


def page_profiles():
    st.header("Cohort profiles")
    st.info("Coming next: student groups from the clustering work, each with a name and its dropout rate.")


def page_rules():
    st.header("Dropout rules")
    st.info("Coming next: association rules from notebook 04 (IF ... THEN Dropout) with support, confidence and lift.")


# ---------------------------------------------------------------- layout
st.title("🎓 Student Success Early-Warning")
st.sidebar.title("Menu")
page = st.sidebar.radio("Page", ["At-risk students", "What-if", "Cohort profiles", "Dropout rules"])
st.sidebar.caption("DS-311 Data Mining, Group 03. Decision support for advisors, never an automatic decision.")

{"At-risk students": page_risk_list, "What-if": page_what_if,
 "Cohort profiles": page_profiles, "Dropout rules": page_rules}[page]()
