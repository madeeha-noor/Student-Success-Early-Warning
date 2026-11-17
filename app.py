"""Student Success Early-Warning: advisor dashboard (DS-311, Group 03).

Pages
1. At-risk students   ranked list from the Random Forest (Areeba's feature set B and preprocessing, notebook 06)
2. What-if            change one student's details and see the predicted risk
3. Cohort profiles    K-means student groups (Qudsia's clustering, notebook 05)
4. Dropout rules      association rules IF ... THEN Dropout (Madeeha's notebook 04)
5. Model results      comparison table from notebook 06
"""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

st.set_page_config(page_title="Student Success Early-Warning", page_icon="🎓", layout="wide")
SEED = 42

# ---- feature sets (same lists as notebook 06, Feature Set B = admission + first semester)
CODE_COLS = ["Marital Status", "Application mode", "Course", "Previous qualification", "Nacionality",
             "Mother's qualification", "Father's qualification", "Mother's occupation", "Father's occupation"]
BINARY_COLS = ["Daytime/evening attendance", "Displaced", "Educational special needs", "Debtor",
               "Tuition fees up to date", "Gender", "Scholarship holder", "International"]
ADMISSION_NUM = ["Application order", "Previous qualification (grade)", "Admission grade", "Age at enrollment",
                 "Unemployment rate", "Inflation rate", "GDP"]
SEM1_NUM = ["Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
            "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
            "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"]
FEATURES_B = CODE_COLS + BINARY_COLS + ADMISSION_NUM + SEM1_NUM

# ---- clustering columns (same as notebook 05)
CL_NUM = ["Application order", "Age at enrollment", "Admission grade", "Previous qualification (grade)",
          "Curricular units 1st sem (credited)", "Curricular units 1st sem (enrolled)",
          "Curricular units 1st sem (evaluations)", "Curricular units 1st sem (approved)",
          "Curricular units 1st sem (grade)", "Curricular units 1st sem (without evaluations)"]
CL_BIN = ["Debtor", "Tuition fees up to date", "Scholarship holder", "Gender", "Displaced",
          "Daytime/evening attendance", "Educational special needs", "International"]


# ================================================================ data and models
@st.cache_data(show_spinner="Loading dataset...")
def load_df():
    from ucimlrepo import fetch_ucirepo
    ds = fetch_ucirepo(id=697)
    df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = df.columns.str.strip()
    return df


def make_pipeline():
    pre = ColumnTransformer([
        ("codes", OneHotEncoder(handle_unknown="ignore"), CODE_COLS),
        ("numbers", RobustScaler(), ADMISSION_NUM + SEM1_NUM),
        ("binary", "passthrough", BINARY_COLS)], sparse_threshold=0)
    rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=SEED, n_jobs=-1)
    return Pipeline([("preprocessing", pre), ("classifier", rf)])


@st.cache_data(show_spinner="Scoring students (first run only, about a minute)...")
def scored_students():
    """Out-of-fold dropout probability: every student is scored by a model that never saw that student."""
    df = load_df()
    X, y = df[FEATURES_B], df["Target"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(make_pipeline(), X, y, cv=cv, method="predict_proba")
    classes = sorted(y.unique())
    out = X.copy()
    out["risk"] = proba[:, classes.index("Dropout")] * 100
    out["actual"] = y.values
    out.index.name = "student_id"
    return out


@st.cache_resource(show_spinner="Training model...")
def final_model():
    df = load_df()
    return make_pipeline().fit(df[FEATURES_B], df["Target"])


@st.cache_data(show_spinner="Building student profiles...")
def cluster_profiles():
    df = load_df()
    X = df[CL_NUM + CL_BIN]
    Xs = StandardScaler().fit_transform(X)
    cum = np.cumsum(PCA(random_state=SEED).fit(Xs).explained_variance_ratio_)
    n_comp = int(np.argmax(cum >= 0.90) + 1)
    Xp = PCA(n_components=n_comp, random_state=SEED).fit_transform(Xs)
    labels = KMeans(n_clusters=5, n_init=10, random_state=SEED).fit_predict(Xp)

    g = df.groupby(labels)
    stats = pd.DataFrame({
        "Students": g.size(),
        "Dropout %": g["Target"].apply(lambda s: (s == "Dropout").mean() * 100),
        "Enrolled %": g["Target"].apply(lambda s: (s == "Enrolled").mean() * 100),
        "Graduate %": g["Target"].apply(lambda s: (s == "Graduate").mean() * 100),
        "Average age": g["Age at enrollment"].mean(),
        "Approved units (1st sem)": g["Curricular units 1st sem (approved)"].mean(),
        "Grade (1st sem)": g["Curricular units 1st sem (grade)"].mean(),
        "Fees up to date %": g["Tuition fees up to date"].mean() * 100,
        "Scholarship %": g["Scholarship holder"].mean() * 100,
    })
    # profile names come from the cluster averages, so they do not depend on the cluster numbering
    names, n = {}, stats["Students"].sum()
    small = list(stats.index[stats["Students"] < 0.03 * n])
    rest = stats.drop(index=small)
    if len(rest):
        low = rest["Approved units (1st sem)"].idxmin(); names[low] = "Passed almost nothing in 1st semester"
        rest = rest.drop(index=low)
    if len(rest):
        heavy = rest["Approved units (1st sem)"].idxmax(); names[heavy] = "Mature students, heavy 1st-semester load"
        rest = rest.drop(index=heavy)
    if len(rest):
        young = rest["Average age"].idxmin(); names[young] = "Young students, on track"
        rest = rest.drop(index=young)
    for c in rest.index: names[c] = "Working-age students, mostly on track"
    for c in small: names[c] = "Very small group (not interpreted)"
    stats.insert(0, "Profile", [names[c] for c in stats.index])
    stats = stats.sort_values("Dropout %", ascending=False).reset_index(drop=True)
    return stats, n_comp


def build_items(df):
    yn = lambda s: s.map({1: "yes", 0: "no"})
    it = pd.DataFrame(index=df.index)
    it["a"] = "Debtor=" + yn(df["Debtor"])
    it["b"] = "Tuition up to date=" + yn(df["Tuition fees up to date"])
    it["c"] = "Scholarship=" + yn(df["Scholarship holder"])
    it["d"] = "Displaced=" + yn(df["Displaced"])
    it["e"] = "Gender=" + df["Gender"].map({1: "male", 0: "female"})
    it["f"] = "Age=" + pd.cut(df["Age at enrollment"], bins=[0, 19, 24, 200], labels=["<=19", "20-24", "25+"]).astype(str)
    it["g"] = "Admission grade=" + pd.qcut(df["Admission grade"].rank(method="first"), q=3, labels=["low", "mid", "high"]).astype(str)
    it["h"] = "Prev. qualification grade=" + pd.qcut(df["Previous qualification (grade)"].rank(method="first"), q=3, labels=["low", "mid", "high"]).astype(str)
    it["i"] = "1st sem approved=" + pd.cut(df["Curricular units 1st sem (approved)"], bins=[-1, 0, 3, 5, 100], labels=["none", "1-3", "4-5", "6+"]).astype(str)
    it["j"] = "1st sem grade=" + pd.cut(df["Curricular units 1st sem (grade)"], bins=[-1, 0, 11, 13, 100], labels=["none", "low", "mid", "high"]).astype(str)
    it["Target"] = "Target=" + df["Target"]
    return pd.get_dummies(it, prefix="", prefix_sep="", dtype=bool)


@st.cache_data(show_spinner="Mining rules...")
def mine_rules(min_sup, min_conf):
    from mlxtend.frequent_patterns import apriori, association_rules
    onehot = build_items(load_df())
    freq = apriori(onehot, min_support=min_sup, use_colnames=True)
    try:
        rules = association_rules(freq, num_itemsets=len(onehot), metric="confidence", min_threshold=min_conf)
    except TypeError:
        rules = association_rules(freq, metric="confidence", min_threshold=min_conf)
    drop = rules["consequents"].apply(lambda s: s == frozenset({"Target=Dropout"}))
    no_t = rules["antecedents"].apply(lambda s: not any(i.startswith("Target=") for i in s))
    r = rules[drop & no_t & (rules["lift"] >= 1.2)].copy()
    r["IF (antecedent)"] = r["antecedents"].apply(lambda s: "  AND  ".join(sorted(s)))
    r = r.sort_values(["lift", "confidence"], ascending=False)
    return r[["IF (antecedent)", "support", "confidence", "lift"]].round(3).reset_index(drop=True), len(freq)


def reasons(row):
    r = []
    if row["Tuition fees up to date"] == 0: r.append("tuition not up to date")
    if row["Debtor"] == 1: r.append("debtor")
    if row["Curricular units 1st sem (approved)"] == 0: r.append("no units approved in 1st sem")
    g = row["Curricular units 1st sem (grade)"]
    if 0 < g < 10: r.append("low 1st sem grade")
    return ", ".join(r) if r else "-"


# ================================================================ pages
def page_risk_list():
    st.header("At-risk students")
    st.caption("Students ranked by predicted dropout risk, using admission data and first-semester results.")
    df = scored_students()

    c1, c2 = st.columns([2, 1])
    threshold = c1.slider("Flag students with risk above (%)", 10, 90, 60, step=5)
    top_n = c2.number_input("Rows to show", 10, 500, 50, step=10)
    flagged = df[df["risk"] >= threshold]

    m1, m2, m3 = st.columns(3)
    m1.metric("Students", f"{len(df):,}")
    m2.metric("Flagged", f"{len(flagged):,}")
    if len(flagged):
        m3.metric("Flagged who actually dropped out", f"{(flagged['actual'] == 'Dropout').mean() * 100:.0f}%",
                  help="Checked against the known outcomes in this dataset. A real advisor would not have this column.")

    view = flagged.sort_values("risk", ascending=False).head(int(top_n)).copy()
    view["Reasons (simple flags)"] = view.apply(reasons, axis=1)
    yes_no = lambda s: s.map({1: "Yes", 0: "No"})
    show = pd.DataFrame({
        "Dropout risk (%)": view["risk"],
        "Reasons (simple flags)": view["Reasons (simple flags)"],
        "Debtor": yes_no(view["Debtor"]),
        "Fees up to date": yes_no(view["Tuition fees up to date"]),
        "Scholarship": yes_no(view["Scholarship holder"]),
        "Age": view["Age at enrollment"],
        "Units approved (1st sem)": view["Curricular units 1st sem (approved)"],
        "Grade (1st sem)": view["Curricular units 1st sem (grade)"].round(1)})
    st.dataframe(show, width="stretch", column_config={"Dropout risk (%)": st.column_config.ProgressColumn(
        "Dropout risk (%)", min_value=0, max_value=100, format="%.0f")})
    st.caption("Model: Random Forest on admission and first-semester features, scored out-of-fold. "
               "Reasons are simple rule-based flags; model-based explanations are a later step.")


def page_what_if():
    st.header("What-if for one student")
    st.caption("Start from a typical student and change a few things to see how the predicted risk moves.")
    df = load_df()
    model = final_model()
    base = pd.concat([df[CODE_COLS + BINARY_COLS].mode().iloc[0], df[ADMISSION_NUM + SEM1_NUM].median()])

    a, b = st.columns(2)
    approved = a.slider("Units approved in 1st semester", 0, int(df["Curricular units 1st sem (approved)"].max()),
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

    p = dict(zip(model.classes_, model.predict_proba(pd.DataFrame([row[FEATURES_B]]))[0]))
    st.subheader(f"Predicted dropout risk: {p['Dropout'] * 100:.0f}%")
    st.progress(float(p["Dropout"]))
    c1, c2, c3 = st.columns(3)
    c1.metric("Dropout", f"{p['Dropout'] * 100:.0f}%")
    c2.metric("Enrolled", f"{p['Enrolled'] * 100:.0f}%")
    c3.metric("Graduate", f"{p['Graduate'] * 100:.0f}%")
    st.caption("All other details are held at the typical student (most common category or median).")


def page_profiles():
    st.header("Cohort profiles")
    st.caption("Students grouped by similarity with K-means (k = 5). The outcome was not used to build the groups, "
               "only to describe them afterwards.")
    stats, n_comp = cluster_profiles()
    top = stats.iloc[0]
    st.metric(f"Highest-risk profile: {top['Profile']}", f"{top['Dropout %']:.0f}% dropout",
              help=f"{int(top['Students'])} students")
    st.dataframe(stats.round(1), width="stretch", hide_index=True)
    st.bar_chart(stats.set_index("Profile")["Dropout %"])
    st.caption(f"Features: 18 columns, standardised, reduced with PCA to {n_comp} components. The silhouette score is low "
               "(about 0.16), so the groups overlap: these are descriptive profiles, not sharply separated groups. "
               "Profile names are derived from each group's averages.")


def page_rules():
    st.header("Dropout rules")
    st.caption("Association rules of the form IF A and B THEN Dropout (Apriori).")
    c1, c2 = st.columns(2)
    min_sup = c1.slider("Minimum support", 0.03, 0.15, 0.05, step=0.01)
    min_conf = c2.slider("Minimum confidence", 0.40, 0.80, 0.50, step=0.05)
    rules, n_items = mine_rules(min_sup, min_conf)
    st.write(f"{len(rules)} Dropout rules with lift of at least 1.2 (from {n_items} frequent itemsets).")
    if len(rules):
        st.dataframe(rules.head(20), width="stretch", hide_index=True)
    else:
        st.info("No rules at these settings. Lower the confidence or support.")
    st.caption("Support: share of all students with the rule's conditions and Dropout. Confidence: share of students "
               "with the conditions who dropped out. Lift: confidence divided by the overall Dropout rate (about 32%). "
               "Rules show association, not cause.")


def page_results():
    st.header("Model results")
    st.caption("From notebook 06: stratified 5-fold cross-validation on the training set (macro-F1).")
    res = pd.DataFrame({
        "Model": ["Decision Tree", "Naive Bayes", "Random Forest", "SVM"],
        "Admission only": [0.488, 0.212, 0.507, 0.574],
        "Admission + 1st semester": [0.575, 0.215, 0.623, 0.674]})
    st.dataframe(res, width="stretch", hide_index=True)
    st.bar_chart(res.set_index("Model"))
    st.write("Majority-class baseline macro-F1 on the test set: 0.222.")
    st.write("Best model (SVM, admission + 1st semester) on the test set: accuracy 0.72, macro-F1 0.68. "
             "Dropout precision 0.83, recall 0.63.")
    st.caption("First-semester results lift every useful model by about 0.06 to 0.10 macro-F1. "
               "The Enrolled class is the hardest to separate.")


# ================================================================ layout
st.title("🎓 Student Success Early-Warning")
st.sidebar.title("Menu")
PAGES = {"At-risk students": page_risk_list, "What-if": page_what_if, "Cohort profiles": page_profiles,
         "Dropout rules": page_rules, "Model results": page_results}
PAGES[st.sidebar.radio("Page", list(PAGES))]()
st.sidebar.caption("DS-311 Data Mining, Group 03. Decision support for advisors, never an automatic decision.")
