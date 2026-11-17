"""Student Success Early-Warning: advisor dashboard (DS-311, Group 03).

Pages
1. At-risk students   ranked list from the Random Forest (Areeba's feature set B and preprocessing, notebook 06)
2. What-if            change one student's details and see the predicted risk
3. Cohort profiles    K-means student groups (Qudsia's clustering, notebook 05)
4. Dropout rules      association rules IF ... THEN Dropout (Madeeha's notebook 04)
5. Model results      comparison table from notebook 06
"""
import altair as alt
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

st.set_page_config(page_title="Student Success Early-Warning", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")
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
    r["IF (antecedent)"] = r["antecedents"].apply(lambda s: " and ".join(sorted(s)))
    r["n"] = r["antecedents"].apply(len)
    r[["support", "confidence", "lift"]] = r[["support", "confidence", "lift"]].round(3)
    # rules with identical numbers describe the same students: keep the shortest wording
    r = r.sort_values(["lift", "confidence", "n"], ascending=[False, False, True])
    r = r.drop_duplicates(subset=["support", "confidence", "lift"])
    return r[["IF (antecedent)", "support", "confidence", "lift"]].reset_index(drop=True), len(freq)


def reasons(row):
    r = []
    if row["Tuition fees up to date"] == 0: r.append("tuition not up to date")
    if row["Debtor"] == 1: r.append("debtor")
    if row["Curricular units 1st sem (approved)"] == 0: r.append("no units approved in 1st sem")
    g = row["Curricular units 1st sem (grade)"]
    if 0 < g < 10: r.append("low 1st sem grade")
    return ", ".join(r) if r else "-"


# ================================================================ look and feel
NAVY, BLUE, TEAL, RED, GREY, LIGHT = "#0B2545", "#1B6CA8", "#1B7A99", "#C0392B", "#8A94A6", "#DCE3EE"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'Inter', 'Segoe UI', Arial, sans-serif; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1240px; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], footer, #MainMenu { display: none !important; }
[data-testid="stSidebar"] { border-right: 1px solid #E5E9F0; }
.hero { background: linear-gradient(115deg, #0B2545 0%, #13315C 55%, #1B6CA8 100%); color: #fff;
        padding: 26px 32px; border-radius: 14px; margin-bottom: 22px; }
.hero .eyebrow { font-size: 11px; letter-spacing: .16em; text-transform: uppercase; opacity: .75; font-weight: 600; }
.hero h1 { color: #fff; font-size: 30px; font-weight: 700; margin: 6px 0 4px 0; padding: 0; line-height: 1.2; }
.hero .sub { font-size: 14px; opacity: .85; }
.ptitle { font-size: 22px; font-weight: 700; color: #0B2545; margin: 4px 0 2px 0; }
.psub { font-size: 14px; color: #5B6577; margin-bottom: 16px; }
.kpi { background: #fff; border: 1px solid #E5E9F0; border-left: 5px solid #1B6CA8; border-radius: 12px;
       padding: 15px 18px; box-shadow: 0 1px 3px rgba(16,24,40,.06); height: 100%; }
.kpi .l { font-size: 11px; letter-spacing: .08em; text-transform: uppercase; color: #6B7280; font-weight: 600; }
.kpi .v { font-size: 30px; font-weight: 700; color: #0B2545; line-height: 1.25; }
.kpi .s { font-size: 12px; color: #6B7280; }
.card { background: #fff; border: 1px solid #E5E9F0; border-radius: 12px; padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(16,24,40,.06); height: 100%; }
.card .n { font-size: 12px; font-weight: 600; color: #0B2545; min-height: 34px; }
.card .p { font-size: 32px; font-weight: 700; line-height: 1.2; }
.card .d { font-size: 12px; color: #6B7280; margin-top: 4px; }
.band { display: inline-block; padding: 3px 12px; border-radius: 999px; font-size: 12px; font-weight: 600; color: #fff; }
.note { font-size: 12.5px; color: #6B7280; margin-top: 10px; }
h2, h3 { color: #0B2545; }
</style>
""", unsafe_allow_html=True)


def kpi(label, value, sub="", accent=BLUE):
    return (f'<div class="kpi" style="border-left-color:{accent}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="s">{sub}</div></div>')


def kpi_row(items):
    for col, it in zip(st.columns(len(items)), items):
        col.markdown(kpi(*it), unsafe_allow_html=True)
    st.write("")


def page_header(title, sub):
    st.markdown(f'<div class="ptitle">{title}</div><div class="psub">{sub}</div>', unsafe_allow_html=True)


def band(risk):
    if risk >= 90: return "Critical", "#7B1E1E"
    if risk >= 75: return "High", RED
    if risk >= 60: return "Elevated", "#D9822B"
    if risk >= 40: return "Moderate", "#C9A227"
    return "Low", "#2E8B57"


def style_chart(ch, height=300):
    return (ch.properties(height=height)
              .configure_axis(labelColor="#4B5563", titleColor="#4B5563", gridColor="#EEF1F6", domainColor="#CBD2DC")
              .configure_view(strokeWidth=0)
              .configure_legend(labelColor="#4B5563", titleColor="#4B5563"))


st.markdown("""
<div class="hero">
  <div class="eyebrow">DS-311 Data Mining &nbsp;\u00b7&nbsp; Group 03 &nbsp;\u00b7&nbsp; Project P10</div>
  <h1>Student Success Early-Warning</h1>
  <div class="sub">Advisor decision-support platform: identify students at risk of dropping out early enough to help.</div>
</div>
""", unsafe_allow_html=True)


def gauge(risk, colour, label, note=""):
    """Pure CSS donut gauge (no chart library, so it always renders the same)."""
    return (f'<div style="display:flex;align-items:center;gap:18px;margin:6px 0 10px 0">'
            f'<div style="width:132px;height:132px;border-radius:50%;flex:none;'
            f'background:conic-gradient({colour} {risk * 3.6:.1f}deg,#E6EBF2 0);display:flex;align-items:center;justify-content:center">'
            f'<div style="width:100px;height:100px;border-radius:50%;background:#fff;display:flex;align-items:center;'
            f'justify-content:center;font-size:30px;font-weight:700;color:{colour}">{risk:.0f}%</div></div>'
            f'<div><span class="band" style="background:{colour}">{label}</span>'
            f'<div class="d" style="margin-top:8px">{note}</div></div></div>')


def support_actions(r):
    acts = []
    if r["Tuition fees up to date"] == 0 or r["Debtor"] == 1:
        acts.append("Financial counselling and a fee-payment plan")
    if r["Curricular units 1st sem (approved)"] == 0 or 0 < r["Curricular units 1st sem (grade)"] < 10:
        acts.append("Academic tutoring and a course-load review")
    if r["Age at enrollment"] >= 25:
        acts.append("Check work and family commitments; offer flexible options")
    acts.append("Advisor check-in within two weeks")
    return acts


def bullet_list(items):
    return "<ul style='margin:6px 0 0 0;padding-left:18px;font-size:13.5px;color:#374151;line-height:1.7'>" + \
           "".join(f"<li>{i}</li>" for i in items) + "</ul>"


# ================================================================ pages
def page_risk_list():
    page_header("At-risk students", "Students ranked by predicted dropout risk from admission data and first-semester results.")
    df = scored_students()
    total_drop = int((df["actual"] == "Dropout").sum())

    c1, c2 = st.columns([3, 1])
    threshold = c1.slider("Flag students with predicted risk of at least (%)", 10, 90, 60, step=5)
    top_n = c2.number_input("Rows to show", 10, 500, 50, step=10)
    flagged = df[df["risk"] >= threshold]
    hit = int((flagged["actual"] == "Dropout").sum())

    kpi_row([
        ("Students screened", f"{len(df):,}", "First-year cohort", NAVY),
        ("Flagged", f"{len(flagged):,}", f"{len(flagged) / len(df) * 100:.0f}% of students", RED),
        ("Flagged who dropped out", f"{hit / len(flagged) * 100:.0f}%" if len(flagged) else "-", "Precision, checked against known outcomes", TEAL),
        ("Dropouts captured", f"{hit / total_drop * 100:.0f}%", f"{hit:,} of {total_drop:,} dropouts", BLUE)])

    left, right = st.columns([3, 2])
    with left:
        hist = df[["risk"]].copy()
        hist["Status"] = np.where(hist["risk"] >= threshold, "Flagged", "Not flagged")
        bars = alt.Chart(hist).mark_bar().encode(
            x=alt.X("risk:Q", bin=alt.Bin(step=5), title="Predicted dropout risk (%)"),
            y=alt.Y("count()", title="Students"),
            color=alt.Color("Status:N", scale=alt.Scale(domain=["Not flagged", "Flagged"], range=[LIGHT, RED]),
                            legend=alt.Legend(orient="top", title=None)))
        st.markdown("**Risk distribution**")
        st.altair_chart(style_chart(bars, 230), width="stretch")
    with right:
        order = ["Low", "Moderate", "Elevated", "High", "Critical"]
        colours = {"Low": "#2E8B57", "Moderate": "#C9A227", "Elevated": "#D9822B", "High": RED, "Critical": "#7B1E1E"}
        counts = df["risk"].apply(lambda r: band(r)[0]).value_counts().reindex(order).fillna(0).reset_index()
        counts.columns = ["Band", "Students"]
        bb = alt.Chart(counts).mark_bar(cornerRadiusEnd=4, size=20).encode(
            x=alt.X("Students:Q"), y=alt.Y("Band:N", sort=order, title=None),
            color=alt.Color("Band:N", legend=None, scale=alt.Scale(domain=order, range=[colours[o] for o in order])),
            tooltip=["Band", "Students"])
        st.markdown("**Students by risk band**")
        st.altair_chart(style_chart(bb, 230), width="stretch")

    view = flagged.sort_values("risk", ascending=False).head(int(top_n)).copy()
    if view.empty:
        st.info("No students are flagged at this threshold. Lower the slider.")
        return
    yes_no = lambda s: s.map({1: "Yes", 0: "No"})
    show = pd.DataFrame({
        "Student ID": view.index,
        "Risk band": view["risk"].apply(lambda r: band(r)[0]).values,
        "Dropout risk (%)": view["risk"].values,
        "Reasons (simple flags)": view.apply(reasons, axis=1).values,
        "Debtor": yes_no(view["Debtor"]).values,
        "Fees up to date": yes_no(view["Tuition fees up to date"]).values,
        "Age": view["Age at enrollment"].values,
        "Units approved": view["Curricular units 1st sem (approved)"].values,
        "Grade": view["Curricular units 1st sem (grade)"].round(1).values})
    st.markdown("**Flagged students**")
    st.dataframe(show, width="stretch", height=340, hide_index=True, column_config={
        "Dropout risk (%)": st.column_config.ProgressColumn("Dropout risk (%)", min_value=0, max_value=100, format="%.0f"),
        "Reasons (simple flags)": st.column_config.TextColumn(width="large")})

    # ---- student record
    st.markdown('<div class="ptitle" style="font-size:18px;margin-top:18px">Student record</div>', unsafe_allow_html=True)
    sid = st.selectbox("Open a student record", list(view.index),
                       format_func=lambda i: f"Student {i}   |   predicted risk {df.loc[i, 'risk']:.0f}%")
    rec = df.loc[sid]
    label, colour = band(rec["risk"])
    typical = pd.concat([df[CODE_COLS + BINARY_COLS].mode().iloc[0], df[ADMISSION_NUM + SEM1_NUM].median()])
    g1, g2, g3 = st.columns([1.15, 1.35, 1.3], gap="medium")
    with g1:
        with st.container(border=True):
            st.markdown(gauge(rec["risk"], colour, f"{label} risk", f"Student {sid}<br>Known outcome in the dataset: {rec['actual']}"),
                        unsafe_allow_html=True)
    with g2:
        with st.container(border=True):
            st.markdown("**Compared with the typical student**")
            rows = [("Age at enrollment", rec["Age at enrollment"], typical["Age at enrollment"]),
                    ("Units approved (1st sem)", rec["Curricular units 1st sem (approved)"], typical["Curricular units 1st sem (approved)"]),
                    ("Grade (1st sem)", round(rec["Curricular units 1st sem (grade)"], 1), round(typical["Curricular units 1st sem (grade)"], 1)),
                    ("Admission grade", round(rec["Admission grade"], 1), round(typical["Admission grade"], 1))]
            t = pd.DataFrame(rows, columns=["Measure", "This student", "Typical"])
            for c in ["Debtor", "Tuition fees up to date", "Scholarship holder"]:
                t.loc[len(t)] = [c.replace("Tuition fees up to date", "Fees up to date"),
                                 "Yes" if rec[c] == 1 else "No", "Yes" if typical[c] == 1 else "No"]
            st.dataframe(t.astype(str), hide_index=True, width="stretch", height=250)
    with g3:
        with st.container(border=True):
            st.markdown("**Why flagged**")
            st.markdown(bullet_list([x.capitalize() for x in reasons(rec).split(", ")] if reasons(rec) != "-"
                                    else ["No single simple flag; the model combines several factors"]), unsafe_allow_html=True)
            st.markdown("**Suggested support**")
            st.markdown(bullet_list(support_actions(rec)), unsafe_allow_html=True)
    st.markdown('<div class="note">Model: Random Forest on admission and first-semester features, scored out-of-fold '
                '(each student is scored by a model that never saw that student). Reasons and suggested support are '
                'rule-based starting points for the advisor. Decision support, never an automatic decision.</div>',
                unsafe_allow_html=True)


def page_what_if():
    page_header("What-if analysis", "Start from a typical student, change a few details and see how the predicted risk moves.")
    df = load_df()
    model = final_model()
    base = pd.concat([df[CODE_COLS + BINARY_COLS].mode().iloc[0], df[ADMISSION_NUM + SEM1_NUM].median()])

    left, right = st.columns([1, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("**Student details**")
            approved = st.slider("Units approved in 1st semester", 0, int(df["Curricular units 1st sem (approved)"].max()),
                                 int(base["Curricular units 1st sem (approved)"]))
            grade = st.slider("Average grade in 1st semester", 0.0, 20.0,
                              float(round(base["Curricular units 1st sem (grade)"], 1)), step=0.5)
            age = st.slider("Age at enrollment", 17, 70, int(base["Age at enrollment"]))
            t1, t2, t3 = st.columns(3)
            tuition = t1.toggle("Fees up to date", value=bool(base["Tuition fees up to date"]))
            debtor = t2.toggle("Debtor", value=bool(base["Debtor"]))
            scholar = t3.toggle("Scholarship", value=bool(base["Scholarship holder"]))

    row = base.copy()
    row["Curricular units 1st sem (approved)"] = approved
    row["Curricular units 1st sem (grade)"] = grade
    row["Age at enrollment"] = age
    row["Tuition fees up to date"] = int(tuition)
    row["Debtor"] = int(debtor)
    row["Scholarship holder"] = int(scholar)
    p = dict(zip(model.classes_, model.predict_proba(pd.DataFrame([row[FEATURES_B]]))[0]))
    risk = p["Dropout"] * 100
    label, colour = band(risk)

    with right:
        with st.container(border=True):
            st.markdown("**Predicted outcome**")
            st.markdown(gauge(risk, colour, f"{label} dropout risk", "Chance of Dropout for this student profile"), unsafe_allow_html=True)
            data = pd.DataFrame({"Outcome": ["Dropout", "Enrolled", "Graduate"],
                                 "Probability (%)": [p["Dropout"] * 100, p["Enrolled"] * 100, p["Graduate"] * 100]})
            ch = alt.Chart(data).mark_bar(cornerRadiusEnd=4, size=26).encode(
                x=alt.X("Probability (%):Q", scale=alt.Scale(domain=[0, 100])),
                y=alt.Y("Outcome:N", sort=["Dropout", "Enrolled", "Graduate"], title=None),
                color=alt.Color("Outcome:N", legend=None,
                                scale=alt.Scale(domain=["Dropout", "Enrolled", "Graduate"], range=[RED, GREY, TEAL])))
            st.altair_chart(style_chart(ch, 150), width="stretch")
    st.markdown('<div class="note">All other details are held at the typical student (most common category or median).</div>',
                unsafe_allow_html=True)


def page_profiles():
    page_header("Cohort profiles", "Students grouped by similarity (K-means, k = 5). The outcome was not used to build the groups, "
                                   "only to describe them afterwards.")
    stats, n_comp = cluster_profiles()
    overall = (load_df()["Target"] == "Dropout").mean() * 100
    top = stats.iloc[0]
    share_all = top["Students"] * top["Dropout %"] / 100 / (load_df()["Target"] == "Dropout").sum() * 100
    kpi_row([
        ("Highest-risk profile", f"{top['Dropout %']:.0f}%", f"dropout: {top['Profile']}", RED),
        ("Size of that profile", f"{int(top['Students']):,}", f"{top['Students'] / stats['Students'].sum() * 100:.0f}% of students", NAVY),
        ("Share of all dropouts", f"{share_all:.0f}%", "in this one profile", BLUE),
        ("Overall dropout rate", f"{overall:.0f}%", "all students", TEAL)])

    cols = st.columns(len(stats))
    for col, (_, r) in zip(cols, stats.iterrows()):
        small = r["Students"] < 0.03 * stats["Students"].sum()
        colour = GREY if small else (RED if r["Dropout %"] >= 50 else NAVY)
        col.markdown(f'<div class="card" style="opacity:{0.6 if small else 1}"><div class="n">{r["Profile"]}</div>'
                     f'<div class="p" style="color:{colour}">{r["Dropout %"]:.0f}%</div>'
                     f'<div class="d">dropout rate<br>{int(r["Students"]):,} students<br>average age {r["Average age"]:.0f}, '
                     f'{r["Approved units (1st sem)"]:.1f} units approved</div></div>', unsafe_allow_html=True)
    st.write("")

    chart_df = stats.copy()
    chart_df["Highlight"] = np.where(chart_df["Dropout %"] == chart_df["Dropout %"].max(), "Highest risk", "Other")
    bars = alt.Chart(chart_df).mark_bar(cornerRadiusEnd=4, size=24).encode(
        x=alt.X("Dropout %:Q", scale=alt.Scale(domain=[0, 100]), title="Dropout rate (%)"),
        y=alt.Y("Profile:N", sort="-x", title=None, axis=alt.Axis(labelLimit=320)),
        color=alt.Color("Highlight:N", legend=None, scale=alt.Scale(domain=["Highest risk", "Other"], range=[RED, BLUE])))
    rule = alt.Chart(pd.DataFrame({"x": [overall]})).mark_rule(color="#111827", strokeDash=[5, 4]).encode(x="x:Q")
    st.altair_chart(style_chart(bars + rule, 260), width="stretch")
    st.caption("Dashed line: overall dropout rate.")

    st.dataframe(stats.round(1), width="stretch", hide_index=True)
    st.markdown(f'<div class="note">Features: 18 columns, standardised, reduced with PCA to {n_comp} components. The silhouette '
                'score is low (about 0.16), so the groups overlap: these are descriptive profiles, not sharply separated groups. '
                'Profile names are derived from each group\'s averages. Associations, not causes.</div>', unsafe_allow_html=True)


def page_rules():
    page_header("Dropout rules", "Association rules of the form IF A and B THEN Dropout, mined with Apriori.")
    c1, c2 = st.columns(2)
    min_sup = c1.slider("Minimum support", 0.03, 0.15, 0.05, step=0.01)
    min_conf = c2.slider("Minimum confidence", 0.40, 0.80, 0.50, step=0.05)
    rules, n_items = mine_rules(min_sup, min_conf)
    if not len(rules):
        st.info("No rules at these settings. Lower the confidence or support.")
        return
    kpi_row([
        ("Distinct rules", f"{len(rules):,}", f"from {n_items:,} frequent itemsets", NAVY),
        ("Strongest lift", f"{rules['lift'].max():.2f}", "times the base dropout rate", RED),
        ("Highest confidence", f"{rules['confidence'].max() * 100:.0f}%", "of matching students dropped out", TEAL),
        ("Rules shown", f"{min(len(rules), 15)}", "top rules by lift", BLUE)])

    left, right = st.columns([3, 2])
    with left:
        show = pd.DataFrame({
            "Rule: if the student has ... then Dropout": rules["IF (antecedent)"],
            "Support (%)": rules["support"] * 100,
            "Confidence (%)": rules["confidence"] * 100,
            "Lift": rules["lift"]}).head(15)
        st.dataframe(show, width="stretch", hide_index=True, height=520, column_config={
            "Support (%)": st.column_config.NumberColumn(format="%.1f"),
            "Confidence (%)": st.column_config.NumberColumn(format="%.1f"),
            "Lift": st.column_config.NumberColumn(format="%.2f")})
    with right:
        sc = alt.Chart(rules.assign(**{"Support (%)": rules["support"] * 100, "Confidence (%)": rules["confidence"] * 100})).mark_circle(
            size=70, opacity=0.75).encode(
            x=alt.X("Support (%):Q", title="Support (%)"), y=alt.Y("Confidence (%):Q", scale=alt.Scale(zero=False)),
            color=alt.Color("lift:Q", title="Lift", scale=alt.Scale(scheme="blues")),
            tooltip=["IF (antecedent)", "support", "confidence", "lift"])
        st.altair_chart(style_chart(sc, 400), width="stretch")
        st.caption("Each dot is a rule. Further up and to the right is stronger and more common.")
    st.markdown('<div class="note"><b>Support:</b> share of all students who have the conditions and dropped out. '
                '<b>Confidence:</b> among students with the conditions, the share who dropped out. '
                '<b>Lift:</b> confidence divided by the overall dropout rate (about 32%); above 1 means more than chance. '
                'Rules with identical numbers are merged. Rules show association, not cause.</div>', unsafe_allow_html=True)


def page_results():
    page_header("Model results", "Classification models compared on admission data alone and with first-semester results "
                                 "(stratified 5-fold cross-validation, macro-F1).")
    kpi_row([
        ("Best model", "SVM", "admission + 1st semester", NAVY),
        ("Test macro-F1", "0.68", "accuracy 0.72", BLUE),
        ("Baseline macro-F1", "0.22", "always predict Graduate", GREY),
        ("Dropout precision / recall", "0.83 / 0.63", "SVM on the test set", RED)])

    res = pd.DataFrame({
        "Model": ["Decision Tree", "Naive Bayes", "Random Forest", "SVM"],
        "Admission only": [0.488, 0.212, 0.507, 0.574],
        "Admission + 1st semester": [0.575, 0.215, 0.623, 0.674]})
    long = res.melt("Model", var_name="Feature set", value_name="Macro-F1")
    bars = alt.Chart(long).mark_bar(cornerRadiusEnd=3).encode(
        x=alt.X("Model:N", title=None, sort=list(res["Model"]), axis=alt.Axis(labelAngle=0)),
        xOffset="Feature set:N",
        y=alt.Y("Macro-F1:Q", scale=alt.Scale(domain=[0, 0.8])),
        color=alt.Color("Feature set:N", legend=alt.Legend(orient="top", title=None),
                        scale=alt.Scale(domain=["Admission only", "Admission + 1st semester"], range=[LIGHT, NAVY])))
    labels = alt.Chart(long).mark_text(dy=-6, fontSize=11, color="#374151").encode(
        x=alt.X("Model:N", sort=list(res["Model"])), xOffset="Feature set:N",
        y="Macro-F1:Q", text=alt.Text("Macro-F1:Q", format=".2f"))
    base = alt.Chart(pd.DataFrame({"y": [0.222]})).mark_rule(color=RED, strokeDash=[5, 4]).encode(y="y:Q")
    st.altair_chart(style_chart(bars + labels + base, 340), width="stretch")
    st.caption("Dashed red line: majority-class baseline (0.222).")

    st.dataframe(res, width="stretch", hide_index=True)
    st.markdown('<div class="note">First-semester results lift every useful model by about 0.06 to 0.10 macro-F1. '
                'Naive Bayes stays near the baseline. The Enrolled class is the hardest to separate. '
                'The dashboard uses a Random Forest because it gives probabilities directly and has the higher Dropout recall.</div>',
                unsafe_allow_html=True)


# ================================================================ layout
PAGES = {"At-risk students": page_risk_list, "What-if analysis": page_what_if, "Cohort profiles": page_profiles,
         "Dropout rules": page_rules, "Model results": page_results}
NAV = list(PAGES)
choice = st.segmented_control("Navigate", NAV, default=NAV[0], key="nav", label_visibility="collapsed") or NAV[0]
st.sidebar.markdown('<div style="font-size:11px;letter-spacing:.14em;text-transform:uppercase;color:#6B7280;font-weight:600">'
                    'About</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div style="font-size:13px;color:#4B5563;line-height:1.65;margin-top:6px"><b style="color:#0B2545">Group 03</b><br>'
                    'Areeba Zahra \u00b7 Qudsia Batool \u00b7 Madeeha Noor<br><br>'
                    'Data: UCI Predict Students\' Dropout and Academic Success (CC BY 4.0).<br><br>'
                    'Decision support for advisors, never an automatic decision.</div>', unsafe_allow_html=True)
PAGES[choice]()
