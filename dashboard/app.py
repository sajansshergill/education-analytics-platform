import pandas as pd
import streamlit as st
import plotly.express as px

st.set_page_config(page_title="AI Education Analytics", layout="wide")

@st.cache_data
def load_data():
    kpi = pd.read_csv("outputs/kpi_daily.csv")
    feat = pd.read_csv("outputs/feature_adoption_daily.csv")
    exp_daily = pd.read_csv("outputs/experiment_daily.csv")
    exp_summary = pd.read_csv("outputs/experiment_summary.csv")
    cohorts = pd.read_csv("outputs/retention_cohorts_weekly.csv")

    # Optional (will exist after step 3)
    try:
        exp_stats = pd.read_csv("outputs/experiment_stats.csv")
    except Exception:
        exp_stats = None

    kpi["event_date"] = pd.to_datetime(kpi["event_date"])
    feat["event_date"] = pd.to_datetime(feat["event_date"])
    exp_daily["event_date"] = pd.to_datetime(exp_daily["event_date"])

    return kpi, feat, exp_daily, exp_summary, cohorts, exp_stats


kpi, feat, exp_daily, exp_summary, cohorts, exp_stats = load_data()

st.title("📘 AI Education Product Analytics Dashboard")

# -----------------------------
# Top KPIs
# -----------------------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Latest DAU", int(kpi["dau"].iloc[-1]))
col2.metric("Avg LLM Rating", round(kpi["avg_llm_rating"].mean(), 2))
col3.metric("Teacher Share", f"{kpi['teacher_share'].mean()*100:.1f}%")
col4.metric("Subscription Share", f"{kpi['subscription_share'].mean()*100:.1f}%")

st.divider()

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3 = st.tabs(["📈 KPIs", "🧪 Experiments", "🧊 Retention Cohorts"])

# =============================
# TAB 1: KPIs
# =============================
with tab1:
    st.subheader("Daily Active Users (DAU)")
    st.plotly_chart(px.line(kpi, x="event_date", y="dau"), use_container_width=True)

    st.subheader("Avg Engagement per Session (sec)")
    st.plotly_chart(px.line(kpi, x="event_date", y="avg_engagement_per_session_sec"), use_container_width=True)

    st.subheader("Feature Adoption (latest day)")
    latest_date = feat["event_date"].max()
    latest_feat = feat[feat["event_date"] == latest_date].copy()

    st.plotly_chart(
        px.bar(
            latest_feat,
            x="feature_used",
            y="feature_dau_share",
            title=f"Feature DAU Share ({latest_date.date()})",
        ),
        use_container_width=True,
    )

# =============================
# TAB 2: Experiments
# =============================
with tab2:
    st.subheader("Experiment Summary (Lift)")
    st.dataframe(exp_summary, use_container_width=True)

    st.subheader("Engagement Trend by Group")
    st.plotly_chart(
        px.line(
            exp_daily,
            x="event_date",
            y="avg_engagement_per_session_sec",
            color="experiment_group",
        ),
        use_container_width=True,
    )

    st.subheader("LLM Rating Trend by Group")
    st.plotly_chart(
        px.line(
            exp_daily,
            x="event_date",
            y="avg_llm_rating",
            color="experiment_group",
        ),
        use_container_width=True,
    )

    st.subheader("Experiment Statistical Tests (95% CI, p-values)")
    if exp_stats is None:
        st.info("Run: `python src/03_experiment_stats.py` to generate outputs/experiment_stats.csv")
    else:
        # Nice formatting
        show = exp_stats.copy()
        for c in ["control_mean", "treatment_mean", "abs_diff", "ci_low_95", "ci_high_95", "relative_lift", "p_value"]:
            if c in show.columns:
                show[c] = pd.to_numeric(show[c], errors="coerce")
        st.dataframe(show, use_container_width=True)

# =============================
# TAB 3: Retention Cohorts
# =============================
with tab3:
    st.subheader("Weekly Cohort Retention Heatmap")

    # cohorts has: cohort_yearweek, users_in_cohort, retained_7d_rate, retained_30d_rate
    metric = st.selectbox("Choose retention metric", ["retained_7d_rate", "retained_30d_rate"])

    # Convert cohort_yearweek to an ordered index (sort)
    c = cohorts.sort_values("cohort_yearweek").copy()
    c["retention_pct"] = (c[metric] * 100).round(2)

    # Heatmap needs 2D — we'll do a single-row heatmap across cohorts (clean + effective)
    heat = c[["cohort_yearweek", "retention_pct"]].set_index("cohort_yearweek").T

    st.plotly_chart(
        px.imshow(
            heat,
            aspect="auto",
            text_auto=True,
            labels={"x": "Cohort (Year-Week)", "y": "", "color": "Retention %"},
            title=f"{metric.replace('_', ' ').title()} by Cohort",
        ),
        use_container_width=True,
    )

    st.caption("Tip: This is a simple cohort view. Next upgrade: full cohort matrix by week-since-first-session.")
