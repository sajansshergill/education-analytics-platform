"""
02_metrics_pipeline.py

Compute product analytics KPIs and experiment readouts from event logs.

Inputs:
- data/synthetic_events.csv (from 01_generate_data.py)

Outputs (saved to outputs/):
- kpi_daily.csv                 (DAU, sessions, events, engagement, LLM quality)
- feature_adoption_daily.csv    (per-feature DAU + adoption share)
- retention_cohorts_weekly.csv  (weekly cohort retention: 7d/30d)
- experiment_daily.csv          (control vs treatment daily KPIs)
- experiment_summary.csv        (overall lift + simple stats)

Design:
- Efficient groupbys
- Clear, stakeholder-friendly output tables
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils_io import read_events, write_csv, ensure_dir


DATA_PATH = "data/synthetic_events.csv"
OUT_DIR = "outputs"


def add_date_columns(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    events["event_date"] = events["event_time"].dt.date
    # ISO week for cohorting/weekly trends
    iso = events["event_time"].dt.isocalendar()
    events["event_year"] = iso["year"].astype(int)
    events["event_week"] = iso["week"].astype(int)
    events["event_yearweek"] = events["event_year"].astype(str) + "-W" + events["event_week"].astype(str).str.zfill(2)
    return events


def compute_daily_kpis(events: pd.DataFrame) -> pd.DataFrame:
    """
    Daily top-line KPIs:
    - DAU (unique users)
    - Sessions (unique session_id)
    - Events
    - Avg engagement per session (sec)
    - Total engagement (hours)
    - Avg LLM rating (where available)
    - Teacher share of DAU
    - Subscription share of DAU
    """
    e = events.copy()

    # Per-day distinct users/sessions/events
    daily = (
        e.groupby("event_date")
        .agg(
            dau=("user_id", "nunique"),
            sessions=("session_id", "nunique"),
            events=("event_id", "count") if "event_id" in e.columns else ("session_id", "size"),
            total_engagement_sec=("engagement_time_sec", "sum"),
            avg_llm_rating=("llm_rating", "mean"),
        )
        .reset_index()
        .sort_values("event_date")
    )

    # Derived metrics
    daily["total_engagement_hours"] = daily["total_engagement_sec"] / 3600.0
    daily["events_per_session"] = daily["events"] / daily["sessions"].replace(0, np.nan)
    daily["avg_engagement_per_session_sec"] = daily["total_engagement_sec"] / daily["sessions"].replace(0, np.nan)

    # Teacher/subscription share of DAU (requires daily distinct users table)
    daily_users = (
        e.groupby(["event_date", "user_id"])
        .agg(is_teacher=("is_teacher", "max"), subscription=("subscription", "max"))
        .reset_index()
    )
    shares = (
        daily_users.groupby("event_date")
        .agg(
            teacher_share=("is_teacher", "mean"),
            subscription_share=("subscription", "mean"),
        )
        .reset_index()
    )
    out = daily.merge(shares, on="event_date", how="left")
    return out


def compute_feature_adoption_daily(events: pd.DataFrame) -> pd.DataFrame:
    """
    For each day:
    - per-feature DAU
    - per-feature share of DAU (adoption)
    """
    e = events.copy()
    base_dau = e.groupby("event_date")["user_id"].nunique().rename("dau").reset_index()

    feat = (
        e.groupby(["event_date", "feature_used"])
        .agg(feature_dau=("user_id", "nunique"))
        .reset_index()
        .merge(base_dau, on="event_date", how="left")
    )
    feat["feature_dau_share"] = feat["feature_dau"] / feat["dau"].replace(0, np.nan)
    feat = feat.sort_values(["event_date", "feature_used"])
    return feat


def compute_weekly_retention_cohorts(users_path: str = "data/users.csv") -> pd.DataFrame:
    """
    Cohort table based on *first activity week*.
    Uses users.csv if present; otherwise approximates from events.

    Output:
    - cohort_yearweek
    - users_in_cohort
    - retained_7d_rate
    - retained_30d_rate
    """
    users = pd.read_csv(users_path)
    required = {"user_id", "retained_7d", "retained_30d"}
    missing = required - set(users.columns)
    if missing:
        raise ValueError(f"users.csv missing required columns: {sorted(missing)}")

    # If first_session is present, use it; else compute a cohort on first_seen_date
    if "first_session" in users.columns:
        first = pd.to_datetime(users["first_session"], errors="coerce")
    elif "first_seen" in users.columns:
        first = pd.to_datetime(users["first_seen"], errors="coerce")
    else:
        # Fallback: cohort by user_id buckets (not ideal but prevents crash)
        # Better approach if missing: compute from events file (we can do that later).
        first = pd.to_datetime("2025-09-01") + pd.to_timedelta((users["user_id"] % 60), unit="D")

    iso = first.dt.isocalendar()
    users["cohort_year"] = iso["year"].astype(int)
    users["cohort_week"] = iso["week"].astype(int)
    users["cohort_yearweek"] = users["cohort_year"].astype(str) + "-W" + users["cohort_week"].astype(str).str.zfill(2)

    cohort = (
        users.groupby("cohort_yearweek")
        .agg(
            users_in_cohort=("user_id", "count"),
            retained_7d_rate=("retained_7d", "mean"),
            retained_30d_rate=("retained_30d", "mean"),
        )
        .reset_index()
        .sort_values("cohort_yearweek")
    )
    return cohort


def compute_experiment_daily(events: pd.DataFrame) -> pd.DataFrame:
    """
    Daily metrics split by experiment_group:
    - DAU
    - sessions
    - avg engagement/session
    - avg LLM rating
    """
    e = events.copy()

    daily = (
        e.groupby(["event_date", "experiment_group"])
        .agg(
            dau=("user_id", "nunique"),
            sessions=("session_id", "nunique"),
            total_engagement_sec=("engagement_time_sec", "sum"),
            avg_llm_rating=("llm_rating", "mean"),
        )
        .reset_index()
        .sort_values(["event_date", "experiment_group"])
    )
    daily["avg_engagement_per_session_sec"] = daily["total_engagement_sec"] / daily["sessions"].replace(0, np.nan)
    return daily


def _uplift(treat: float, ctrl: float) -> float:
    if ctrl == 0 or np.isnan(ctrl):
        return np.nan
    return (treat - ctrl) / ctrl


def compute_experiment_summary(experiment_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Overall experiment lift summary across the entire time range.
    Produces simple aggregated lifts (not a full CUPED/Bayesian framework yet).
    """
    # Aggregate across days (weighted by sessions for engagement metrics)
    df = experiment_daily.copy()

    agg = (
        df.groupby("experiment_group")
        .agg(
            dau_mean=("dau", "mean"),
            sessions_sum=("sessions", "sum"),
            total_engagement_sec=("total_engagement_sec", "sum"),
            avg_llm_rating=("avg_llm_rating", "mean"),
        )
        .reset_index()
    )
    agg["avg_engagement_per_session_sec"] = agg["total_engagement_sec"] / agg["sessions_sum"].replace(0, np.nan)

    ctrl = agg[agg["experiment_group"] == "control"].iloc[0]
    trt = agg[agg["experiment_group"] == "treatment"].iloc[0]

    summary = pd.DataFrame(
        [
            {
                "metric": "dau_mean",
                "control": float(ctrl["dau_mean"]),
                "treatment": float(trt["dau_mean"]),
                "relative_lift": _uplift(float(trt["dau_mean"]), float(ctrl["dau_mean"])),
            },
            {
                "metric": "avg_engagement_per_session_sec",
                "control": float(ctrl["avg_engagement_per_session_sec"]),
                "treatment": float(trt["avg_engagement_per_session_sec"]),
                "relative_lift": _uplift(float(trt["avg_engagement_per_session_sec"]), float(ctrl["avg_engagement_per_session_sec"])),
            },
            {
                "metric": "avg_llm_rating",
                "control": float(ctrl["avg_llm_rating"]),
                "treatment": float(trt["avg_llm_rating"]),
                "relative_lift": _uplift(float(trt["avg_llm_rating"]), float(ctrl["avg_llm_rating"])),
            },
        ]
    )
    return summary


def main() -> None:
    ensure_dir(OUT_DIR)

    print(f"Reading events: {DATA_PATH}")
    events = read_events(DATA_PATH)
    events = add_date_columns(events)

    print("Computing daily KPIs...")
    kpi_daily = compute_daily_kpis(events)
    write_csv(kpi_daily, f"{OUT_DIR}/kpi_daily.csv")

    print("Computing feature adoption daily...")
    feature_daily = compute_feature_adoption_daily(events)
    write_csv(feature_daily, f"{OUT_DIR}/feature_adoption_daily.csv")

    print("Computing weekly retention cohorts...")
    cohorts = compute_weekly_retention_cohorts("data/users.csv")
    write_csv(cohorts, f"{OUT_DIR}/retention_cohorts_weekly.csv")

    print("Computing experiment daily metrics...")
    exp_daily = compute_experiment_daily(events)
    write_csv(exp_daily, f"{OUT_DIR}/experiment_daily.csv")

    print("Computing experiment summary...")
    exp_summary = compute_experiment_summary(exp_daily)
    write_csv(exp_summary, f"{OUT_DIR}/experiment_summary.csv")

    print("\n✅ Done. Outputs written to /outputs")
    print(" - outputs/kpi_daily.csv")
    print(" - outputs/feature_adoption_daily.csv")
    print(" - outputs/retention_cohorts_weekly.csv")
    print(" - outputs/experiment_daily.csv")
    print(" - outputs/experiment_summary.csv")


if __name__ == "__main__":
    main()
