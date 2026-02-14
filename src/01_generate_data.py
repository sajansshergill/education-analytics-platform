"""
01_generate_data.py

Generate a realistic synthetic event log for an AI Education product:
"AI Study Assistant" (features: summarize, quiz, explain, flashcards, chat)

Outputs:
- data/synthetic_events.csv (event-level log)
- data/users.csv (user-level table)
- data/sessions.csv (session-level table)

Design goals:
- Realistic user heterogeneity (teachers vs students)
- Country mix
- Natural feature preferences
- A/B experiment assignment (control vs treatment)
- LLM quality ratings influenced by behavior + treatment
- Retention labels (7d, 30d) influenced by engagement + quality
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

@dataclass(frozen=True)
class GenConfig:
    seed: int = 42
    n_users: int = 50_000
    start_date: str = "2025-09-01"
    end_date: str = "2025-12-01"

    # sessions/user: drawn from negative binomial-ish mixture
    avg_sessions_per_user: float = 8.0
    max_sessions_per_user: int = 80

    # events/session
    min_events_per_session: int = 2
    max_events_per_session: int = 20

    # experiment split
    treatment_rate: float = 0.5

    out_dir: str = "data"


FEATURES = ["summarize", "quiz", "explain", "flashcards", "chat"]
COUNTRIES = ["US", "IN", "BR", "UK", "CA", "MX", "PH", "DE", "FR", "AU"]


# -----------------------------
# Helpers
# -----------------------------

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _date_range_days(start: str, end: str) -> int:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return int((e - s).days)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _choice_with_probs(rng: np.random.Generator, items: list, probs: np.ndarray, size: int) -> np.ndarray:
    probs = probs / probs.sum()
    return rng.choice(items, size=size, replace=True, p=probs)


def _clip(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.minimum(np.maximum(a, lo), hi)


# -----------------------------
# Core generators
# -----------------------------

def generate_users(cfg: GenConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = cfg.n_users

    # Teacher flag (roughly 12% teachers)
    is_teacher = rng.random(n) < 0.12

    # Country distribution (skewed)
    country_probs = np.array([0.35, 0.18, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05, 0.05, 0.05])
    country = _choice_with_probs(rng, COUNTRIES, country_probs, n)

    # "Ability" and "intent" (latent; affect engagement + quality ratings)
    ability = rng.normal(loc=0.0, scale=1.0, size=n)
    intent = rng.normal(loc=0.0, scale=1.0, size=n)

    # Subscription propensity baseline
    sub_base = -2.0 + 0.35 * ability + 0.25 * intent + 0.35 * is_teacher.astype(float)
    sub_prob = _sigmoid(sub_base)
    subscribed = rng.random(n) < sub_prob

    # Experiment assignment
    experiment_group = np.where(rng.random(n) < cfg.treatment_rate, "treatment", "control")

    users = pd.DataFrame(
        {
            "user_id": np.arange(1, n + 1, dtype=np.int64),
            "is_teacher": is_teacher.astype(int),
            "country": country,
            "ability": ability,
            "intent": intent,
            "subscription": subscribed.astype(int),
            "experiment_group": experiment_group,
        }
    )
    return users


def generate_sessions(cfg: GenConfig, rng: np.random.Generator, users: pd.DataFrame) -> pd.DataFrame:
    """
    Create session table with session_id, user_id, session_start.
    Sessions per user are heterogeneous (some power users).
    """
    n_users = len(users)

    # mixture: most users light, some heavy
    light = rng.poisson(lam=max(cfg.avg_sessions_per_user - 2.0, 1.0), size=n_users)
    heavy = rng.poisson(lam=18.0, size=n_users)
    heavy_mask = rng.random(n_users) < 0.10  # 10% heavy users
    sessions_per_user = np.where(heavy_mask, heavy, light)

    sessions_per_user = np.clip(sessions_per_user, 1, cfg.max_sessions_per_user)

    total_sessions = int(sessions_per_user.sum())

    # allocate sessions
    user_ids = np.repeat(users["user_id"].to_numpy(), sessions_per_user)

    # sample start times across range
    n_days = _date_range_days(cfg.start_date, cfg.end_date)
    start_ts = pd.Timestamp(cfg.start_date)

    # day offsets + time-of-day seconds
    day_offsets = rng.integers(0, max(n_days, 1), size=total_sessions)
    tod_seconds = rng.integers(0, 24 * 3600, size=total_sessions)
    session_start = start_ts + pd.to_timedelta(day_offsets, unit="D") + pd.to_timedelta(tod_seconds, unit="s")

    sessions = pd.DataFrame(
        {
            "session_id": np.arange(1, total_sessions + 1, dtype=np.int64),
            "user_id": user_ids.astype(np.int64),
            "session_start": session_start,
        }
    )

    # sort sessions per user chronologically (more realistic)
    sessions.sort_values(["user_id", "session_start"], inplace=True, ignore_index=True)
    sessions["session_id"] = np.arange(1, len(sessions) + 1, dtype=np.int64)  # re-id after sort
    return sessions


def _feature_probs_for_user(user_row: pd.Series) -> np.ndarray:
    """
    Feature preference:
    - Teachers more likely to use quiz + explain
    - Students more likely to use summarize + chat
    """
    teacher = int(user_row["is_teacher"])
    base = np.array([0.28, 0.20, 0.22, 0.12, 0.18])  # summarize, quiz, explain, flashcards, chat

    if teacher:
        base += np.array([-0.04, 0.06, 0.06, 0.02, -0.10])
    else:
        base += np.array([0.04, -0.03, -0.02, 0.00, 0.01])

    # subscription increases flashcards usage a bit
    if int(user_row["subscription"]) == 1:
        base += np.array([0.00, 0.00, 0.00, 0.04, -0.04])

    base = np.clip(base, 0.02, None)
    return base / base.sum()


def generate_events(cfg: GenConfig, rng: np.random.Generator, users: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """
    Event-level table:
    - Each session has multiple events
    - Each event has feature_used, engagement_time
    - LLM rating for some events (e.g., summarize/quiz/explain/chat)
    """
    # Map user attributes for quick lookup
    users_idx = users.set_index("user_id")

    n_sessions = len(sessions)
    events_per_session = rng.integers(cfg.min_events_per_session, cfg.max_events_per_session + 1, size=n_sessions)

    total_events = int(events_per_session.sum())

    session_ids = np.repeat(sessions["session_id"].to_numpy(), events_per_session)
    user_ids = np.repeat(sessions["user_id"].to_numpy(), events_per_session)

    # event_time within session (0..45 minutes)
    within_seconds = rng.integers(0, 45 * 60, size=total_events)
    session_start_map = sessions.set_index("session_id")["session_start"]
    base_times = session_start_map.loc[session_ids].to_numpy()
    event_time = pd.to_datetime(base_times) + pd.to_timedelta(within_seconds, unit="s")

    # feature_used: depends on user type
    feature_used = np.empty(total_events, dtype=object)

    # engagement time: depends on feature + user intent/ability + treatment
    engagement_time = np.empty(total_events, dtype=np.int32)

    # llm_rating: only for some features (1-5), else NaN
    llm_rating = np.full(total_events, np.nan, dtype=float)

    # Vectorized-ish generation in chunks for speed
    chunk = 200_000
    for start in range(0, total_events, chunk):
        end = min(start + chunk, total_events)
        uids = user_ids[start:end]

        # pull user rows
        u = users_idx.loc[uids].reset_index()

        # choose features per row
        probs_teacher = _feature_probs_for_user(pd.Series({"is_teacher": 1, "subscription": 0}))
        probs_student = _feature_probs_for_user(pd.Series({"is_teacher": 0, "subscription": 0}))

        # Build per-row probs based on is_teacher & subscription (approx)
        # We'll do simple branching with two main distributions + small subscription tweak.
        teacher_mask = u["is_teacher"].to_numpy().astype(bool)
        subs_mask = u["subscription"].to_numpy().astype(bool)

        probs = np.where(teacher_mask[:, None], probs_teacher[None, :], probs_student[None, :]).astype(float)

        # subscription tweak
        probs[subs_mask, :] += np.array([0.00, 0.00, 0.00, 0.03, -0.03])[None, :]
        probs = np.clip(probs, 0.01, None)
        probs = probs / probs.sum(axis=1, keepdims=True)

        # sample features row-wise (multinomial draw)
        # Use cumulative probs and uniform randoms
        r = rng.random(end - start)
        cdf = np.cumsum(probs, axis=1)
        idx = (cdf < r[:, None]).sum(axis=1)
        feats = np.array(FEATURES, dtype=object)[idx]
        feature_used[start:end] = feats

        # engagement baseline by feature
        base_eng = np.select(
            [
                feats == "summarize",
                feats == "quiz",
                feats == "explain",
                feats == "flashcards",
                feats == "chat",
            ],
            [55, 80, 95, 65, 70],
            default=60,
        ).astype(float)

        # user effects
        ability = u["ability"].to_numpy()
        intent = u["intent"].to_numpy()
        teacher = u["is_teacher"].to_numpy().astype(float)
        treatment = (u["experiment_group"].to_numpy() == "treatment").astype(float)

        # treatment improves engagement slightly for quiz/explain (new prompting UX)
        treat_boost = np.where((feats == "quiz") | (feats == "explain"), 8.0 * treatment, 2.0 * treatment)

        # compute engagement time seconds
        eng = base_eng + 8.0 * intent + 4.0 * ability + 6.0 * teacher + treat_boost + rng.normal(0, 12, size=end-start)
        eng = _clip(eng, 5, 15 * 60)  # 5s to 15min
        engagement_time[start:end] = eng.astype(np.int32)

        # LLM ratings for LLM-touch features
        is_llm = (feats != "flashcards")
        # rating depends on ability (higher standards -> slightly lower), intent (more engaged -> higher),
        # treatment improves perceived quality modestly
        quality = 3.6 + 0.25 * intent - 0.12 * ability + 0.20 * treatment + 0.10 * teacher + rng.normal(0, 0.55, size=end-start)
        quality = _clip(quality, 1.0, 5.0)
        llm_rating[start:end] = np.where(is_llm, quality, np.nan)

    events = pd.DataFrame(
        {
            "event_id": np.arange(1, total_events + 1, dtype=np.int64),
            "session_id": session_ids.astype(np.int64),
            "user_id": user_ids.astype(np.int64),
            "event_time": pd.to_datetime(event_time),
            "feature_used": feature_used,
            "engagement_time_sec": engagement_time,
            "llm_rating": llm_rating,
        }
    )

    # add user columns (denormalize for convenience)
    events = events.merge(
        users[["user_id", "is_teacher", "country", "subscription", "experiment_group"]],
        on="user_id",
        how="left",
        validate="many_to_one",
    )

    return events


def add_retention_labels(cfg: GenConfig, rng: np.random.Generator, users: pd.DataFrame, sessions: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create user-level retention labels based on whether they have sessions after +7d and +30d,
    plus a small stochastic component based on latent variables.
    """
    start_ts = pd.Timestamp(cfg.start_date)
    sessions_u = sessions.merge(users[["user_id", "ability", "intent", "experiment_group"]], on="user_id", how="left")

    # first session time
    first = sessions_u.groupby("user_id")["session_start"].min().rename("first_session")
    last = sessions_u.groupby("user_id")["session_start"].max().rename("last_session")

    user_tbl = users.merge(first, on="user_id", how="left").merge(last, on="user_id", how="left")

    # deterministically: session exists after threshold
    # stochastic: user with higher intent more likely retained even if sparse
    has_after_7 = (user_tbl["last_session"] >= (user_tbl["first_session"] + pd.Timedelta(days=7))).astype(int)
    has_after_30 = (user_tbl["last_session"] >= (user_tbl["first_session"] + pd.Timedelta(days=30))).astype(int)

    # stochastic smoothing (some users drop even if they have later sessions; some retain by chance)
    treatment = (user_tbl["experiment_group"] == "treatment").astype(float)
    p7 = _sigmoid(-0.6 + 0.55 * user_tbl["intent"] + 0.15 * user_tbl["ability"] + 0.15 * treatment)
    p30 = _sigmoid(-1.1 + 0.60 * user_tbl["intent"] + 0.10 * user_tbl["ability"] + 0.10 * treatment)

    churn_noise7 = (rng.random(len(user_tbl)) < p7).astype(int)
    churn_noise30 = (rng.random(len(user_tbl)) < p30).astype(int)

    user_tbl["retained_7d"] = ((has_after_7 & churn_noise7) | churn_noise7).astype(int)
    user_tbl["retained_30d"] = ((has_after_30 & churn_noise30) | churn_noise30).astype(int)

    # attach to sessions
    sessions_l = sessions.merge(user_tbl[["user_id", "retained_7d", "retained_30d"]], on="user_id", how="left")
    return user_tbl, sessions_l


def main() -> None:
    cfg = GenConfig()
    rng = np.random.default_rng(cfg.seed)

    _ensure_dir(cfg.out_dir)

    print("Generating users...")
    users = generate_users(cfg, rng)

    print("Generating sessions...")
    sessions = generate_sessions(cfg, rng, users)

    print("Adding retention labels...")
    users_l, sessions_l = add_retention_labels(cfg, rng, users, sessions)

    print("Generating events...")
    events = generate_events(cfg, rng, users_l, sessions_l)

    # merge retention onto events
    events = events.merge(users_l[["user_id", "retained_7d", "retained_30d"]], on="user_id", how="left")

    # Save
    users_path = os.path.join(cfg.out_dir, "users.csv")
    sessions_path = os.path.join(cfg.out_dir, "sessions.csv")
    events_path = os.path.join(cfg.out_dir, "synthetic_events.csv")

    print(f"Saving: {users_path}")
    users_l.to_csv(users_path, index=False)

    print(f"Saving: {sessions_path}")
    sessions_l.to_csv(sessions_path, index=False)

    print(f"Saving: {events_path}")
    events.to_csv(events_path, index=False)

    print("\nDone.")
    print(f"Users:    {len(users_l):,}")
    print(f"Sessions: {len(sessions_l):,}")
    print(f"Events:   {len(events):,}")
    print("\nNext: run src/02_metrics_pipeline.py to compute KPIs.")


if __name__ == "__main__":
    main()
