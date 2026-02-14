from __future__ import annotations

import os
import pandas as pd

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
    
def read_events(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # parse timestamps
    df["event_time"] = pd.to_datetime(df["event_time"], errors="coerce")
    # basic validation
    required = {
    "user_id", "session_id", "event_time", "feature_used",
    "engagement_time_sec", "experiment_group", "llm_rating",
    "retained_7d", "retained_30d"
}

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Events missing required columns: {sorted(missing)}")
    return df

def write_csv(df:pd.DataFrame, out_path: str) -> None:
    ensure_dir(os.path.dirname(out_path))
    df.to_csv(out_path, index=False)