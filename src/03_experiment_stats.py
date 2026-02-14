from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from utils_io import ensure_dir, write_csv

IN_PATH = "outputs/experiment_daily.csv"
OUT_PATH = "outputs/experiment_stats.csv"


def welch_ci_and_pvalue(x: np.ndarray, y: np.ndarray, alpha: float = 0.05):
    """
    Returns:
      diff = mean(y) - mean(x)  (treatment - control)
      ci_low, ci_high
      p_value (two-sided Welch t-test)
    """
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    if len(x) < 3 or len(y) < 3:
        return np.nan, np.nan, np.nan, np.nan

    mx, my = x.mean(), y.mean()
    diff = my - mx

    # Welch t-test
    tstat, pval = stats.ttest_ind(y, x, equal_var=False, nan_policy="omit")

    # Welch-Satterthwaite df
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    nx, ny = len(x), len(y)
    se = np.sqrt(vx / nx + vy / ny)
    if se == 0:
        return diff, diff, diff, pval

    df_num = (vx / nx + vy / ny) ** 2
    df_den = (vx**2) / (nx**2 * (nx - 1)) + (vy**2) / (ny**2 * (ny - 1))
    df = df_num / df_den if df_den > 0 else nx + ny - 2

    tcrit = stats.t.ppf(1 - alpha / 2, df)
    ci_low = diff - tcrit * se
    ci_high = diff + tcrit * se

    return diff, ci_low, ci_high, pval


def main():
    df = pd.read_csv(IN_PATH)
    df["event_date"] = pd.to_datetime(df["event_date"])

    # Pivot daily metrics into control vs treatment arrays
    metrics = ["dau", "avg_engagement_per_session_sec", "avg_llm_rating"]

    out_rows = []
    for m in metrics:
        pivot = df.pivot(index="event_date", columns="experiment_group", values=m)

        if "control" not in pivot.columns or "treatment" not in pivot.columns:
            raise ValueError(f"Missing control/treatment columns for metric: {m}")

        ctrl = pivot["control"].to_numpy(dtype=float)
        trt = pivot["treatment"].to_numpy(dtype=float)

        diff, lo, hi, p = welch_ci_and_pvalue(ctrl, trt)

        ctrl_mean = np.nanmean(ctrl)
        trt_mean = np.nanmean(trt)
        rel_lift = (trt_mean - ctrl_mean) / ctrl_mean if ctrl_mean and not np.isnan(ctrl_mean) else np.nan

        out_rows.append(
            {
                "metric": m,
                "control_mean": ctrl_mean,
                "treatment_mean": trt_mean,
                "abs_diff": diff,
                "ci_low_95": lo,
                "ci_high_95": hi,
                "relative_lift": rel_lift,
                "p_value": p,
                "n_days": int(np.isfinite(ctrl).sum()),
            }
        )

    out = pd.DataFrame(out_rows)
    ensure_dir("outputs")
    write_csv(out, OUT_PATH)
    print(f"✅ Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
