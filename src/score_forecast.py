"""
score_forecast.py  --  Phase 5: score published forecasts against hand-entered actuals.

What this does
--------------
Takes a small actuals file that YOU create by hand (the original raw workbook in
data/raw/ is never modified), computes the actual average-daily volume and value
for that month, and compares them against every model in a published forecast
CSV. It writes one scorecard per month:

    forecasts/scorecard_YYYY-MM.csv

Input actuals format  (data/raw/actuals_YYYY-MM.csv)
----------------------------------------------------
    month,volume_mn,value_cr
    2026-10,25100.00,3100000.00

    month      : YYYY-MM or YYYY-MM-DD (anything pandas can parse)
    volume_mn  : monthly UPI volume, in millions of transactions (NPCI units)
    value_cr   : monthly UPI value, in Rs crore (NPCI units)

    Optional columns:
      days_in_month  : override the calendar day count (else derived from month)
      avg_daily_volume / avg_daily_value : use these directly if you already
                                           have per-day figures.

    Extra columns are ignored. Only ONE month per file.

Scorecard columns
-----------------
    month, target, model, designated_primary, point, actual, error_pct,
    abs_error_pct, lower, upper, in_range, is_closest

    error_pct = (actual - point) / point * 100   (positive = actual above forecast)
    in_range  : did the actual land inside [lower, upper]?
    is_closest: did this model have the smallest |error_pct| for that target?

Usage
-----
    python src/score_forecast.py data/raw/actuals_2026-10.csv
    python src/score_forecast.py                     # newest data/raw/actuals_*.csv

Options
-------
    --forecast PATH   forecast CSV to score against
                      (default: forecasts/forecast_2026-10-15.csv)
    --out PATH        scorecard CSV path
                      (default: forecasts/scorecard_<month>.csv)

A missing actuals file, or a month the forecast file does not cover, is reported
politely and exits 0 without writing a misleading scorecard.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FORECASTS_DIR = PROJECT_ROOT / "forecasts"
DEFAULT_FORECAST = FORECASTS_DIR / "forecast_2026-10-15.csv"

# Only the two per-day targets are scored; the monthly_* targets are just these
# multiplied by the number of days in the month.
SCORED_TARGETS = ["avg_daily_volume", "avg_daily_value"]
MODELS = ["ETS_damped", "b2", "naive+growth", "SARIMA"]


def load_actuals(path: Path) -> dict:
    """Read one hand-made actuals row and return the month + daily averages."""
    raw = pd.read_csv(path)
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    if "month" not in raw.columns:
        raise ValueError(f"{path.name}: expected a 'month' column "
                         f"(got {list(raw.columns)})")
    if len(raw) == 0:
        raise ValueError(f"{path.name}: no data rows")

    row = raw.iloc[-1]
    month = pd.Timestamp(row["month"]).to_period("M").to_timestamp()
    days = (int(row["days_in_month"]) if "days_in_month" in raw.columns
            else month.days_in_month)

    def daily(daily_col: str, monthly_col: str) -> float:
        if daily_col in raw.columns and pd.notna(row.get(daily_col)):
            return float(row[daily_col])
        if monthly_col in raw.columns and pd.notna(row.get(monthly_col)):
            return float(row[monthly_col]) / days
        raise ValueError(f"{path.name}: need a '{monthly_col}' or "
                         f"'{daily_col}' column")

    return dict(
        month=month,
        days_in_month=days,
        avg_daily_volume=daily("avg_daily_volume", "volume_mn"),
        avg_daily_value=daily("avg_daily_value", "value_cr"),
        source=path.name,
    )


def newest_actuals() -> Path | None:
    """Newest data/raw/actuals_*.csv, or None if there are none."""
    files = sorted(RAW_DIR.glob("actuals_*.csv"))
    return files[-1] if files else None


def build_scorecard(actuals: dict, forecast: pd.DataFrame) -> pd.DataFrame:
    """One row per (target, model) for the actuals month."""
    month = actuals["month"]
    sub = forecast.copy()
    sub["month"] = pd.to_datetime(sub["month"]).dt.to_period("M").dt.to_timestamp()
    sub = sub[(sub["month"] == month) & (sub["target"].isin(SCORED_TARGETS))]
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for target in SCORED_TARGETS:
        actual = actuals[target]
        tsub = sub[sub["target"] == target]
        if tsub.empty:
            continue
        errs = (tsub["point"].astype(float) - actual).abs()
        # mark ALL models tied for the smallest absolute error
        best = errs.min()
        for _, r in tsub.iterrows():
            point = float(r["point"])
            lower, upper = float(r["lower"]), float(r["upper"])
            error_pct = (actual - point) / point * 100 if point else np.nan
            in_range = (np.isfinite(lower) and np.isfinite(upper)
                        and lower <= actual <= upper)
            rows.append(dict(
                month=month.strftime("%Y-%m"),
                target=target,
                model=r["model"],
                designated_primary=bool(r.get("designated_primary", False)),
                point=round(point, 2),
                actual=round(actual, 2),
                error_pct=round(error_pct, 2),
                abs_error_pct=round(abs(error_pct), 2),
                lower=round(lower, 2) if np.isfinite(lower) else np.nan,
                upper=round(upper, 2) if np.isfinite(upper) else np.nan,
                in_range=bool(in_range),
                is_closest=bool(abs(errs.loc[r.name] - best) < 1e-9),
            ))
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score UPI forecasts against actuals.")
    ap.add_argument("actuals", nargs="?", type=Path,
                    help="data/raw/actuals_YYYY-MM.csv (default: newest)")
    ap.add_argument("--forecast", type=Path, default=DEFAULT_FORECAST,
                    help=f"forecast CSV (default: {DEFAULT_FORECAST.name})")
    ap.add_argument("--out", type=Path, default=None,
                    help="scorecard CSV (default: forecasts/scorecard_<month>.csv)")
    args = ap.parse_args(argv)

    actuals_path = args.actuals or newest_actuals()
    if actuals_path is None or not actuals_path.exists():
        print("No actuals file found. Create one at "
              "data/raw/actuals_YYYY-MM.csv (columns: month,volume_mn,value_cr) "
              "and run again.")
        return 0

    if not args.forecast.exists():
        print(f"Forecast file not found: {args.forecast}")
        return 0

    try:
        actuals = load_actuals(actuals_path)
    except Exception as exc:                      # keep it graceful
        print(f"Could not read actuals: {exc}")
        return 0

    forecast = pd.read_csv(args.forecast)
    card = build_scorecard(actuals, forecast)
    month_str = actuals["month"].strftime("%Y-%m")

    if card.empty:
        covered = sorted(pd.to_datetime(forecast["month"]).dt.strftime("%Y-%m").unique())
        print(f"No forecast rows for {month_str}. "
              f"{args.forecast.name} covers: {', '.join(covered)}. "
              f"Nothing written.")
        return 0

    out = args.out or (FORECASTS_DIR / f"scorecard_{month_str}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    card.to_csv(out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(f"Actuals  : {actuals['source']}  (month {month_str}, "
          f"{actuals['days_in_month']} days)")
    print(f"Forecast : {args.forecast.name}")
    print(f"Wrote    : {out}\n")
    for target in SCORED_TARGETS:
        tsub = card[card["target"] == target]
        if tsub.empty:
            continue
        print(f"{target}  (actual = {tsub['actual'].iloc[0]})")
        print(tsub.drop(columns=["month", "target", "actual"]).to_string(index=False))
        best = tsub[tsub["is_closest"]]
        names = ", ".join(best["model"].tolist())
        print(f"  -> closest: {names}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
