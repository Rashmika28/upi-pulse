"""
make_forecast.py  --  Phase 4: publishable Oct/Nov 2026 forecasts.

What this produces
------------------
forecasts/forecast_2026-10-15.csv, one row per
(month, target, model) with:
    month, target, model, point, lower, upper, trained_through,
    interval_method, median_signed_error_pct, point_in_range,
    designated_primary

Targets (units)
---------------
    avg_daily_volume   millions of transactions per day
    monthly_volume_mn  millions of transactions in the month
    avg_daily_value    Rs crore per day
    monthly_value_cr   Rs crore in the month

Models (all four are shipped for both series)
---------------------------------------------
    ETS_damped    Holt ETS, additive DAMPED trend, no seasonal
    b2            ETS damped trend + additive seasonality
    naive+growth  seasonal naive x recent YoY growth
    SARIMA        SARIMA(1,1,1)(1,0,0,12) on the log series

Primary models: FIXED RULE (decided before looking at the result)
----------------------------------------------------------------
For each target we pick the model with the LOWEST festival-season MAPE
(2019-2025, Oct+Nov pooled) whose median signed log error is closest to zero;
ties break toward the simpler model. We do NOT choose by overall (all-months)
MAPE. See `choose_primaries`.

Prediction intervals (honest, not a 95% CI)
-------------------------------------------
Intervals are the EMPIRICAL RANGE of each model's own festival-season backtest
errors, not a nominal 95% CI. The festival backtest runs one origin per year
2019..2025: train on everything up to September of that year (no later data),
then forecast October (step 1) and November (step 2). That gives n = 7 signed
log-scale errors per model per month. For the 2026-10 row we use the 7 October
errors; for 2026-11 the 7 November errors:

    error_i   = log(actual_i) - log(point_i)
    lower     = point * exp(min_i error_i)
    upper     = point * exp(max_i error_i)

WARNING: because the range keeps the SIGN, a range that lies ENTIRELY ABOVE or
ENTIRELY BELOW the point means the model has a systematic festival-month bias;
it is NOT evidence that the forecast is fine. `point_in_range` flags this.

Run it with:   python src/make_forecast.py

The output CSV is treated as FROZEN: if it already exists the script refuses
to overwrite it and exits with an error. Pass --force to overwrite anyway.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import backtest as bt  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT_CSV = PROJECT_ROOT / "forecasts" / "forecast_2026-10-15.csv"

FORECAST_MONTHS = pd.to_datetime(["2026-10-01", "2026-11-01"])
HORIZON_DAYCOUNT = {pd.Timestamp("2026-10-01"): 31, pd.Timestamp("2026-11-01"): 30}
TRAINED_THROUGH = pd.Timestamp("2026-09-01")

# Festival-season backtest origins (train through September, forecast Oct/Nov).
BACKTEST_YEARS = list(range(2019, 2026))   # 2019..2025  -> n = 7
CHECK_YEARS = list(range(2022, 2026))      # 2022..2025  -> check window

MODELS = {
    "ETS_damped": lambda t, f, c: bt.model_ets_damped(t, f, c, seasonal=False),
    "b2": lambda t, f, c: bt.model_ets_damped(t, f, c, seasonal=True),
    "naive+growth": bt.model_seasonal_naive_growth,
    "SARIMA": bt.model_sarima,
}
MODEL_ORDER = ["ETS_damped", "b2", "naive+growth", "SARIMA"]
# Lower = simpler. Used only as the FINAL tie-break in the primary rule.
SIMPLICITY = {"naive+growth": 0, "ETS_damped": 1, "b2": 2, "SARIMA": 3}

TARGETS = {
    "avg_daily_volume": ("avg_daily_volume", 1.0),
    "monthly_volume_mn": ("avg_daily_volume", None),   # None -> use days in month
    "avg_daily_value": ("avg_daily_value", 1.0),
    "monthly_value_cr": ("avg_daily_value", None),
}
TARGET_ORDER = ["avg_daily_volume", "monthly_volume_mn",
                "avg_daily_value", "monthly_value_cr"]
SERIES_ORDER = ["avg_daily_volume", "avg_daily_value"]

INTERVAL_NOTE = "empirical backtest error range (n={n}, 2019-2025), not a 95% CI"


# =============================================================================
# Helpers
# =============================================================================
def build_context(months: pd.DatetimeIndex, festivals: pd.DataFrame) -> bt.ModelContext:
    flags = bt.build_festival_flags(months, festivals)
    return bt.ModelContext(flags=flags, growth_window=3, sims=300, random_state=0)


def _log_series(monthly: pd.DataFrame, series_col: str) -> pd.Series:
    """Strictly-positive monthly series as a log, indexed by month."""
    s = (monthly.loc[monthly[series_col] > 0, ["month", series_col]]
         .set_index("month")[series_col].astype(float).sort_index())
    return np.log(s)


def festival_backtest_errors(monthly: pd.DataFrame, series_col: str, ctx: bt.ModelContext,
                             years: list[int]) -> dict:
    """Signed log errors per model for Oct (step 1) and Nov (step 2), per origin.

    For year Y we train ONLY on months up to and including September Y, then
    forecast October Y and November Y (no look-ahead).
    Returns {model: {'oct': [...], 'nov': [...]}}.
    """
    logy = _log_series(monthly, series_col)
    out = {m: {"oct": [], "nov": []} for m in MODELS}
    for year in years:
        train = logy[logy.index <= pd.Timestamp(year, 9, 1)]
        future = pd.to_datetime([f"{year}-10-01", f"{year}-11-01"])
        for name, fn in MODELS.items():
            try:
                res = fn(train, future, ctx)
            except Exception:
                continue
            point = np.asarray(res["point"], dtype=float)
            for step, key in ((0, "oct"), (1, "nov")):
                actual = logy.get(future[step], np.nan)
                if np.isfinite(point[step]) and np.isfinite(actual):
                    out[name][key].append(float(actual - point[step]))
    return out


def _stats(errs: list[float]) -> dict:
    """MAPE(%), median signed log error, and whether zero is inside the range."""
    e = np.asarray(errs, dtype=float)
    if e.size == 0:
        return dict(n=0, mape=np.nan, median_log=np.nan, median_pct=np.nan,
                    point_in_range=False)
    mape = float(np.mean(np.abs((np.exp(e) - 1.0) / np.exp(e))) * 100.0)
    med = float(np.median(e))
    return dict(n=int(e.size), mape=mape, median_log=med,
                median_pct=(np.exp(med) - 1.0) * 100.0,
                point_in_range=bool(e.min() <= 0.0 <= e.max()))


def summarise_festival(monthly: pd.DataFrame, ctx: bt.ModelContext,
                       years: list[int]) -> pd.DataFrame:
    """Long table of festival-season backtest stats per target/model/month."""
    rows = []
    for series in SERIES_ORDER:
        errs = festival_backtest_errors(monthly, series, ctx, years)
        for model in MODEL_ORDER:
            for key, label in (("oct", "Oct (h1)"), ("nov", "Nov (h2)")):
                st = _stats(errs[model][key])
                rows.append(dict(target=series, model=model, month=label, **st))
    return pd.DataFrame(rows)


def choose_primaries(monthly: pd.DataFrame, ctx: bt.ModelContext) -> dict:
    """Apply the fixed primary rule and print the reasoning.

    Rule: lowest festival-season MAPE (2019-2025, Oct+Nov pooled); tie-break by
    |median signed log error| closest to zero; final tie-break = simpler model.
    """
    print("PRIMARY-MODEL RULE  (fixed before looking)")
    print("Pick lowest festival-season MAPE (2019-2025, Oct+Nov pooled);")
    print("tie-break: |median signed log error| closest to zero, then simpler.\n")
    chosen: dict[str, str] = {}
    for series in SERIES_ORDER:
        errs = festival_backtest_errors(monthly, series, ctx, BACKTEST_YEARS)
        table = []
        for model in MODEL_ORDER:
            e = np.asarray(errs[model]["oct"] + errs[model]["nov"], dtype=float)
            mape = float(np.mean(np.abs((np.exp(e) - 1.0) / np.exp(e))) * 100.0)
            med = float(np.median(e))
            table.append(dict(model=model, mape=mape, med=med, abs_med=abs(med),
                              simple=SIMPLICITY[model]))
        table.sort(key=lambda r: (round(r["mape"], 6), round(r["abs_med"], 6),
                                  r["simple"]))
        winner = table[0]
        chosen[series] = winner["model"]
        print(f"  {series}:")
        for r in table:
            mark = "  <-- chosen" if r["model"] == winner["model"] else ""
            print(f"    {r['model']:13s} festival MAPE={r['mape']:6.2f}%  "
                  f"median signed log err={r['med']:+.4f}  (|.|={r['abs_med']:.4f}){mark}")
        print(f"    -> {winner['model']} wins: lowest festival MAPE "
              f"({winner['mape']:.2f}%) and its median error is closest to zero "
              f"({winner['med']:+.4f}) among the low-MAPE candidates.\n")
    return chosen


def final_point_forecasts(monthly: pd.DataFrame, ctx: bt.ModelContext) -> dict:
    """Fit each model on ALL data through 2026-09 and forecast Oct/Nov 2026."""
    result: dict[str, dict] = {}
    for series in SERIES_ORDER:
        logy = _log_series(monthly, series)
        train = logy[logy.index <= TRAINED_THROUGH]
        result[series] = {name: np.asarray(fn(train, FORECAST_MONTHS, ctx)["point"],
                                           dtype=float)
                          for name, fn in MODELS.items()}
    return result


def build_rows(monthly: pd.DataFrame, ctx: bt.ModelContext, primaries: dict) -> pd.DataFrame:
    errors = {s: festival_backtest_errors(monthly, s, ctx, BACKTEST_YEARS)
              for s in SERIES_ORDER}
    points = final_point_forecasts(monthly, ctx)

    rows = []
    for m_idx, month in enumerate(FORECAST_MONTHS):
        month_key = "oct" if m_idx == 0 else "nov"
        days = HORIZON_DAYCOUNT[month]
        for target in TARGET_ORDER:
            series_col, scale = TARGETS[target]
            for model in MODEL_ORDER:
                point_log = points[series_col][model][m_idx]
                errs = errors[series_col][model][month_key]
                st = _stats(errs)
                mult = days if scale is None else 1.0
                point = float(np.exp(point_log) * mult)
                lower = float(np.exp(point_log + np.min(errs)) * mult) if errs else np.nan
                upper = float(np.exp(point_log + np.max(errs)) * mult) if errs else np.nan

                rows.append(dict(
                    month=month.strftime("%Y-%m"),
                    target=target,
                    model=model,
                    point=round(point, 2),
                    lower=round(lower, 2) if np.isfinite(lower) else np.nan,
                    upper=round(upper, 2) if np.isfinite(upper) else np.nan,
                    trained_through=TRAINED_THROUGH.strftime("%Y-%m"),
                    interval_method=INTERVAL_NOTE.format(n=st["n"]),
                    median_signed_error_pct=round(st["median_pct"], 2),
                    point_in_range=st["point_in_range"],
                    designated_primary=bool(model == primaries[series_col]),
                ))
    return pd.DataFrame(rows)


def primary_mom_yoy(monthly: pd.DataFrame, rows: pd.DataFrame, primaries: dict
                    ) -> pd.DataFrame:
    """MoM / YoY (%) implied by each PRIMARY forecast, on a daily basis."""
    actual = monthly.set_index("month")
    out = []
    for target in SERIES_ORDER:
        sub = rows[(rows["target"] == target) & rows["designated_primary"]]
        f = {r["month"]: r["point"] for _, r in sub.iterrows()}
        prev_month_actual = actual.loc[pd.Timestamp("2026-09-01"), target]
        for m_idx, month in enumerate(FORECAST_MONTHS):
            mkey = month.strftime("%Y-%m")
            prev_year_actual = actual.loc[month - pd.DateOffset(years=1), target]
            prev_for_mom = prev_month_actual if m_idx == 0 else f[FORECAST_MONTHS[0].strftime("%Y-%m")]
            out.append(dict(
                target=target, primary_model=primaries[target], month=mkey,
                forecast_daily=round(f[mkey], 2),
                mom_pct=round((f[mkey] / prev_for_mom - 1) * 100, 2),
                yoy_pct=round((f[mkey] / prev_year_actual - 1) * 100, 2),
            ))
    return pd.DataFrame(out)


def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Generate the published Oct/Nov 2026 UPI forecast CSV.")
    ap.add_argument("--force", action="store_true",
                    help="overwrite the output CSV if it already exists")
    return ap.parse_args(argv)


def main(force: bool = False) -> None:
    if OUT_CSV.exists() and not force:
        print(
            f"ERROR: {OUT_CSV} already exists.\n"
            "The published forecast file is frozen; refusing to overwrite it.\n"
            "Re-run with --force to overwrite.",
            file=sys.stderr,
        )
        sys.exit(1)

    monthly = pd.read_csv(PROCESSED / "upi_monthly.csv", parse_dates=["month"])
    festivals = pd.read_csv(PROCESSED / "festivals.csv",
                            parse_dates=["start_date", "end_date"])
    months = pd.date_range(monthly["month"].min(), "2026-12-01", freq="MS")
    ctx = build_context(months, festivals)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", None)

    backtest = summarise_festival(monthly, ctx, BACKTEST_YEARS)
    check = summarise_festival(monthly, ctx, CHECK_YEARS)
    primaries = choose_primaries(monthly, ctx)

    rows = build_rows(monthly, ctx, primaries)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_CSV, index=False)

    def _show(title: str, tab: pd.DataFrame) -> None:
        print(title)
        print("=" * 120)
        print(tab.round(4).to_string(index=False))
        print()

    _show("A. FESTIVAL-SEASON BACKTEST  (origins 2019-2025, train through Sep each year)",
          backtest)
    _show("B. CHECK WINDOW  (origins 2022-2025; shorter history, closer to today)",
          check)
    print("C. PRIMARY MODELS")
    print("=" * 120)
    for series in SERIES_ORDER:
        print(f"  {series:18s} -> {primaries[series]}")

    print(f"\n\nWrote {OUT_CSV}  ({len(rows)} rows)\n")
    print("D. FULL FORECAST CSV")
    print("=" * 120)
    print(rows.to_string(index=False))
    print("\n\nE. MoM / YoY implied by the PRIMARY forecast (daily basis, %)")
    print("=" * 120)
    print(primary_mom_yoy(monthly, rows, primaries).to_string(index=False))


if __name__ == "__main__":
    args = parse_args()
    main(force=args.force)
