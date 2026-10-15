"""
backtest.py  --  Phase 3: forecasting models + honest rolling-origin backtest.

Everything here is deliberately small, readable and leak-free.

Targets
-------
We model the LOG of a monthly series (default: log of average-daily volume).
Working in logs turns multiplicative growth (UPI roughly x1.2 per year) into a
straight line, which simple models handle well.

Festival features come from data/processed/festivals.csv (real dates), NOT from
assuming "festival = November". Because Diwali moves between October and
November, we map each festival *window* to a single calendar month using a fixed
rule (see `assign_festival_month`). Navratri and Dussehra are merged into one
window to avoid double-counting.

The backtest
------------
`rolling_origin_backtest` trains on an expanding window, starting with
`initial_train` months, and predicts 1..H months ahead. At every origin it uses
ONLY data up to that origin -- so there is no look-ahead leakage. Nothing is
tuned on the test period.

Forecasts are made in log space and converted back to levels with exp() before
any error is measured, so MAPE is a normal "percent error" on the real series.
Prediction intervals (where a model provides them) are also converted to levels
and used to report interval coverage (share of actuals that fell inside).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Festivals we merge to avoid double counting (Navratri leads into Dussehra).
NAVRATRI_GROUP = {"Navratri", "Dussehra"}

# Columns of festival flags we build.
FLAG_COLUMNS = ["diwali", "navratri_dussehra", "other_festival"]

# --------- tournaments are reported for these months as "festival months" -----
FESTIVAL_MONTHS = {10, 11}  # October, November


# =============================================================================
# 1. Festival flags (built from REAL dates)
# =============================================================================
def assign_festival_month(start: pd.Timestamp, end: pd.Timestamp) -> pd.Period:
    """Assign a festival window to exactly ONE calendar month.

    Rule (for festivals that span two months): pick the month that contains the
    MAJORITY of the festival's days; if the days split exactly evenly, pick the
    EARLIER month. This guarantees one month per festival -> no double counting,
    and it is deterministic.

    Examples: Diwali 31 Oct - 1 Nov 2024 -> October (tie broken to earlier).
    """
    days = pd.date_range(start, end, freq="D")
    counts = pd.Series(days).dt.to_period("M").value_counts()
    # sort by (most days, then earliest month)
    best = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return best


def build_festival_flags(months: pd.DatetimeIndex, festivals: pd.DataFrame,
                         merge_navratri_dussehra: bool = True) -> pd.DataFrame:
    """Return a 0/1 flag table (index = month) for each festival group.

    Parameters
    ----------
    months : DatetimeIndex of month-start timestamps to build flags for.
    festivals : the festival calendar with columns festival, year, start_date,
                end_date (as produced/edited by the user).
    merge_navratri_dussehra : merge Navratri + Dussehra into one window.
    """
    flags = pd.DataFrame(0, index=pd.DatetimeIndex(months, name="month"),
                         columns=FLAG_COLUMNS + ["any_festival"])
    for row in festivals.itertuples(index=False):
        name = row.festival
        if merge_navratri_dussehra and name in NAVRATRI_GROUP:
            col = "navratri_dussehra"
        elif name == "Diwali":
            col = "diwali"
        else:
            col = "other_festival"
        period = assign_festival_month(pd.Timestamp(row.start_date), pd.Timestamp(row.end_date))
        stamp = period.to_timestamp()  # first day of the chosen month
        if stamp in flags.index:
            flags.loc[stamp, col] = 1
    flags["any_festival"] = (flags[FLAG_COLUMNS].sum(axis=1) > 0).astype(int)
    return flags


# =============================================================================
# 2. Models
#    Each model is a function  f(train_log, future_dates, ctx) -> dict
#    returning arrays of length h (in LOG space):
#        point, lower, upper   (lower/upper may be nan if no interval)
# =============================================================================
@dataclass
class ModelContext:
    """Bag of things a model may need (festival flags, config)."""
    flags: pd.DataFrame
    growth_window: int = 3          # months of recent YoY used by seasonal-naive
    seasonal_on: bool = True        # for ETS
    random_state: int = 0
    sims: int = 300                 # simulation draws for ETS intervals
    extra: dict = field(default_factory=dict)


def _seasonal_residual_sd(train_log: pd.Series) -> float:
    """Std dev of the (in-sample) seasonal-naive residual, used for intervals."""
    resid = (train_log - train_log.shift(12)).dropna()
    return float(resid.std()) if len(resid) >= 2 else float("nan")


def model_seasonal_naive(train_log: pd.Series, future_dates: pd.DatetimeIndex,
                         ctx: ModelContext) -> dict:
    """Baseline: same month last year, no growth adjustment.

    point[t] = y[t-12].  Interval = point +/- 1.96 * sd * sqrt(step).
    """
    h = len(future_dates)
    sd = _seasonal_residual_sd(train_log)
    point = np.array([train_log.get(d - pd.DateOffset(months=12), np.nan) for d in future_dates])
    step = np.arange(1, h + 1)
    half = 1.96 * sd * np.sqrt(step)
    return dict(point=point, lower=point - half, upper=point + half)


def model_seasonal_naive_growth(train_log: pd.Series, future_dates: pd.DatetimeIndex,
                                ctx: ModelContext) -> dict:
    """(a) Seasonal naive *with growth*: last year's month x recent YoY growth.

    recent growth g = mean of the last `growth_window` log year-on-year changes
    available at the origin. In log space:  point[t] = y[t-12] + g.
    """
    k = ctx.growth_window
    yoy_log = (train_log - train_log.shift(12)).dropna()
    g = float(yoy_log.tail(k).mean())
    point = np.array([train_log.get(d - pd.DateOffset(months=12), np.nan) + g
                      for d in future_dates])
    resid = (train_log - (train_log.shift(12) + g)).dropna()
    sd = float(resid.std()) if len(resid) >= 2 else float("nan")
    step = np.arange(1, len(future_dates) + 1)
    half = 1.96 * sd * np.sqrt(step)
    return dict(point=point, lower=point - half, upper=point + half)


def model_ets_damped(train_log: pd.Series, future_dates: pd.DatetimeIndex,
                     ctx: ModelContext, seasonal: bool = False) -> dict:
    """(b) ETS (Holt) with a DAMPED additive trend; optional additive seasonality.

    A damped trend is chosen because growth is slowing: the trend contribution
    decays towards a constant rather than compounding forever.
    Intervals come from `res.simulate` (residual-based simulation).
    """
    h = len(future_dates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            train_log.values,
            trend="add",
            damped_trend=True,
            seasonal="add" if seasonal else None,
            seasonal_periods=12 if seasonal else None,
            initialization_method="estimated",
        )
        res = model.fit(optimized=True)
        fc = np.asarray(res.forecast(h), dtype=float)
        try:
            # In this statsmodels version random_errors only accepts 'bootstrap'
            # (we resample the in-sample residuals to build uncertainty).
            sim = res.simulate(h, repetitions=ctx.sims, random_errors="bootstrap",
                               rng=np.random.default_rng(ctx.random_state))
            sim = np.asarray(sim, dtype=float)
            lower = np.nanpercentile(sim, 2.5, axis=0)
            upper = np.nanpercentile(sim, 97.5, axis=0)
        except Exception:
            lower = upper = np.full(h, np.nan)
    return dict(point=fc, lower=lower, upper=upper)


def model_regression_trend_festival(train_log: pd.Series, future_dates: pd.DatetimeIndex,
                                    ctx: ModelContext, degree: int = 2) -> dict:
    """(c) OLS regression: damped (polynomial) trend + festival-window flags.

    Fit on the training months; forecast by extrapolating the trend and plugging
    in the *known* festival flags for the future months (festival dates are known
    in advance, so this is not leakage). 95% intervals from OLS prediction.

    `degree` sets the trend shape: 1 = straight line, 2 = quadratic.  The series
    is modelled in LOG space and UPI's log-growth is DECELERATING (concave), so a
    straight line over-shoots badly (~20% MAPE).  degree=2 adds a downside-curved
    ("damped") trend and cuts that to ~3-4%.  Default is therefore 2.
    """
    flags = ctx.flags
    use_cols = ["diwali", "navratri_dussehra", "other_festival"]

    def trend(t: np.ndarray) -> np.ndarray:
        return np.column_stack([t ** p for p in range(1, degree + 1)])

    t_train = np.arange(1, len(train_log) + 1, dtype=float)
    X_train = np.column_stack([trend(t_train), flags.loc[train_log.index, use_cols].values])
    X_train = sm.add_constant(X_train, has_constant="add")

    res = sm.OLS(train_log.values, X_train).fit()

    t_fut = np.arange(len(train_log) + 1, len(train_log) + 1 + len(future_dates), dtype=float)
    X_fut = np.column_stack([trend(t_fut), flags.loc[future_dates, use_cols].values])
    X_fut = sm.add_constant(X_fut, has_constant="add")

    pred = res.get_prediction(X_fut)
    frame = pred.summary_frame(alpha=0.05)
    return dict(point=frame["mean"].values,
                lower=frame["obs_ci_lower"].values,
                upper=frame["obs_ci_upper"].values)


def model_sarima(train_log: pd.Series, future_dates: pd.DatetimeIndex,
                 ctx: ModelContext) -> dict:
    """(c-alt) SARIMA(1,1,1)(1,0,0,12): a seasonal ARIMA on the log series.

    This is the alternative to the festival-flag regression. It captures the
    trend (via d=1) and a yearly seasonal AR term, and returns 95% intervals from
    the state-space forecast. No festival information is used.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    h = len(future_dates)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(train_log.values, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        fc = res.get_forecast(steps=h)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
    return dict(point=mean, lower=ci[:, 0], upper=ci[:, 1])


def model_sarimax_exog(train_log: pd.Series, future_dates: pd.DatetimeIndex,
                       ctx: ModelContext) -> dict:
    """(c3) SARIMAX(1,1,1)(1,0,0,12) with festival-window flags as exogenous.

    Same seasonal ARIMA as model_sarima, but the Navratri+Dussehra and Diwali
    windows enter as exogenous regressors. Future flags are known in advance, so
    using them for the forecast is not leakage. Lets us test whether telling the
    model about festival months helps at all.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    h = len(future_dates)
    use_cols = ["diwali", "navratri_dussehra"]
    exog = ctx.flags.loc[train_log.index, use_cols].values
    exog_fut = ctx.flags.loc[future_dates, use_cols].values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(train_log.values, exog=exog, order=(1, 1, 1),
                      seasonal_order=(1, 0, 0, 12),
                      enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False)
        fc = res.get_forecast(steps=h, exog=exog_fut)
        mean = np.asarray(fc.predicted_mean, dtype=float)
        ci = np.asarray(fc.conf_int(alpha=0.05), dtype=float)
    return dict(point=mean, lower=ci[:, 0], upper=ci[:, 1])


# =============================================================================
# 3. Rolling-origin expanding-window backtest
# =============================================================================
def rolling_origin_backtest(df: pd.DataFrame, target_col: str, models: dict,
                            initial_train: int = 36, max_horizon: int = 3,
                            date_col: str = "month",
                            ctx: ModelContext | None = None) -> pd.DataFrame:
    """Backtest models with an expanding window and no look-ahead leakage.

    Parameters
    ----------
    df : monthly dataframe, unsorted is fine.
    target_col : the (strictly positive) series to model, e.g. 'avg_daily_volume'.
                 It is logged internally.
    models : dict {name: model_function}. Each function has the signature
             (train_log: pd.Series, future_dates: pd.DatetimeIndex, ctx) -> dict
             with keys point, lower, upper (arrays in LOG space, length h).
    initial_train : number of months in the FIRST training window (default 36).
    max_horizon : largest horizon to score (default 3).
    date_col : datetime column used as the time index.
    ctx : ModelContext passed to every model.

    Returns
    -------
    Long dataframe, one row per (origin, model, step) with:
      model, horizon, step, origin (last training month), target_month,
      point, lower, upper, actual,  all on the LEVEL scale,
      plus point_log, actual_log.

    Leakage guarantee
    -----------------
    For an origin whose last training month is month i-1, the model receives
    `train_log` = rows 0..i-1 only, and `future_dates` = months i..i+h-1.
    The actuals for i..i+h-1 are attached AFTER forecasting, purely for scoring.
    """
    data = df.sort_values(date_col).reset_index(drop=True)
    y = data[target_col].astype(float).values
    if np.any(y <= 0):
        raise ValueError("target must be strictly positive (we take its log)")
    logy = pd.Series(np.log(y), index=pd.DatetimeIndex(data[date_col]), name="log_target")

    records = []
    n = len(logy)
    for i in range(initial_train, n):                      # i = first test index
        h = min(max_horizon, n - i)
        if h < 1:
            break
        train = logy.iloc[:i]                              # ONLY past data
        future_dates = logy.index[i:i + h]
        actual_log = logy.iloc[i:i + h].values

        for name, fn in models.items():
            try:
                out = fn(train, future_dates, ctx)
            except Exception:
                continue
            point = np.asarray(out["point"], dtype=float)
            lower = np.asarray(out.get("lower", np.full(h, np.nan)), dtype=float)
            upper = np.asarray(out.get("upper", np.full(h, np.nan)), dtype=float)
            for j in range(h):
                records.append(dict(
                    model=name, horizon=j + 1, step=j + 1,
                    origin=train.index[-1], target_month=future_dates[j],
                    point_log=point[j], lower_log=lower[j], upper_log=upper[j],
                    actual_log=actual_log[j],
                    point=np.exp(point[j]), lower=np.exp(lower[j]),
                    upper=np.exp(upper[j]), actual=np.exp(actual_log[j]),
                ))
    return pd.DataFrame.from_records(records)


# =============================================================================
# 4. Metrics
# =============================================================================
def mape(actual: pd.Series, pred: pd.Series) -> float:
    """Mean absolute percentage error on the LEVEL scale (in %)."""
    actual = np.asarray(actual, float)
    pred = np.asarray(pred, float)
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100)


def interval_coverage(actual: pd.Series, lower: pd.Series, upper: pd.Series) -> float:
    """Share (%) of actuals that fell inside [lower, upper] on the LEVEL scale."""
    actual = np.asarray(actual, float)
    lower = np.asarray(lower, float)
    upper = np.asarray(upper, float)
    ok = np.isfinite(lower) & np.isfinite(upper)
    if ok.sum() == 0:
        return float("nan")
    inside = (actual[ok] >= lower[ok]) & (actual[ok] <= upper[ok])
    return float(np.mean(inside) * 100)


def summarize(bt: pd.DataFrame) -> pd.DataFrame:
    """Build the results table: MAPE per model x horizon, split festival vs not,
    plus interval coverage."""
    out = []
    for (model, horizon), g in bt.groupby(["model", "horizon"]):
        is_fest = g["target_month"].dt.month.isin(FESTIVAL_MONTHS)
        out.append(dict(
            model=model, horizon=horizon, n=len(g),
            mape_all=mape(g["actual"], g["point"]),
            mape_festival=mape(g.loc[is_fest, "actual"], g.loc[is_fest, "point"]),
            mape_nonfestival=mape(g.loc[~is_fest, "actual"], g.loc[~is_fest, "point"]),
            coverage=interval_coverage(g["actual"], g["lower"], g["upper"]),
        ))
    return pd.DataFrame(out)
