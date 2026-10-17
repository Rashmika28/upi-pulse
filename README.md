
## Forecast scoring (pending: October and November 2026)
- Create data/raw/actuals_YYYY-MM.csv by hand (columns: month, volume_mn, value_cr).
- Run: python -m src.score_forecast data/raw/actuals_YYYY-MM.csv --out forecasts/scorecard_YYYY-MM.csv

## Limitations
Autumn results use verified Diwali/Dussehra dates; Navratri start dates and non-autumn festivals are unverified, so any all-festivals result is tentative.

## LinkedIn post link
[placeholder]

## Dashboard screenshots
[placeholder]

## Project explained simply
1. Cleaned monthly UPI data (volume/value/days) to get comparable per-day metrics.
2. Built dataset and explored seasonality/trends.
3. Modeled with ETS/backtests to avoid leakage.
4. Produced frozen forecasts (immutable) with tag forecast-2026-10-15; value primary chosen after sanity checks.
5. Scored pending actuals via src/score_forecast.py.
6. Analyzed festival effects with STL (robust) on log series and reported ranges.
7. Flagged anomalies via rolling-origin 1-step errors with robust MAD.
8. Documented verification status (Diwali/Dussehra verified, others tentative).
9. Power BI artifacts prepared (no .pbix).
10. Kept forecasts/data/raw immutable throughout.

## Published forecast

- **File:** forecasts/forecast_2026-10-15.csv  
- **Git tag:** forecast-2026-10-15  
- **Training:** through September 2026  
- **Timing:** made before NPCI published October 2026 data  
- **Primary models:** ETS damped (no seasonality) for volume; naive+growth for value  
- **Model selection:** chosen by festival-season backtest MAPE (Oct/Nov, origins 2019-2025)  
- **Ranges:** empirical backtest error range (n=7, 2019-2025), not a 95% CI. The same backtest both selects the model and sets the range, so the range is somewhat optimistic  
- **Primary-model rule change:** the primary-model rule was changed after the first version (which used overall MAPE and chose b2 for value) failed a sanity check, because its October value forecast fell about 4% below September; the first version was never committed  
- **Integrity:** the forecast files are immutable. The actuals are not yet published, so no accuracy is claimed.
