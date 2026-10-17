# UPI Pulse: Forecasting India's Digital Payments and Festival Effects

## Question
- How much does the festival season lift UPI once the underlying trend is removed?
- Can simple models forecast monthly volume and value, and how accurate are they?

## Data
- NPCI Product Statistics: monthly volume, value and banks live, April 2016 to September 2026 (modeling window April 2021+), downloaded 2026-10-15 from npci.org.in.
- Festival calendar: data/processed/festivals.csv (Diwali and Dussehra verified, others tentative).

## Method
- Clean the NPCI monthly workbook and derive average-per-day metrics (volume/day, value/day, ticket size).
- Run STL festival analysis on the log series, with a window sensitivity check.
- Backtest eight model variants with a leak-free rolling-origin scheme (1-3 months ahead, no look-ahead).
- Publish a pre-release forecast (trained through September 2026) before NPCI released October data.

## Headline results
- Autumn festival months run only a few percent above trend: total trend-adjusted festival lift volume +2.6%, value +2.9%, ticket size +0.15%.
- Value responds to Diwali timing more than volume (suggestive, not established): Oct-vs-Nov value +3.6% (main) / +3.4% (sensitivity); volume +0.6% / -0.9%.
- Simple models forecast monthly UPI well: rolling-origin 1-month MAPE on volume ETS damped 1.1%, naive+growth 1.9%.

## Status
Forecast scoring pending: NPCI October 2026 data not yet published.

## How to rerun
1. `python src/build_dataset.py`
2. `python src/backtest.py`
3. Forecast generation (src/make_forecast.py) is frozen for the published forecast; do not rerun it for forecasts/forecast_2026-10-15.csv (tag forecast-2026-10-15).
4. `python src/score_forecast.py data/raw/actuals_YYYY-MM.csv --out forecasts/scorecard_YYYY-MM.csv`
5. Notebooks: `notebooks/02_eda.ipynb` -> `notebooks/03_models.ipynb` -> `notebooks/04_festival.ipynb`

## Repository layout
- `data/` -- raw NPCI workbook and processed tables (`upi_monthly.csv`, `festivals.csv`).
- `forecasts/` -- published pre-release forecast and forecasting notes.
- `notebooks/` -- EDA, modeling and festival analysis notebooks plus figures.
- `powerbi/` -- exported CSVs and DAX measures for the dashboard.
- `sql/` -- SQLite schema for `upi.db`.
- `src/` -- build, backtest, forecast and scoring scripts.

## Forecast scoring (pending: October and November 2026)
- Create data/raw/actuals_YYYY-MM.csv by hand (columns: month, volume_mn, value_cr).
- Run: python src/score_forecast.py data/raw/actuals_YYYY-MM.csv --out forecasts/scorecard_YYYY-MM.csv

## Limitations
Autumn results use verified Diwali/Dussehra dates; Navratri start dates and non-autumn festivals are unverified, so any all-festivals result is tentative.

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
- **Provenance:** The forecast file was committed to this public GitHub repository on 2026-10-15 (commit d2c5e65, tag forecast-2026-10-15), before NPCI published October 2026 data.  
- **Primary models:** ETS damped (no seasonality) for volume; naive+growth for value  
- **Model selection:** chosen by festival-season backtest MAPE (Oct/Nov, origins 2019-2025)  
- **Ranges:** empirical backtest error range (n=7, 2019-2025), not a 95% CI. The same backtest both selects the model and sets the range, so the range is somewhat optimistic  
- **Primary-model rule change:** the primary-model rule was changed after the first version (which used overall MAPE and chose b2 for value) failed a sanity check, because its October value forecast fell about 4% below September; the first version was never committed  
- **Integrity:** the forecast files are immutable. The actuals are not yet published, so no accuracy is claimed.
