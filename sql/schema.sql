-- ============================================================================
-- upi_pulse  |  sql/schema.sql
-- ----------------------------------------------------------------------------
-- Creates the cleaned monthly table plus analytical views.
-- This file is executed automatically by src/build_dataset.py, so the schema
-- lives in ONE place. Safe to re-run: the tables are dropped and rebuilt.
-- ============================================================================

DROP TABLE IF EXISTS upi_monthly;
CREATE TABLE upi_monthly (
    month               TEXT PRIMARY KEY,  -- first day of the month, 'YYYY-MM-DD'
    year                INTEGER NOT NULL,
    month_num           INTEGER NOT NULL,  -- 1..12
    volume_mn           REAL,              -- UPI volume, MILLIONS of transactions
    value_cr            REAL,              -- UPI value, Rs CRORE
    banks_live          INTEGER,           -- no. of banks live on UPI
    days_in_month       INTEGER NOT NULL,  -- calendar days in the month
    avg_daily_volume    REAL,              -- volume_mn / days_in_month   (Mn / day)
    avg_daily_value     REAL,              -- value_cr  / days_in_month   (Cr / day)
    avg_ticket_size_inr REAL,              -- Rs per transaction = value / volume
    yoy_volume_pct      REAL,              -- % growth vs same month last year
    mom_volume_pct      REAL,              -- % growth vs previous month
    yoy_value_pct       REAL,
    mom_value_pct       REAL
);

-- Festival calendar (tentative dates for the user to verify).
DROP TABLE IF EXISTS festivals;
CREATE TABLE festivals (
    festival   TEXT    NOT NULL,
    year       INTEGER NOT NULL,
    start_date TEXT    NOT NULL,   -- 'YYYY-MM-DD'
    end_date   TEXT    NOT NULL,   -- 'YYYY-MM-DD'
    source     TEXT,
    PRIMARY KEY (festival, year)
);

-- ----------------------------------------------------------------------------
-- Growth recomputed with WINDOW FUNCTIONS (LAG) straight from the table rows.
-- This is deliberately independent of the stored *_pct columns so the two can
-- cross-check each other in an interview.
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_upi_growth;
CREATE VIEW v_upi_growth AS
SELECT
    month,
    volume_mn,
    value_cr,
    ROUND(100.0 * (volume_mn / LAG(volume_mn,  1) OVER (ORDER BY month) - 1), 2) AS mom_volume_pct_w,
    ROUND(100.0 * (volume_mn / LAG(volume_mn, 12) OVER (ORDER BY month) - 1), 2) AS yoy_volume_pct_w,
    ROUND(100.0 * (value_cr  / LAG(value_cr,   1) OVER (ORDER BY month) - 1), 2) AS mom_value_pct_w,
    ROUND(100.0 * (value_cr  / LAG(value_cr,  12) OVER (ORDER BY month) - 1), 2) AS yoy_value_pct_w
FROM upi_monthly;

-- ----------------------------------------------------------------------------
-- Centered 12-month moving average = the "trend" used later to measure the
-- festival uplift. 5 preceding + current + 6 following = 12 rows.
-- (NULL for the first 6 and last 6 rows where a full centered window is absent.)
-- ----------------------------------------------------------------------------
DROP VIEW IF EXISTS v_upi_trend;
CREATE VIEW v_upi_trend AS
SELECT
    month,
    year,
    month_num,
    volume_mn,
    value_cr,
    avg_daily_volume,
    avg_daily_value,
    avg_ticket_size_inr,
    AVG(volume_mn) OVER (ORDER BY month ROWS BETWEEN 5 PRECEDING AND 6 FOLLOWING) AS trend_volume_mn,
    AVG(value_cr)  OVER (ORDER BY month ROWS BETWEEN 5 PRECEDING AND 6 FOLLOWING) AS trend_value_cr
FROM upi_monthly;
