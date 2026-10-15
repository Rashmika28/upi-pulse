"""
build_dataset.py  --  Phase 1: load & clean the NPCI UPI monthly workbook.

What this script does
---------------------
1. Reads the READ-ONLY raw Excel file in data/raw/ (never writes to it).
2. Cleans the messy text cells (Indian digit grouping, stray tab characters,
   blanks) into proper numbers.
3. Parses the "September-2026" style month labels into real dates.
4. Builds ONE tidy monthly table with the required derived columns.
5. Saves data/processed/upi_monthly.csv.
6. Builds the SQLite database upi.db using sql/schema.sql (table + views),
   then inserts both upi_monthly and the festival calendar.

Units used throughout
---------------------
    volume_mn        = millions of transactions (as published by NPCI)
    value_cr         = Rs crore (as published by NPCI)
    avg_daily_*      = monthly figure / number of calendar days in the month
    avg_ticket_size  = value / volume, converted to Rs per transaction:
                       (value_cr * 1e7 Rs) / (volume_mn * 1e6 txns) = value_cr*10/volume_mn

Run it with:   python src/build_dataset.py
"""

from __future__ import annotations

import calendar
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl

# ---------------------------------------------------------------------------
# Paths (all relative to the project root, one level above this file's folder)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX     = PROJECT_ROOT / "data" / "raw" / "Product-Statistics-UPI-Product-statistics-upi-2016-27-Monthly.xlsx"
SHEET_NAME   = "Product Statistics"
PROCESSED    = PROJECT_ROOT / "data" / "processed"
OUT_CSV      = PROCESSED / "upi_monthly.csv"
FESTIVALS_CSV = PROCESSED / "festivals.csv"
SCHEMA_SQL   = PROJECT_ROOT / "sql" / "schema.sql"
DB_PATH      = PROJECT_ROOT / "upi.db"

# Column names as they appear in the raw sheet (documented so a reader can map
# every output column back to a source column).
RAW_COLUMNS = ["month_raw", "banks_raw", "volume_raw", "value_raw"]

# Month name -> number, e.g. "September" -> 9.
MONTHS = {name: i for i, name in enumerate(calendar.month_name) if name}

# Columns we write to CSV / SQLite, in order.
OUT_COLUMNS = [
    "month", "year", "month_num",
    "volume_mn", "value_cr", "banks_live", "days_in_month",
    "avg_daily_volume", "avg_daily_value", "avg_ticket_size_inr",
    "yoy_volume_pct", "mom_volume_pct", "yoy_value_pct", "mom_value_pct",
]


def clean_number(raw) -> float:
    """Convert one messy text cell into a float.

    Handles every quirk seen in the raw file:
      '29,37,396.67'  Indian digit grouping        -> 2937396.67
      '23201.93'      no commas                    -> 23201.93
      '675\\t'        trailing tab                  -> 675.0
      '29,029,88.05'  grouping typo (April-2026)   -> 2902988.05
      '' / None       blank                        -> NaN
    """
    if raw is None:
        return np.nan
    s = str(raw)
    # remove the characters that break float(): tabs, commas, stray quotes
    s = s.replace("\t", "").replace(",", "").strip().strip("'").strip('"').strip()
    if s == "" or s.lower() == "nan":
        return np.nan
    return float(s)


def parse_month(raw) -> pd.Timestamp:
    """Convert 'September-2026' (newest-first file) into a 1st-of-month Timestamp."""
    name, _, year = str(raw).strip().strip("'").partition("-")
    return pd.Timestamp(int(year), MONTHS[name], 1)


def load_raw(path: Path = RAW_XLSX) -> pd.DataFrame:
    """Read the raw sheet into a dataframe of raw text cells (no cleaning yet)."""
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    ws = wb[SHEET_NAME]
    rows = list(ws.iter_rows(values_only=True))
    wb.close()  # release the file handle; raw file is only ever read
    header, data = rows[0], rows[1:]
    return pd.DataFrame(data, columns=RAW_COLUMNS)


def build_table(path: Path = RAW_XLSX) -> pd.DataFrame:
    """Load, clean, sort oldest-first, and add every derived column."""
    df = load_raw(path)

    # --- parse the four raw columns -----------------------------------------
    df["month"]      = df["month_raw"].map(parse_month)
    df["banks_live"] = df["banks_raw"].map(clean_number)
    df["volume_mn"]  = df["volume_raw"].map(clean_number)
    df["value_cr"]   = df["value_raw"].map(clean_number)

    # keep only what matters and sort oldest -> newest (file is newest -> oldest)
    df = (df[["month", "volume_mn", "value_cr", "banks_live"]]
          .sort_values("month")
          .reset_index(drop=True))

    # --- calendar fields ----------------------------------------------------
    df["year"]          = df["month"].dt.year
    df["month_num"]     = df["month"].dt.month
    df["days_in_month"] = df["month"].dt.days_in_month

    # --- daily averages (short months inflate daily rates, so normalise) -----
    df["avg_daily_volume"] = df["volume_mn"] / df["days_in_month"]
    df["avg_daily_value"]  = df["value_cr"] / df["days_in_month"]

    # --- average ticket size in rupees (value / volume, unit-corrected) ------
    # Guard against divide-by-zero (2016 has zero-volume months -> NaN).
    df["avg_ticket_size_inr"] = np.where(
        df["volume_mn"] > 0, df["value_cr"] * 10.0 / df["volume_mn"], np.nan
    )

    # --- growth rates -------------------------------------------------------
    # YoY = vs the same month last year (12 rows back); MoM = 1 row back.
    df["yoy_volume_pct"] = df["volume_mn"].pct_change(12, fill_method=None) * 100
    df["mom_volume_pct"] = df["volume_mn"].pct_change(1,  fill_method=None) * 100
    df["yoy_value_pct"]  = df["value_cr"].pct_change(12,  fill_method=None) * 100
    df["mom_value_pct"]  = df["value_cr"].pct_change(1,   fill_method=None) * 100

    # 2016 contains zero values, which make some growth rates infinite; mask them.
    df = df.replace([np.inf, -np.inf], np.nan)

    # store month as a clean 'YYYY-MM-DD' string for CSV / SQLite
    df["month"] = df["month"].dt.strftime("%Y-%m-%d")
    return df[OUT_COLUMNS]


def _to_db_value(v):
    """SQLite has no NaN; convert it to None (NULL) before inserting."""
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def load_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    """Create the schema (table + views) and insert the monthly data."""
    schema = SCHEMA_SQL.read_text(encoding="utf-8")
    con = sqlite3.connect(db_path)
    try:
        con.executescript(schema)  # drops & recreates upi_monthly and festivals

        placeholders = ",".join("?" * len(OUT_COLUMNS))
        insert = f"INSERT INTO upi_monthly ({','.join(OUT_COLUMNS)}) VALUES ({placeholders})"
        rows = [tuple(_to_db_value(v) for v in rec)
                for rec in df[OUT_COLUMNS].itertuples(index=False, name=None)]
        con.executemany(insert, rows)
        con.commit()
    finally:
        con.close()


def load_festivals_sqlite(db_path: Path = DB_PATH, csv_path: Path = FESTIVALS_CSV) -> int:
    """Load the festival calendar CSV into the `festivals` table. Returns row count."""
    if not csv_path.exists():
        return 0
    fest = pd.read_csv(csv_path)
    con = sqlite3.connect(db_path)
    try:
        con.executemany(
            "INSERT INTO festivals (festival, year, start_date, end_date, source) VALUES (?,?,?,?,?)",
            list(fest[["festival", "year", "start_date", "end_date", "source"]]
                 .itertuples(index=False, name=None)),
        )
        con.commit()
    finally:
        con.close()
    return len(fest)


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)

    if not RAW_XLSX.exists():
        raise SystemExit(f"Raw file not found: {RAW_XLSX}")

    df = build_table()

    # Save the tidy CSV (UTF-8, no index column).
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")

    # Load into SQLite using the schema file.
    load_sqlite(df)
    n_fest = load_festivals_sqlite()

    # ---- console summary so the user can see exactly what was produced ----
    print(f"Rows written            : {len(df)}")
    print(f"Month range             : {df['month'].iloc[0]}  ->  {df['month'].iloc[-1]}")
    print(f"CSV                     : {OUT_CSV}")
    print(f"DB                      : {DB_PATH}  (festivals rows: {n_fest})")
    print(f"Null counts per column  :\n{df.isna().sum().to_string()}")
    print("\nHead (5 oldest):")
    print(df.head(5).to_string(index=False))
    print("\nTail (5 newest):")
    print(df.tail(5).to_string(index=False))


if __name__ == "__main__":
    main()
