# DAX Measures

Load monthly_facts.csv, scorecard_placeholder.csv, festival_uplift.csv, backtest_results.csv, forecast.csv.

## Date table (create first)
Date = CALENDAR(DATE(YEAR(MIN(monthly_facts[month])), 1, 1), DATE(YEAR(MAX(monthly_facts[month])), 12, 31))
Then: Table view -> select Date -> Mark as date table -> column Date.
Relationship: Date[Date] (one) -> monthly_facts[month] (many).
Put Date[Year] and Date[Month] on slicers and chart axes, not monthly_facts[month].

## Base measures
Sum_Volume = SUM(monthly_facts[volume])
Sum_Value = SUM(monthly_facts[value])
Sum_Days = SUM(monthly_facts[days])

## Average per day (fair across 28-31 day months)
Avg_Daily_Volume = DIVIDE([Sum_Volume], [Sum_Days])
Avg_Daily_Value = DIVIDE([Sum_Value], [Sum_Days])

## Ticket size in rupees (value is in crore, volume in million)
Ticket_Size_INR = DIVIDE([Sum_Value] * 10, [Sum_Volume])

## Growth
Avg_Daily_Volume_LY = CALCULATE([Avg_Daily_Volume], SAMEPERIODLASTYEAR('Date'[Date]))
YoY_% = DIVIDE([Avg_Daily_Volume] - [Avg_Daily_Volume_LY], [Avg_Daily_Volume_LY])
Avg_Daily_Volume_PM = CALCULATE([Avg_Daily_Volume], PREVIOUSMONTH('Date'[Date]))
MoM_% = DIVIDE([Avg_Daily_Volume] - [Avg_Daily_Volume_PM], [Avg_Daily_Volume_PM])

## Forecast scoring (use with one model selected)
Actual = SUM(scorecard_placeholder[actual])
Point = SUM(scorecard_placeholder[point])
Forecast_Error_% = DIVIDE([Actual] - [Point], [Point])
Positive means the forecast was too low.

## Pages
1 Overview: line charts of Avg_Daily_Volume and Avg_Daily_Value by Date[Year-Month]; cards for latest Avg_Daily_Volume, Ticket_Size_INR, YoY_%, MoM_%; year slicer.
2 Festival Effect: table and column chart from festival_uplift; slicers for series and window.
3 Forecast: table from forecast.csv filtered to one target; backtest_results table; text box defining the range.
4 October 2026 scorecard: pending (scorecard_placeholder).
