"""Clean the raw Eskom hourly demand file into demand_clean.csv (one row per hour)."""
import pandas as pd

FILE = "data/ESK20096.csv"          # raw file from the Eskom Data Portal (not in the repo)
OUT = "data/demand_clean.csv"

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

# 1. Load (commas first; if everything lands in one column, try semicolons)
df = pd.read_csv(FILE, encoding="utf-8-sig")
if df.shape[1] == 1:
    df = pd.read_csv(FILE, sep=";", encoding="utf-8-sig")
df.columns = df.columns.str.strip()          # Eskom names start with a space
print(df.head())
df.info()

# 2. Short column names
df = df.rename(columns={
    "Date Time Hour Beginning": "timestamp",
    "RSA Contracted Demand": "demand_mw",
    "RSA Contracted Forecast": "eskom_forecast_mw",
})
if "demand_mw" not in df.columns:
    raise SystemExit("No demand column found. Columns are: " + str(df.columns.tolist()))

# 3. Data types (12-hour clock with AM/PM, so 12:00 AM is midnight)
df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y-%m-%d %I:%M:%S %p", errors="coerce")
for col in df.columns.drop("timestamp"):
    df[col] = pd.to_numeric(df[col], errors="coerce")
print("Unreadable dates:", df["timestamp"].isna().sum())
df = df.dropna(subset=["timestamp"])

# 4. Duplicate hours
print("Duplicate hours:", df.duplicated(subset="timestamp").sum())
df = df.drop_duplicates(subset="timestamp")

# 5. Keep only hours with observed demand (the file also holds future forecasts)
df = df.set_index("timestamp").sort_index()
df = df.loc[:df["demand_mw"].last_valid_index()]

# 6. Missing hours
full = pd.date_range(df.index.min(), df.index.max(), freq="h")
print("Missing hours:", len(full.difference(df.index)))
df = df.reindex(full)
df.index.name = "timestamp"

# 7. Impossible values
bad = (df["demand_mw"] < 10000) | (df["demand_mw"] > 40000)
print("Impossible values:", bad.sum())
df.loc[bad, "demand_mw"] = None

# 8. Fill only short gaps (3 hours or less)
print("Empty demand values before filling:", df["demand_mw"].isna().sum())
df["demand_mw"] = df["demand_mw"].interpolate(limit=3, limit_area="inside")
print("Empty demand values after filling:", df["demand_mw"].isna().sum())

# 9. Save
df.to_csv(OUT)
print(f"Saved {OUT} with {len(df)} rows")
