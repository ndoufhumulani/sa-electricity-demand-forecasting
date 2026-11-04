"""Exploratory analysis: five charts and Eskom's forecast error."""
import matplotlib
matplotlib.use("Agg")
import os
import pandas as pd
import matplotlib.pyplot as plt
import holidays

# ---------- Load the cleaned data ----------
df = pd.read_csv("data/demand_clean.csv", parse_dates=["timestamp"], index_col="timestamp")

# Eskom's forecast column may have either name, depending on your rename step
if "RSA Contracted Forecast" in df.columns:
    df = df.rename(columns={"RSA Contracted Forecast": "eskom_forecast_mw"})

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 120, "axes.spines.top": False, "axes.spines.right": False})

print("Period:", df.index.min(), "to", df.index.max(), "|", len(df), "hours")
print(df["demand_mw"].describe().round(0))

# ---------- Chart 1: demand over the whole period ----------
daily = df["demand_mw"].resample("D").mean() / 1000
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(daily.index, daily, lw=0.8)
ax.set_ylabel("Daily mean demand (GW)")
ax.set_title("1. RSA contracted demand, daily mean")
fig.tight_layout(); fig.savefig("figures/01_demand_over_time.png"); plt.close(fig)

# ---------- Chart 2: average day, winter vs summer ----------
month = df.index.month
fig, ax = plt.subplots(figsize=(9, 3.5))
for name, months in [("Winter (Jun-Aug)", [6, 7, 8]), ("Summer (Dec-Feb)", [12, 1, 2])]:
    profile = df[month.isin(months)].groupby(df[month.isin(months)].index.hour)["demand_mw"].mean() / 1000
    ax.plot(profile.index, profile, marker="o", ms=3, label=name)
ax.set_xlabel("Hour of day"); ax.set_ylabel("Mean demand (GW)"); ax.set_xticks(range(0, 24, 2))
ax.legend(frameon=False); ax.set_title("2. Average daily profile by season")
fig.tight_layout(); fig.savefig("figures/02_daily_profile.png"); plt.close(fig)

# ---------- Chart 3: weekday vs weekend vs public holiday ----------
sa_holidays = holidays.country_holidays("ZA", years=range(df.index.year.min(), df.index.year.max() + 1))
is_holiday = pd.Series([d in sa_holidays for d in df.index.date], index=df.index)
day_type = pd.Series("Weekday", index=df.index)
day_type[df.index.day_of_week >= 5] = "Weekend"
day_type[is_holiday] = "Public holiday"
means = df.groupby(day_type)["demand_mw"].mean().reindex(["Weekday", "Weekend", "Public holiday"]) / 1000
print("\nMean demand by day type (GW):"); print(means.round(2))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.bar(means.index, means)
for i, v in enumerate(means):
    ax.text(i, v + 0.2, f"{v:.1f}", ha="center")
ax.set_ylim(0, means.max() * 1.12); ax.set_ylabel("Mean demand (GW)")
ax.set_title("3. Mean demand by day type")
fig.tight_layout(); fig.savefig("figures/03_day_type.png"); plt.close(fig)

# ---------- Chart 4: load shedding hours per month ----------
shed_cols = [c for c in df.columns if "Load_Reduction" in c or "ILS" in c]
if shed_cols:
    shedding = (df[shed_cols].fillna(0).sum(axis=1) > 0)
    per_month = shedding.resample("MS").sum()
    print("\nHours with load shedding or ILS:", int(shedding.sum()), f"({shedding.mean():.1%})")
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(per_month.index, per_month, width=25)
    ax.set_ylabel("Hours in month"); ax.set_title("4. Hours with load shedding or ILS, per month")
    fig.tight_layout(); fig.savefig("figures/04_load_shedding.png"); plt.close(fig)

# ---------- Benchmark: how accurate is Eskom's own forecast? ----------
if "eskom_forecast_mw" in df.columns:
    both = df[["demand_mw", "eskom_forecast_mw"]].dropna()
    err = both["demand_mw"] - both["eskom_forecast_mw"]
    mae = err.abs().mean()
    mape = (err.abs() / both["demand_mw"]).mean() * 100
    print(f"\nEskom forecast error: MAE = {mae:.0f} MW, MAPE = {mape:.2f}%  <- the benchmark to beat")

    week = both.loc["2025-07-14":"2025-07-20"] / 1000
    if len(week):
        fig, ax = plt.subplots(figsize=(11, 3.5))
        ax.plot(week.index, week["demand_mw"], label="Actual", lw=1.4)
        ax.plot(week.index, week["eskom_forecast_mw"], label="Eskom forecast", ls="--")
        ax.set_ylabel("Demand (GW)"); ax.legend(frameon=False)
        ax.set_title("5. Actual demand and Eskom's forecast, 14-20 July 2025")
        fig.tight_layout(); fig.savefig("figures/05_eskom_forecast_week.png"); plt.close(fig)

print("\nCharts saved in the 'figures' folder.")
