"""Features, models and evaluation against Eskom's published forecast."""
import matplotlib
matplotlib.use("Agg")                       # save charts without opening windows
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import holidays
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

os.makedirs("figures", exist_ok=True)
os.makedirs("results", exist_ok=True)

# =====================================================================
# 1. Load the cleaned data
# =====================================================================
df = pd.read_csv("data/demand_clean.csv", parse_dates=["timestamp"], index_col="timestamp")
if "RSA Contracted Forecast" in df.columns:
    df = df.rename(columns={"RSA Contracted Forecast": "eskom_forecast_mw"})
y_name = "demand_mw"

# =====================================================================
# 2. Features (only information known at least 24 hours in advance)
# =====================================================================
t = df.index
df["hour"] = t.hour
df["dayofweek"] = t.day_of_week                      # Monday = 0
df["month"] = t.month
df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

sa_holidays = holidays.country_holidays("ZA", years=range(t.year.min(), t.year.max() + 2))
dates = pd.Series(t.date, index=t)
df["is_holiday"] = dates.map(lambda d: d in sa_holidays).astype(int)
df["day_after_holiday"] = dates.map(lambda d: (d - pd.Timedelta(days=1)) in sa_holidays).astype(int)
festive = ((t.month == 12) & (t.day >= 20)) | ((t.month == 1) & (t.day <= 6))
df["festive_season"] = festive.astype(int)

# Load shedding flag (Eskom announces stages in advance)
shed_cols = [c for c in df.columns if "Load_Reduction" in c]
df["load_shedding"] = (df[shed_cols].fillna(0).sum(axis=1) > 0).astype(int) if shed_cols else 0

# Fourier terms: daily (24 h), weekly (168 h) and yearly cycles
hours_since_start = (t - t[0]) / pd.Timedelta(hours=1)
hour_of_week = df["dayofweek"] * 24 + df["hour"]
fourier = []
for k in (1, 2, 3, 4):
    for name, x, P in [("day", df["hour"], 24)]:
        df[f"sin_{name}_{k}"] = np.sin(2 * np.pi * k * x / P)
        df[f"cos_{name}_{k}"] = np.cos(2 * np.pi * k * x / P)
        fourier += [f"sin_{name}_{k}", f"cos_{name}_{k}"]
for k in (1, 2, 3):
    df[f"sin_week_{k}"] = np.sin(2 * np.pi * k * hour_of_week / 168)
    df[f"cos_week_{k}"] = np.cos(2 * np.pi * k * hour_of_week / 168)
    fourier += [f"sin_week_{k}", f"cos_week_{k}"]
for k in (1, 2):
    df[f"sin_year_{k}"] = np.sin(2 * np.pi * k * hours_since_start / 8766)
    df[f"cos_year_{k}"] = np.cos(2 * np.pi * k * hours_since_start / 8766)
    fourier += [f"sin_year_{k}", f"cos_year_{k}"]

# Season x hour interactions: lets the linear model change the daily shape by season
interactions = []
for k in (1, 2, 3, 4):
    for trig in ("sin", "cos"):
        for s in ("sin_year_1", "cos_year_1"):
            name = f"{trig}_day_{k}_x_{s}"
            df[name] = df[f"{trig}_day_{k}"] * df[s]
            interactions.append(name)

# Lagged demand: what the model knows about the current level
df["lag_24"] = df[y_name].shift(24)                              # same hour yesterday
df["lag_168"] = df[y_name].shift(168)                            # same hour last week
df["mean_last_7_days"] = df[y_name].shift(24).rolling(168).mean()  # recent average level

calendar = ["is_weekend", "is_holiday", "day_after_holiday", "festive_season", "load_shedding"]
lags = ["lag_24", "lag_168", "mean_last_7_days"]
linear_features = fourier + interactions + calendar + ["mean_last_7_days"]
tree_features = ["hour", "dayofweek", "month"] + fourier + calendar + lags

df = df.dropna(subset=[y_name] + lags)

# =====================================================================
# 3. Split by time: the last six months are the test period
# =====================================================================
TEST_START = "2026-05-01"
test = df[df.index >= TEST_START]
train_all = df[df.index < TEST_START]                       # Option A: all data
train_recent = train_all[train_all.index >= "2024-04-01"]   # Option B: after heavy load shedding
print(f"Test: {test.index.min():%d %b %Y} to {test.index.max():%d %b %Y} ({len(test)} hours)")
print(f"Option A training: {len(train_all)} hours | Option B training: {len(train_recent)} hours\n")

# =====================================================================
# 4. Models
# =====================================================================
def fit_models(train):
    lin = LinearRegression().fit(train[linear_features], train[y_name])
    rf = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=-1)
    rf.fit(train[tree_features], train[y_name])
    xgb = XGBRegressor(n_estimators=800, learning_rate=0.03, max_depth=6,
                       subsample=0.8, colsample_bytree=0.8, random_state=42)
    xgb.fit(train[tree_features], train[y_name])
    return lin, rf, xgb

preds = {
    "Naive (same hour yesterday)": test["lag_24"].to_numpy(),
    "Seasonal naive (same hour last week)": test["lag_168"].to_numpy(),
}
models = {}
for option, train in [("A: all data", train_all), ("B: from Apr 2024", train_recent)]:
    lin, rf, xgb = fit_models(train)
    models[option] = xgb
    preds[f"Fourier regression ({option})"] = lin.predict(test[linear_features])
    preds[f"Random Forest ({option})"] = rf.predict(test[tree_features])
    preds[f"XGBoost ({option})"] = xgb.predict(test[tree_features])
    print(f"Trained models for option {option}")

if "eskom_forecast_mw" in test.columns:
    preds["Eskom's own forecast"] = test["eskom_forecast_mw"].to_numpy()

# Check: least squares by hand gives the same Fourier regression coefficients
A = np.column_stack([np.ones(len(train_recent)), train_recent[linear_features]])
coef, *_ = np.linalg.lstsq(A, train_recent[y_name], rcond=None)
lin_b = LinearRegression().fit(train_recent[linear_features], train_recent[y_name])
print(f"\nLeast squares by hand matches scikit-learn: {np.allclose(coef[1:], lin_b.coef_, rtol=1e-4, atol=1e-4)}")

# =====================================================================
# 5. Evaluate
# =====================================================================
y = test[y_name].to_numpy()
peak = test.index.hour.isin([17, 18, 19, 20])
rows = []
for name, p in preds.items():
    rows.append({
        "Model": name,
        "MAE (MW)": mean_absolute_error(y, p),
        "RMSE (MW)": mean_squared_error(y, p) ** 0.5,
        "MAPE (%)": 100 * np.mean(np.abs((y - p) / y)),
        "Peak MAE (MW)": mean_absolute_error(y[peak], p[peak]),
    })
results = pd.DataFrame(rows).sort_values("MAE (MW)").reset_index(drop=True)
results.to_csv("results/results.csv", index=False)
pd.set_option("display.width", 200)
print("\nTest-set results (sorted by MAE):")
print(results.round(2).to_string(index=False))

# =====================================================================
# 6. Charts
# =====================================================================
best = results[~results["Model"].isin(["Eskom's own forecast"])].iloc[0]["Model"]
week = test.loc["2026-07-13":"2026-07-19 23:00"]
pos = test.index.get_indexer(week.index)
if len(week):
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(week.index, week[y_name] / 1000, color="black", lw=1.4, label="Actual")
    ax.plot(week.index, preds[best][pos] / 1000, ls="--", label=best)
    if "Eskom's own forecast" in preds:
        ax.plot(week.index, preds["Eskom's own forecast"][pos] / 1000, alpha=0.7, label="Eskom forecast")
    ax.set_ylabel("Demand (GW)"); ax.legend(frameon=False, fontsize=8)
    ax.set_title("6. Actual and forecast demand, 13-19 July 2026 (test period)")
    fig.tight_layout(); fig.savefig("figures/06_forecast_week.png")

option = "B: from Apr 2024" if "Apr 2024" in best else "A: all data"
imp = pd.Series(models[option].feature_importances_, index=tree_features).sort_values().tail(10)
fig, ax = plt.subplots(figsize=(7, 4))
ax.barh(imp.index, imp * 100)
ax.set_xlabel("Importance (%)"); ax.set_title(f"7. XGBoost feature importance (option {option[0]})")
fig.tight_layout(); fig.savefig("figures/07_feature_importance.png")

print(f"\nBest model: {best}")
print("Saved results/results.csv, figures/06_forecast_week.png and figures/07_feature_importance.png")
