import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# -----------------------------
# Settings
# -----------------------------
CSV_PATH = "google_pue_long_clean.csv"   # change this if your file has another name
OUTDIR = Path("figures")
OUTDIR.mkdir(exist_ok=True)

EXCLUDE_FLEET_FROM_AVERAGE = True

# "minmax" shows full spread across locations.
# "quantile" shows 10th–90th percentile, which looks cleaner.
BAND_MODE = "minmax"


# -----------------------------
# Load and clean data
# -----------------------------
df = pd.read_csv(CSV_PATH)

required_cols = {"period", "year", "quarter", "location", "quarterly_pue", "ttm_pue"}
missing = required_cols - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {missing}")

# Convert Q1/Q2/Q3/Q4 into quarter number
df["quarter_num"] = df["quarter"].astype(str).str.extract(r"Q([1-4])").astype(int)

# Make a true quarterly datetime for plotting
period_strings = df["year"].astype(str) + "Q" + df["quarter_num"].astype(str)

df["date"] = (
    pd.PeriodIndex(period_strings, freq="Q")
    .to_timestamp(how="end")
)

# Make sure PUE columns are numeric
df["quarterly_pue"] = pd.to_numeric(df["quarterly_pue"], errors="coerce")
df["ttm_pue"] = pd.to_numeric(df["ttm_pue"], errors="coerce")

# Exclude Fleet when computing averages/spread across individual data centers
plot_df = df.copy()
if EXCLUDE_FLEET_FROM_AVERAGE:
    plot_df = plot_df[plot_df["location"].str.lower() != "fleet"].copy()


# -----------------------------
# Aggregate by quarter
# -----------------------------
summary = (
    plot_df
    .groupby("date")
    .agg(
        avg_quarterly_pue=("quarterly_pue", "mean"),
        avg_ttm_pue=("ttm_pue", "mean"),

        min_quarterly_pue=("quarterly_pue", "min"),
        max_quarterly_pue=("quarterly_pue", "max"),
        min_ttm_pue=("ttm_pue", "min"),
        max_ttm_pue=("ttm_pue", "max"),

        q10_quarterly_pue=("quarterly_pue", lambda x: x.quantile(0.10)),
        q90_quarterly_pue=("quarterly_pue", lambda x: x.quantile(0.90)),
        q10_ttm_pue=("ttm_pue", lambda x: x.quantile(0.10)),
        q90_ttm_pue=("ttm_pue", lambda x: x.quantile(0.90)),

        n_locations=("location", "nunique"),
    )
    .reset_index()
    .sort_values("date")
)

# Choose shaded band type
if BAND_MODE == "quantile":
    q_low = "q10_quarterly_pue"
    q_high = "q90_quarterly_pue"
    ttm_low = "q10_ttm_pue"
    ttm_high = "q90_ttm_pue"
    band_label = "10th–90th percentile range"
else:
    q_low = "min_quarterly_pue"
    q_high = "max_quarterly_pue"
    ttm_low = "min_ttm_pue"
    ttm_high = "max_ttm_pue"
    band_label = "min–max range"


# -----------------------------
# Style helper
# -----------------------------
def polish_axis(ax):
    ax.grid(True, alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_ylabel("PUE", fontsize=12)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=35)
    ax.set_ylim(
        max(1.00, summary[[q_low, ttm_low]].min().min() - 0.03),
        summary[[q_high, ttm_high]].max().max() + 0.04,
    )


# -----------------------------
# Visual 1:
# Average quarterly PUE over time + TTM dashed
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6), dpi=180)

ax.plot(
    summary["date"],
    summary["avg_quarterly_pue"],
    linewidth=2.7,
    marker="o",
    markersize=3.8,
    label="Average quarterly PUE",
)

ax.plot(
    summary["date"],
    summary["avg_ttm_pue"],
    linewidth=2.7,
    linestyle="--",
    label="Average TTM PUE",
)

ax.set_title(
    "Google Data Center PUE Over Time",
    fontsize=17,
    weight="bold",
    pad=14,
)

subtitle = "Average across reported locations"
if EXCLUDE_FLEET_FROM_AVERAGE:
    subtitle += " excluding Fleet row"

ax.text(
    0.01,
    0.98,
    subtitle,
    transform=ax.transAxes,
    fontsize=10.5,
    va="top",
    alpha=0.75,
)

polish_axis(ax)
ax.legend(frameon=False, fontsize=10.5, loc="best")

fig.tight_layout()
fig.savefig(OUTDIR / "google_pue_average_with_ttm.png", bbox_inches="tight")
fig.savefig(OUTDIR / "google_pue_average_with_ttm.pdf", bbox_inches="tight")
plt.close(fig)


# -----------------------------
# Visual 2:
# Average lines + shaded spread across locations
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 6.4), dpi=180)

# Quarterly PUE spread
ax.fill_between(
    summary["date"],
    summary[q_low],
    summary[q_high],
    alpha=0.22,
    label=f"Quarterly PUE {band_label}",
)

# TTM PUE spread
ax.fill_between(
    summary["date"],
    summary[ttm_low],
    summary[ttm_high],
    alpha=0.16,
    label=f"TTM PUE {band_label}",
)

# Average quarterly line
ax.plot(
    summary["date"],
    summary["avg_quarterly_pue"],
    linewidth=2.8,
    marker="o",
    markersize=3.6,
    label="Average quarterly PUE",
)

# Average TTM line
ax.plot(
    summary["date"],
    summary["avg_ttm_pue"],
    linewidth=2.8,
    linestyle="--",
    label="Average TTM PUE",
)

ax.set_title(
    "Google Data Center PUE: Average and Location Spread",
    fontsize=17,
    weight="bold",
    pad=14,
)

ax.text(
    0.01,
    0.98,
    f"Shaded regions show the {band_label} across reported locations",
    transform=ax.transAxes,
    fontsize=10.5,
    va="top",
    alpha=0.75,
)

polish_axis(ax)
ax.legend(frameon=False, fontsize=10.3, loc="best", ncol=2)

fig.tight_layout()
fig.savefig(OUTDIR / "google_pue_average_with_spread.png", bbox_inches="tight")
fig.savefig(OUTDIR / "google_pue_average_with_spread.pdf", bbox_inches="tight")
plt.close(fig)


# -----------------------------
# Save summary table too
# -----------------------------
summary.to_csv(OUTDIR / "google_pue_summary_by_quarter.csv", index=False)

print("Saved visuals to:")
print(f"  {OUTDIR / 'google_pue_average_with_ttm.png'}")
print(f"  {OUTDIR / 'google_pue_average_with_ttm.pdf'}")
print(f"  {OUTDIR / 'google_pue_average_with_spread.png'}")
print(f"  {OUTDIR / 'google_pue_average_with_spread.pdf'}")
print()
print("Saved summary data to:")
print(f"  {OUTDIR / 'google_pue_summary_by_quarter.csv'}")