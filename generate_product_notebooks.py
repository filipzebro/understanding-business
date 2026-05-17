from __future__ import annotations

import re
from pathlib import Path

import nbformat as nbf
import numpy as np
import pandas as pd

from problem1_pipeline import (
    DATA_PATH,
    OUTPUT_DIR,
    build_emd_table,
    prepare_booking_features,
)


NOTEBOOK_DIR = OUTPUT_DIR / "notebooks"
ANALYSIS_DIR = OUTPUT_DIR / "product_driver_analysis"
MIN_OFFERS = 250

PRODUCTS = [
    "BAGGAGE",
    "PET",
    "SPECIAL EQUIPMENT",
    "FAST TRACK",
    "MEAL",
    "BUSINESS LOUNGE",
]

DRIVER_COLUMNS = [
    "MARKET",
    "FLIGHT_TYPE",
    "fare_bucket",
    "dtd_bucket",
    "stay_bucket",
    "route_direction",
    "origin_area",
    "destination_area",
    "is_hub_caj_route",
    "first_booking_class",
    "first_rbd",
    "BROWSER_TYPE",
    "FIRST_TOUCH_CHANNEL",
    "LAST_TOUCH_CHANNEL",
    "LOGIN_STATUS",
    "has_loyalty",
    "has_child_or_infant",
    "pax_bucket",
    "emd_price_bucket",
    "display_position_bucket",
]


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def add_buckets(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared["pax_bucket"] = pd.cut(
        prepared["pax_total"],
        bins=[-np.inf, 1, 2, 3, np.inf],
        labels=["1", "2", "3", "4_plus"],
    ).astype("object")
    prepared["emd_price_bucket"] = pd.qcut(
        prepared["emd_price"].rank(method="first"),
        q=5,
        labels=["very_low", "low", "mid", "high", "very_high"],
    ).astype("object")
    prepared["display_position_bucket"] = prepared["display_position"].fillna(99).astype(int).astype(str)
    for column in DRIVER_COLUMNS:
        if column in prepared.columns:
            prepared[column] = prepared[column].astype("object").where(prepared[column].notna(), "MISSING")
    return prepared


def markdown_table(df: pd.DataFrame, max_rows: int = 10) -> str:
    if df.empty:
        return "_No segments met the minimum volume threshold._"
    return df.head(max_rows).to_markdown(index=False, floatfmt=".4f")


def product_summary_line(product_df: pd.DataFrame) -> dict:
    purchases = int(product_df["sold"].sum())
    offers = int(len(product_df))
    revenue = float(product_df["revenue"].sum())
    return {
        "offers": offers,
        "purchases": purchases,
        "conversion_rate": purchases / offers if offers else 0,
        "revenue": revenue,
        "avg_sold_price": revenue / purchases if purchases else 0,
        "avg_offered_price": float(product_df["emd_price"].mean()),
    }


def analyze_driver(product_df: pd.DataFrame, column: str, baseline_cr: float) -> pd.DataFrame:
    grouped = (
        product_df.groupby(column, dropna=False)
        .agg(
            offers=("sold", "size"),
            purchases=("sold", "sum"),
            conversion_rate=("sold", "mean"),
            revenue=("revenue", "sum"),
            avg_price=("emd_price", "mean"),
        )
        .reset_index()
    )
    grouped = grouped[grouped["offers"] >= MIN_OFFERS].copy()
    grouped["cr_lift_vs_product"] = grouped["conversion_rate"] / baseline_cr if baseline_cr > 0 else np.nan
    grouped["revenue_share"] = grouped["revenue"] / grouped["revenue"].sum() if grouped["revenue"].sum() else 0
    grouped.insert(0, "driver", column)
    grouped = grouped.rename(columns={column: "segment_value"})
    return grouped.sort_values(["cr_lift_vs_product", "revenue"], ascending=False)


def compact_driver_score(driver_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for driver, group in driver_df.groupby("driver"):
        if group.empty:
            continue
        pitch_group = group[group["segment_value"].astype(str) != "MISSING"]
        if pitch_group.empty:
            pitch_group = group
        top = pitch_group.sort_values("cr_lift_vs_product", ascending=False).iloc[0]
        revenue_top = pitch_group.sort_values("revenue", ascending=False).iloc[0]
        rows.append(
            {
                "driver": driver,
                "best_cr_segment": top["segment_value"],
                "best_cr": top["conversion_rate"],
                "best_cr_lift": top["cr_lift_vs_product"],
                "highest_revenue_segment": revenue_top["segment_value"],
                "highest_segment_revenue": revenue_top["revenue"],
            }
        )
    return pd.DataFrame(rows).sort_values(["best_cr_lift", "highest_segment_revenue"], ascending=False)


def make_notebook(product: str, product_df: pd.DataFrame, driver_df: pd.DataFrame, driver_score: pd.DataFrame) -> nbf.NotebookNode:
    slug = slugify(product)
    summary = product_summary_line(product_df)
    pitch_driver_df = driver_df[driver_df["segment_value"].astype(str) != "MISSING"].copy()
    top_cr = pitch_driver_df.sort_values(["cr_lift_vs_product", "offers"], ascending=False).head(12)
    top_revenue = driver_df.sort_values("revenue", ascending=False).head(12)

    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }

    nb.cells = [
        nbf.v4.new_markdown_cell(
            f"""# {product}: purchase driver analysis

Business question: which booking, route, passenger and sales-context factors are most associated with buying **{product}**?

Core KPI:

| Metric | Value |
|---|---:|
| Offers | {summary['offers']:,} |
| Purchases | {summary['purchases']:,} |
| Conversion rate | {summary['conversion_rate']:.2%} |
| Revenue | {summary['revenue']:,.0f} |
| Avg sold price | {summary['avg_sold_price']:.2f} |

Interpretation rule: this is directional driver analysis, not causal proof. The production recommendation should be validated with an A/B test."""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Strongest segments by conversion lift

These are the segments where conversion is highest versus the product baseline, after filtering out low-volume groups.

{markdown_table(top_cr[['driver', 'segment_value', 'offers', 'conversion_rate', 'cr_lift_vs_product', 'revenue']], 12)}"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Biggest revenue segments

These are often more useful for the business pitch than pure CR, because they combine volume, price and conversion.

{markdown_table(top_revenue[['driver', 'segment_value', 'offers', 'purchases', 'conversion_rate', 'revenue']], 12)}"""
        ),
        nbf.v4.new_markdown_cell(
            f"""## Driver summary

Each row shows the best segment by conversion lift and the largest segment by revenue for a candidate driver.

{markdown_table(driver_score, 20)}"""
        ),
        nbf.v4.new_code_cell(
            f"""from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
base = Path("outputs/problem1/product_driver_analysis")
if not base.exists():
    base = Path("../product_driver_analysis")
driver_df = pd.read_csv(base / "{slug}_driver_segments.csv")
driver_score = pd.read_csv(base / "{slug}_driver_score.csv")

driver_df.head()"""
        ),
        nbf.v4.new_code_cell(
            """plot_df = driver_score.sort_values("best_cr_lift", ascending=False).head(10)
plt.figure(figsize=(10, 5))
sns.barplot(data=plot_df, y="driver", x="best_cr_lift", color="#2F6F9F")
plt.axvline(1, color="black", linewidth=1)
plt.title("Top drivers by best-segment CR lift")
plt.xlabel("Lift vs product baseline")
plt.ylabel("")
plt.tight_layout()"""
        ),
        nbf.v4.new_code_cell(
            """plot_df = driver_df.sort_values("revenue", ascending=False).head(15)
plt.figure(figsize=(10, 6))
sns.barplot(data=plot_df, y="segment_value", x="revenue", hue="driver", dodge=False)
plt.title("Highest-revenue segments")
plt.xlabel("Revenue")
plt.ylabel("")
plt.legend(loc="lower right")
plt.tight_layout()"""
        ),
        nbf.v4.new_code_cell(
            """plot_df = driver_df[driver_df["driver"].isin(["fare_bucket", "dtd_bucket", "stay_bucket", "FLIGHT_TYPE", "display_position_bucket"])].copy()
plot_df = plot_df.sort_values(["driver", "segment_value"])
g = sns.catplot(
    data=plot_df,
    kind="bar",
    x="segment_value",
    y="conversion_rate",
    col="driver",
    col_wrap=2,
    sharex=False,
    sharey=False,
    color="#2F6F9F",
    height=3.2,
    aspect=1.5,
)
g.set_xticklabels(rotation=35, ha="right")
g.set_axis_labels("", "Conversion rate")
g.fig.suptitle("Conversion rate by selected business buckets", y=1.03)
plt.tight_layout()"""
        ),
        nbf.v4.new_code_cell(
            """# Use this table to pick slide examples: high lift, enough volume, and meaningful revenue.
driver_df.query("offers >= 1000").sort_values(
    ["cr_lift_vs_product", "revenue"],
    ascending=False,
).head(25)"""
        ),
    ]
    return nb


def main() -> None:
    NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...")
    df = pd.read_parquet(DATA_PATH)
    features = prepare_booking_features(df)
    emd = build_emd_table(features)
    merged = add_buckets(emd.merge(features, on="booking_id", how="left"))

    index_rows = []
    for product in PRODUCTS:
        slug = slugify(product)
        print(f"Generating notebook for {product}...")
        product_df = merged[merged["product"] == product].copy()
        summary = product_summary_line(product_df)

        driver_parts = [
            analyze_driver(product_df, column, summary["conversion_rate"])
            for column in DRIVER_COLUMNS
            if column in product_df.columns
        ]
        driver_df = pd.concat(driver_parts, ignore_index=True)
        driver_score = compact_driver_score(driver_df)

        driver_df.to_csv(ANALYSIS_DIR / f"{slug}_driver_segments.csv", index=False)
        driver_score.to_csv(ANALYSIS_DIR / f"{slug}_driver_score.csv", index=False)

        nb = make_notebook(product, product_df, driver_df, driver_score)
        notebook_path = NOTEBOOK_DIR / f"{slug}_driver_analysis.ipynb"
        nbf.write(nb, notebook_path)
        index_rows.append({"product": product, "notebook": str(notebook_path), **summary})

    pd.DataFrame(index_rows).to_csv(NOTEBOOK_DIR / "notebook_index.csv", index=False)
    print(f"Saved notebooks to {NOTEBOOK_DIR.resolve()}")
    print(f"Saved analysis tables to {ANALYSIS_DIR.resolve()}")


if __name__ == "__main__":
    main()
