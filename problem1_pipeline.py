from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


DATA_PATH = Path("data_ub.parquet")
OUTPUT_DIR = Path("outputs/problem1")
RANDOM_STATE = 42
TOP_N_PRODUCTS = 3
MAX_TRAIN_ROWS_PER_PRODUCT = 60_000


def safe_json_loads(value):
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def airport_area(code: str) -> str:
    """Synthetic airport code grouping. Keeps the rule simple and auditable."""
    if not isinstance(code, str) or len(code) == 0:
        return "UNKNOWN"
    first = code[0].upper()
    return {
        "A": "AREA_A",
        "B": "AREA_B",
        "C": "AREA_C",
        "D": "AREA_D",
    }.get(first, "AREA_OTHER")


def bucketize_days_to_departure(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 0, 3, 7, 14, 30, 60, 120, np.inf]
    labels = [
        "invalid_or_past",
        "0_3",
        "4_7",
        "8_14",
        "15_30",
        "31_60",
        "61_120",
        "120_plus",
    ]
    return pd.cut(series, bins=bins, labels=labels).astype("object")


def bucketize_stay(series: pd.Series) -> pd.Series:
    bins = [-np.inf, 0, 2, 5, 8, 14, 30, np.inf]
    labels = [
        "unknown_or_oneway",
        "1_2",
        "3_5",
        "6_8",
        "9_14",
        "15_30",
        "30_plus",
    ]
    return pd.cut(series.fillna(-1), bins=bins, labels=labels).astype("object")


def bucketize_fare(series: pd.Series) -> pd.Series:
    labels = ["very_low", "low", "mid", "high", "very_high"]
    ranked = series.rank(method="first")
    return pd.qcut(ranked, q=5, labels=labels).astype("object")


def parse_segment_features(df: pd.DataFrame) -> pd.DataFrame:
    segment_rows = df["segments"].map(safe_json_loads)
    first_booking_class = []
    first_rbd = []
    segment_count = []
    first_departure_dates = []
    final_arrival_dates = []
    last_departure_dates = []
    connection_flags = []

    for segments in segment_rows:
        segment_count.append(len(segments))
        connection_flags.append(int(len(segments) > 1))
        if len(segments) == 0:
            first_booking_class.append("UNKNOWN")
            first_rbd.append("UNKNOWN")
            first_departure_dates.append(None)
            final_arrival_dates.append(None)
            last_departure_dates.append(None)
            continue

        first = segments[0]
        last = segments[-1]
        first_booking_class.append(first.get("BOOKING_CLASS", "UNKNOWN"))
        first_rbd.append(first.get("RBD", "UNKNOWN"))
        first_departure_dates.append(first.get("DEPARTURE_DATE"))
        final_arrival_dates.append(last.get("ARRIVAL_DATE"))
        last_departure_dates.append(last.get("DEPARTURE_DATE"))

    parsed = pd.DataFrame(
        {
            "first_booking_class": first_booking_class,
            "first_rbd": first_rbd,
            "segment_count": segment_count,
            "has_connection": connection_flags,
            "first_departure_dt": pd.to_datetime(first_departure_dates, errors="coerce"),
            "final_arrival_dt": final_arrival_dates,
            "last_departure_dt": pd.to_datetime(last_departure_dates, errors="coerce"),
        },
        index=df.index,
    )
    parsed["final_arrival_dt"] = pd.to_datetime(parsed["final_arrival_dt"], errors="coerce")
    parsed["travel_span_hours"] = (
        parsed["final_arrival_dt"] - parsed["first_departure_dt"]
    ).dt.total_seconds() / 3600
    parsed["segment_based_stay_days"] = (
        parsed["last_departure_dt"] - parsed["first_departure_dt"]
    ).dt.total_seconds() / (3600 * 24)
    parsed.loc[parsed["segment_based_stay_days"] <= 0, "segment_based_stay_days"] = np.nan
    return parsed.drop(columns=["first_departure_dt", "final_arrival_dt", "last_departure_dt"])


def prepare_booking_features(df: pd.DataFrame) -> pd.DataFrame:
    features = df.copy()
    ond_split = features["OND"].str.split("-", n=1, expand=True)
    features["origin"] = ond_split[0]
    features["destination"] = ond_split[1]
    features["origin_area"] = features["origin"].map(airport_area)
    features["destination_area"] = features["destination"].map(airport_area)
    features["route_direction"] = features["origin_area"] + "_TO_" + features["destination_area"]
    features["origin_prefix"] = features["origin"].str[:2].fillna("UNKNOWN")
    features["destination_prefix"] = features["destination"].str[:2].fillna("UNKNOWN")
    features["is_hub_caj_route"] = (
        (features["origin"] == "CAJ") | (features["destination"] == "CAJ")
    ).astype(int)

    features["dtd_bucket"] = bucketize_days_to_departure(features["DTD"])
    features["fare_bucket"] = bucketize_fare(features["FARE_YQ_TICKET"])
    features["pax_total"] = (
        features["ADULTS"] + features["TEENAGERS"] + features["CHILDREN"] + features["INFANTS"]
    )
    features["has_child_or_infant"] = ((features["CHILDREN"] + features["INFANTS"]) > 0).astype(int)
    features["has_teenager"] = (features["TEENAGERS"] > 0).astype(int)
    features["is_logged_in"] = features["LOGIN_STATUS"].eq("Logged-In").astype(int)
    features["has_loyalty"] = features["LOYALTY_MEMBERSHIP_PROGRAM"].notna().astype(int)

    segment_features = parse_segment_features(features)
    features = pd.concat([features, segment_features], axis=1)
    features["stay_length_days"] = np.where(
        features["FLIGHT_TYPE"].eq("Round-Trip"),
        features["segment_based_stay_days"],
        np.nan,
    )
    features["stay_bucket"] = bucketize_stay(features["stay_length_days"])
    features["booking_id"] = np.arange(len(features))
    return features


def build_emd_table(features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for booking_id, emd_info, emds_order in zip(
        features["booking_id"], features["EMD_INFO"], features["emds_order"]
    ):
        items = safe_json_loads(emd_info)
        order = safe_json_loads(emds_order)
        order_position = {name: position + 1 for position, name in enumerate(order)}
        for item in items:
            product = item.get("emd_name")
            price = pd.to_numeric(item.get("emd_price"), errors="coerce")
            sold = str(item.get("sold")).lower() == "true"
            rows.append(
                {
                    "booking_id": booking_id,
                    "product": product,
                    "emd_price": price,
                    "sold": int(sold),
                    "display_position": order_position.get(product, np.nan),
                    "revenue": price if sold else 0.0,
                }
            )
    return pd.DataFrame(rows)


def save_product_summary(emd: pd.DataFrame) -> pd.DataFrame:
    summary = (
        emd.groupby("product")
        .agg(
            offers=("sold", "size"),
            purchases=("sold", "sum"),
            conversion_rate=("sold", "mean"),
            avg_offered_price=("emd_price", "mean"),
            avg_sold_price=("revenue", lambda s: s.sum() / max((s > 0).sum(), 1)),
            revenue=("revenue", "sum"),
        )
        .sort_values("revenue", ascending=False)
    )
    summary["revenue_share"] = summary["revenue"] / summary["revenue"].sum()
    summary.to_csv(OUTPUT_DIR / "product_summary.csv")
    return summary


def save_eda_charts(features: pd.DataFrame, emd: pd.DataFrame, product_summary: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    plot_df = product_summary.reset_index()
    sns.barplot(data=plot_df, x="product", y="revenue", ax=ax1, color="#2F6F9F")
    ax1.set_title("Ancillary revenue by product")
    ax1.set_xlabel("")
    ax1.set_ylabel("Revenue")
    ax1.tick_params(axis="x", rotation=25)
    ax2 = ax1.twinx()
    sns.lineplot(data=plot_df, x="product", y="conversion_rate", ax=ax2, color="#C44900", marker="o")
    ax2.set_ylabel("Conversion rate")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "revenue_vs_conversion.png", dpi=180)
    plt.close(fig)

    top_products = product_summary.head(TOP_N_PRODUCTS).index.tolist()
    merged = emd.merge(features, on="booking_id", how="left")
    rows = []
    for product in top_products:
        product_rows = merged[merged["product"] == product]
        for feature in ["MARKET", "FLIGHT_TYPE", "fare_bucket", "dtd_bucket", "route_direction"]:
            cr = (
                product_rows.groupby(feature, dropna=False)
                .agg(offers=("sold", "size"), cr=("sold", "mean"), revenue=("revenue", "sum"))
                .query("offers >= 200")
                .sort_values("cr", ascending=False)
                .head(6)
                .reset_index()
            )
            cr.insert(0, "feature", feature)
            cr.insert(0, "product", product)
            rows.append(cr)
    pd.concat(rows, ignore_index=True).to_csv(OUTPUT_DIR / "top_segments_by_cr.csv", index=False)

    target_wide = (
        emd.pivot_table(index="booking_id", columns="product", values="sold", aggfunc="max")
        .add_prefix("sold_")
        .fillna(0)
    )
    numeric_features = features[
        [
            "booking_id",
            "DTD",
            "FARE_YQ_TICKET",
            "stay_length_days",
            "pax_total",
            "has_child_or_infant",
            "has_teenager",
            "is_logged_in",
            "has_loyalty",
            "segment_count",
            "has_connection",
            "travel_span_hours",
            "HOUR_OF_THE_DAY_PL_TIME",
        ]
    ].merge(target_wide, on="booking_id")
    corr = numeric_features.drop(columns=["booking_id"]).corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, cmap="vlag", center=0, linewidths=0.3, ax=ax)
    ax.set_title("Correlation heatmap: booking context vs EMD purchase")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=180)
    plt.close(fig)


def fit_models(features: pd.DataFrame, emd: pd.DataFrame, product_summary: pd.DataFrame) -> pd.DataFrame:
    merged = emd.merge(features, on="booking_id", how="left")
    top_products = product_summary.head(TOP_N_PRODUCTS).index.tolist()

    numeric_cols = [
        "DTD",
        "FARE_YQ_TICKET",
        "stay_length_days",
        "pax_total",
        "ADULTS",
        "TEENAGERS",
        "CHILDREN",
        "INFANTS",
        "has_child_or_infant",
        "has_teenager",
        "is_logged_in",
        "has_loyalty",
        "segment_count",
        "has_connection",
        "travel_span_hours",
        "HOUR_OF_THE_DAY_PL_TIME",
        "emd_price",
        "display_position",
        "is_hub_caj_route",
    ]
    categorical_cols = [
        "product",
        "MARKET",
        "LANGUAGE",
        "FLIGHT_TYPE",
        "BROWSER_TYPE",
        "FIRST_TOUCH_CHANNEL",
        "LAST_TOUCH_CHANNEL",
        "LOGIN_STATUS",
        "origin_area",
        "destination_area",
        "route_direction",
        "origin_prefix",
        "destination_prefix",
        "dtd_bucket",
        "fare_bucket",
        "stay_bucket",
        "first_booking_class",
        "first_rbd",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=500)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )

    model_rows = []
    for product in top_products:
        product_df = merged[merged["product"] == product].copy()
        positives = int(product_df["sold"].sum())
        if positives < 30:
            continue

        X = product_df[numeric_cols + categorical_cols]
        y = product_df["sold"]
        X_train_full, X_test, y_train_full, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=RANDOM_STATE,
            stratify=y,
        )

        train_df = X_train_full.copy()
        train_df["sold"] = y_train_full.to_numpy()
        if len(train_df) > MAX_TRAIN_ROWS_PER_PRODUCT:
            pos = train_df[train_df["sold"] == 1]
            neg = train_df[train_df["sold"] == 0]
            target_neg_n = max(MAX_TRAIN_ROWS_PER_PRODUCT - len(pos), min(len(neg), len(pos) * 4))
            sampled_neg = neg.sample(n=min(target_neg_n, len(neg)), random_state=RANDOM_STATE)
            train_df = pd.concat([pos, sampled_neg]).sample(frac=1, random_state=RANDOM_STATE)

        X_train = train_df[numeric_cols + categorical_cols]
        y_train = train_df["sold"]

        logit = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=250,
                        class_weight="balanced",
                        solver="saga",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )
        logit.fit(X_train, y_train)
        logit_pred = logit.predict_proba(X_test)[:, 1]

        scale_pos_weight = max((y_train == 0).sum() / max((y_train == 1).sum(), 1), 1)
        xgb = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.06,
                        subsample=0.85,
                        colsample_bytree=0.85,
                        eval_metric="logloss",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        scale_pos_weight=scale_pos_weight,
                    ),
                ),
            ]
        )
        xgb.fit(X_train, y_train)
        xgb_pred = xgb.predict_proba(X_test)[:, 1]

        for model_name, pred in [("logistic_regression", logit_pred), ("xgboost", xgb_pred)]:
            model_rows.append(
                {
                    "product": product,
                    "model": model_name,
                    "train_rows_used": len(train_df),
                    "test_rows": len(X_test),
                    "train_positive_rate": y_train.mean(),
                    "test_positive_rate": y_test.mean(),
                    "roc_auc": roc_auc_score(y_test, pred),
                    "average_precision": average_precision_score(y_test, pred),
                    "expected_revenue_top_20pct": float(
                        (pred * X_test["emd_price"].to_numpy())[
                            pred >= np.quantile(pred, 0.8)
                        ].mean()
                    ),
                }
            )

        xgb_model = xgb.named_steps["model"]
        feature_names = xgb.named_steps["preprocess"].get_feature_names_out()
        importance = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": xgb_model.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False)
            .head(25)
        )
        importance.to_csv(OUTPUT_DIR / f"feature_importance_{product.replace(' ', '_').lower()}.csv", index=False)

        scored = X_test.copy()
        scored["actual_sold"] = y_test.to_numpy()
        scored["predicted_probability"] = xgb_pred
        scored["expected_revenue"] = scored["predicted_probability"] * scored["emd_price"]
        scored.sort_values("expected_revenue", ascending=False).head(2000).to_csv(
            OUTPUT_DIR / f"top_expected_revenue_opportunities_{product.replace(' ', '_').lower()}.csv",
            index=False,
        )

    results = pd.DataFrame(model_rows).sort_values(
        ["product", "average_precision"], ascending=[True, False]
    )
    results.to_csv(OUTPUT_DIR / "model_results.csv", index=False)
    return results


def save_impact_estimate(product_summary: pd.DataFrame) -> pd.DataFrame:
    top = product_summary.head(TOP_N_PRODUCTS).copy()
    scenarios = []
    for scenario_name, relative_uplift in [
        ("conservative", 0.03),
        ("base", 0.07),
        ("upside", 0.12),
    ]:
        scenario = top.copy()
        scenario["scenario"] = scenario_name
        scenario["relative_cr_uplift"] = relative_uplift
        scenario["extra_purchases"] = scenario["purchases"] * relative_uplift
        scenario["extra_revenue"] = scenario["extra_purchases"] * scenario["avg_sold_price"]
        scenario["new_conversion_rate"] = scenario["conversion_rate"] * (1 + relative_uplift)
        scenarios.append(scenario.reset_index())
    impact = pd.concat(scenarios, ignore_index=True)
    impact.to_csv(OUTPUT_DIR / "economic_impact_scenarios.csv", index=False)
    return impact


def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    print(f"Rows: {len(df):,}, columns: {len(df.columns):,}", flush=True)

    print("Preparing booking features...", flush=True)
    features = prepare_booking_features(df)
    print("Parsing EMD offers...", flush=True)
    emd = build_emd_table(features)
    emd.head(50_000).to_csv(OUTPUT_DIR / "emd_long_sample.csv", index=False)

    print("Saving EDA outputs...", flush=True)
    product_summary = save_product_summary(emd)
    save_eda_charts(features, emd, product_summary)
    impact = save_impact_estimate(product_summary)

    print("Training models...", flush=True)
    model_results = fit_models(features, emd, product_summary)

    print("\nTop products by revenue:")
    print(product_summary.head(TOP_N_PRODUCTS).to_string())
    print("\nModel results:")
    print(model_results.to_string(index=False))
    print("\nImpact scenarios:")
    print(
        impact.groupby("scenario")["extra_revenue"]
        .sum()
        .reindex(["conservative", "base", "upside"])
        .to_string()
    )
    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
