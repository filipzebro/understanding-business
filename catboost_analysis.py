from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from problem1_pipeline import (
    DATA_PATH,
    OUTPUT_DIR,
    RANDOM_STATE,
    TOP_N_PRODUCTS,
    build_emd_table,
    prepare_booking_features,
    save_product_summary,
)


CATBOOST_OUTPUT_DIR = OUTPUT_DIR / "catboost"
MAX_TRAIN_ROWS_PER_PRODUCT = 70_000


NUMERIC_COLS = [
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

CATEGORICAL_COLS = [
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


def prepare_catboost_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df[NUMERIC_COLS + CATEGORICAL_COLS].copy()
    for column in CATEGORICAL_COLS:
        prepared[column] = prepared[column].astype("object").where(
            prepared[column].notna(), "MISSING"
        )
        prepared[column] = prepared[column].astype(str)
    return prepared


def sample_training_data(X_train_full: pd.DataFrame, y_train_full: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    train_df = X_train_full.copy()
    train_df["sold"] = y_train_full.to_numpy()

    if len(train_df) <= MAX_TRAIN_ROWS_PER_PRODUCT:
        return X_train_full, y_train_full

    pos = train_df[train_df["sold"] == 1]
    neg = train_df[train_df["sold"] == 0]
    target_neg_n = max(MAX_TRAIN_ROWS_PER_PRODUCT - len(pos), min(len(neg), len(pos) * 4))
    sampled_neg = neg.sample(n=min(target_neg_n, len(neg)), random_state=RANDOM_STATE)
    sampled = pd.concat([pos, sampled_neg]).sample(frac=1, random_state=RANDOM_STATE)
    return sampled.drop(columns=["sold"]), sampled["sold"]


def train_catboost(product: str, product_df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    X = prepare_catboost_frame(product_df)
    y = product_df["sold"].astype(int)

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    X_train, y_train = sample_training_data(X_train_full, y_train_full)

    cat_features = [X.columns.get_loc(column) for column in CATEGORICAL_COLS]
    model = CatBoostClassifier(
        iterations=450,
        depth=6,
        learning_rate=0.045,
        loss_function="Logloss",
        eval_metric="AUC",
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_test, y_test),
        use_best_model=True,
        early_stopping_rounds=40,
    )

    pred = model.predict_proba(X_test)[:, 1]
    metrics = {
        "product": product,
        "model": "catboost",
        "train_rows_used": len(X_train),
        "test_rows": len(X_test),
        "train_positive_rate": float(y_train.mean()),
        "test_positive_rate": float(y_test.mean()),
        "roc_auc": roc_auc_score(y_test, pred),
        "average_precision": average_precision_score(y_test, pred),
        "expected_revenue_top_20pct": float(
            (pred * X_test["emd_price"].to_numpy())[pred >= pd.Series(pred).quantile(0.8)].mean()
        ),
        "best_iteration": model.get_best_iteration(),
    }

    feature_importance = (
        pd.DataFrame(
            {
                "feature": X.columns,
                "importance": model.get_feature_importance(),
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    scored = X_test.copy()
    scored["actual_sold"] = y_test.to_numpy()
    scored["predicted_probability"] = pred
    scored["expected_revenue"] = scored["predicted_probability"] * scored["emd_price"]
    scored = scored.sort_values("expected_revenue", ascending=False)
    return metrics, feature_importance, scored


def save_comparison_chart(comparison: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    plot_df = comparison[comparison["model"].isin(["xgboost", "catboost"])].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    sns.barplot(data=plot_df, x="product", y="roc_auc", hue="model", ax=axes[0])
    axes[0].set_title("ROC AUC by product")
    axes[0].set_xlabel("")
    axes[0].tick_params(axis="x", rotation=20)

    sns.barplot(data=plot_df, x="product", y="average_precision", hue="model", ax=axes[1])
    axes[1].set_title("Average Precision by product")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(CATBOOST_OUTPUT_DIR / "catboost_vs_xgboost.png", dpi=180)
    plt.close(fig)


def main() -> None:
    CATBOOST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and preparing data...", flush=True)
    df = pd.read_parquet(DATA_PATH)
    features = prepare_booking_features(df)
    emd = build_emd_table(features)
    product_summary = save_product_summary(emd)

    merged = emd.merge(features, on="booking_id", how="left")
    top_products = product_summary.head(TOP_N_PRODUCTS).index.tolist()

    rows = []
    for product in top_products:
        print(f"Training CatBoost for {product}...", flush=True)
        product_df = merged[merged["product"] == product].copy()
        metrics, importance, scored = train_catboost(product, product_df)
        rows.append(metrics)

        slug = product.lower().replace(" ", "_")
        importance.to_csv(CATBOOST_OUTPUT_DIR / f"catboost_feature_importance_{slug}.csv", index=False)
        scored.head(2_000).to_csv(
            CATBOOST_OUTPUT_DIR / f"catboost_top_expected_revenue_{slug}.csv",
            index=False,
        )

    catboost_results = pd.DataFrame(rows).sort_values("product")
    catboost_results.to_csv(CATBOOST_OUTPUT_DIR / "catboost_model_results.csv", index=False)

    existing_results_path = OUTPUT_DIR / "model_results.csv"
    if existing_results_path.exists():
        existing_results = pd.read_csv(existing_results_path)
        comparison = pd.concat([existing_results, catboost_results], ignore_index=True, sort=False)
    else:
        comparison = catboost_results
    comparison.to_csv(CATBOOST_OUTPUT_DIR / "model_comparison_with_catboost.csv", index=False)
    save_comparison_chart(comparison)

    print("\nCatBoost results:")
    print(catboost_results.to_string(index=False))
    print(f"\nSaved outputs to: {CATBOOST_OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
