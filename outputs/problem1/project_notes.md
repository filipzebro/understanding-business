# Problem 1 - EMD revenue optimization

## Recommended modeling approach

Use CatBoost binary classifiers per EMD product and rank products by expected revenue:

```text
expected_revenue = predicted_purchase_probability * offered_emd_price
```

This fits the business goal better than pure conversion optimization, because the pitch is about ancillary revenue, not only CR.

## Why not LSTM

The dataset is mostly tabular booking context with JSON-like route and EMD fields. There is no real clickstream, session event sequence, or customer history sequence, so LSTM would add complexity without a strong data reason. Segment lists are better converted into tabular features such as route direction, connection flag, booking class, travel span, and stay length.

## Product priority

Top products by revenue:

| Product | Revenue share | CR |
|---|---:|---:|
| BAGGAGE | 68.0% | 5.25% |
| PET | 16.4% | 1.16% |
| SPECIAL EQUIPMENT | 7.5% | 0.64% |

Business implication: focus the story on BAGGAGE first, then PET and SPECIAL EQUIPMENT. FAST TRACK and MEAL convert better than some products, but they are much smaller revenue pools.

## Model choice

Use three layers in the presentation:

1. Manual rules as a business-friendly baseline.
2. Logistic regression as an interpretable benchmark.
3. CatBoost as the final model.

CatBoost won on every top-revenue product after the additional comparison run:

| Product | CatBoost ROC AUC | XGBoost ROC AUC | CatBoost AP | XGBoost AP |
|---|---:|---:|---:|---:|
| BAGGAGE | 0.839 | 0.802 | 0.308 | 0.229 |
| PET | 0.817 | 0.773 | 0.139 | 0.053 |
| SPECIAL EQUIPMENT | 0.871 | 0.846 | 0.131 | 0.066 |

Average precision is low in absolute terms because purchases are rare, which is normal for ancillary conversion data. The important business use is ranking: put the best opportunities at the top.

## Uplift logic

Use scenario-based uplift, not a fake causal claim:

```text
extra_purchases = current_purchases * assumed_relative_CR_uplift
extra_revenue = extra_purchases * average_sold_price
```

For the top 3 revenue products:

| Scenario | Assumed relative CR uplift | Extra revenue |
|---|---:|---:|
| Conservative | 3% | 15,918 |
| Base | 7% | 37,141 |
| Upside | 12% | 63,670 |

Say clearly that this should be validated with an A/B test in production.

## Slide storyline

1. Problem: current EMD ordering leaves revenue on the table.
2. Revenue priority: BAGGAGE dominates revenue, so optimize there first.
3. Data prep: parse EMDs, route direction, synthetic OND areas, DTD/fare/stay buckets.
4. EDA: revenue vs conversion and top segments by CR.
5. Drivers: correlation heatmap plus feature importance for BAGGAGE.
6. Solution: XGBoost per product, rank by expected revenue.
7. Impact: conservative/base/upside uplift scenarios.
8. Next step: A/B test and add real eligibility constraints from LOT.com.

## Generated files

- `product_summary.csv` - product revenue, CR, price summary.
- `revenue_vs_conversion.png` - slide-ready product chart.
- `correlation_heatmap.png` - slide-ready driver heatmap.
- `model_results.csv` - logistic vs XGBoost comparison.
- `catboost/model_comparison_with_catboost.csv` - logistic vs XGBoost vs CatBoost comparison.
- `catboost/catboost_vs_xgboost.png` - slide-ready model comparison chart.
- `feature_importance_*.csv` - top model drivers per product.
- `top_segments_by_cr.csv` - quick EDA segments for slides.
- `economic_impact_scenarios.csv` - uplift and revenue impact.
- `top_expected_revenue_opportunities_*.csv` - examples of high-value opportunities.
