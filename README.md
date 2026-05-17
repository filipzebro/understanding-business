# UB Project - Problem 1 Findings

## Executive Summary

The project focuses on increasing ancillary revenue from EMD products. The main
finding is that recommendations should be optimized for **expected revenue**, not
only for conversion rate.

The recommended logic is:

```text
expected_revenue = predicted_purchase_probability * offered_emd_price
```

This means the best product to show first is not always the product with the
highest generic conversion rate. It is the product with the highest expected
business value for a specific booking context.

## Revenue Priority

Ancillary revenue is highly concentrated in a small number of products.

| Product | Revenue Share | Conversion Rate |
|---|---:|---:|
| BAGGAGE | 68.0% | 5.25% |
| PET | 16.4% | 1.16% |
| SPECIAL EQUIPMENT | 7.5% | 0.64% |
| FAST TRACK | 3.4% | 2.46% |
| MEAL | 3.0% | 1.13% |
| BUSINESS LOUNGE | 1.8% | 0.47% |

The strongest business opportunity is **BAGGAGE**. It generates almost 68% of
total ancillary revenue and also has the highest conversion rate. The second
priority should be **PET** and **SPECIAL EQUIPMENT**, because they have lower
conversion but higher revenue potential than smaller products.

FAST TRACK has a relatively good conversion rate, but its low unit price makes it
less important from a revenue perspective. MEAL and BUSINESS LOUNGE are useful
for personalization, but they should not drive the main revenue story.

## Main Business Insight

The current EMD recommendation problem should be treated as a **revenue ranking
problem**, not just a product conversion problem.

Instead of asking:

```text
Which product has the highest probability of purchase?
```

the better business question is:

```text
Which product has the highest expected revenue for this booking?
```

This is especially important because products differ strongly in price and
purchase frequency. A lower-conversion product can still be valuable if the
expected revenue is high.

## Key Purchase Drivers

### BAGGAGE

The strongest signals for baggage purchase are:

- larger passenger groups, especially 4+ passengers,
- presence of children or infants,
- market,
- multi-city trips,
- high fare bucket,
- long days-to-departure,
- high EMD price,
- longer stay or more planned trips.

Business interpretation: baggage is most attractive for families, larger groups,
more complex trips and higher-value bookings. This should be the core use case
for the recommendation strategy.

### PET

The strongest signals for PET purchase are:

- market,
- booking class / RBD,
- multi-city trips,
- route direction,
- hub vs non-hub route.

Business interpretation: PET is a niche product, but it has meaningful revenue
because of its higher unit value. It should be targeted to specific markets and
route contexts rather than broadly promoted.

### SPECIAL EQUIPMENT

The strongest signals for SPECIAL EQUIPMENT purchase are:

- multi-city trips,
- market,
- route direction,
- RBD,
- hub vs non-hub route.

Business interpretation: SPECIAL EQUIPMENT behaves similarly to a specialized
travel-need product. It is less frequent, but can be relevant for more complex
travel patterns and selected route/market combinations.

### FAST TRACK, MEAL and BUSINESS LOUNGE

These products show segment-level patterns, but they are secondary in the revenue
story.

FAST TRACK converts relatively well, but the average price is low. MEAL has
limited revenue impact. BUSINESS LOUNGE has low conversion and low total revenue
in this dataset, although some segments show high lift and may be useful for
future personalization.

## Model Findings

CatBoost is the strongest tested model for the top revenue products.

| Product | Best Model | ROC AUC | Average Precision |
|---|---|---:|---:|
| BAGGAGE | CatBoost | 0.839 | 0.308 |
| PET | CatBoost | 0.817 | 0.139 |
| SPECIAL EQUIPMENT | CatBoost | 0.871 | 0.131 |

CatBoost outperformed both XGBoost and logistic regression across the top revenue
products. This makes sense because the dataset contains many categorical
features, such as market, route direction, booking class, RBD, language, browser,
touchpoint channels and bucketed fare/DTD/stay variables.

Average precision remains modest because EMD purchases are rare events. This is
expected in ancillary conversion data. The model should therefore be used mainly
for ranking and prioritization, not as a perfect individual-level purchase
prediction.

## Estimated Business Impact

A simple scenario-based uplift estimate was used for the top three revenue
products. The estimate assumes that better ranking and targeting increase
conversion rate by a relative uplift.

| Scenario | Assumed Relative CR Uplift | Estimated Extra Revenue |
|---|---:|---:|
| Conservative | 3% | 15,918 |
| Base | 7% | 37,141 |
| Upside | 12% | 63,670 |

This should be presented as a business scenario, not a causal proof. The final
impact would need to be validated through an A/B test.

## Recommended Business Direction

The best recommendation strategy is:

1. Prioritize BAGGAGE as the main revenue lever.
2. Use PET and SPECIAL EQUIPMENT as targeted high-value secondary products.
3. Rank products by expected revenue, not only predicted conversion.
4. Use CatBoost scores to personalize product ordering per booking.
5. Validate the recommendation order with an A/B test.

The final pitch should be:

```text
Better EMD ordering can increase ancillary revenue by showing the highest-value
product to the right passenger at the right moment.
```
