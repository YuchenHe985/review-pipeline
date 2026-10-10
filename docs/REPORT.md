# Pipeline report: Health_and_Personal_Care

ingest `ce382b1697a4a4a9` -> build `653b1c4f073308a8` -> train `575215318583ca11`

## Data

| | read | accepted | quarantined |
|---|---:|---:|---:|
| reviews | 494,121 | 494,121 | 0 |
| products | 60,293 | 60,293 | 0 |

Duplicates removed: 5,131. Examples: 488,990 (0 without product metadata). Reviewers: 461,656.

## Helpful-vote model (target: log(1 + helpful votes))

Time split: train 391,192 reviews up to 2021-04-18, test 97,798 reviews from 2021-04-18. 23.4% of test reviews have at least one helpful vote.

| model | RMSE | MAE | R² | Spearman | top-10% helpful share | lift |
|---|---:|---:|---:|---:|---:|---:|
| train_mean | 0.5534 | 0.4389 | -0.0294 | 0.000 | 23.1% | 0.99× |
| ridge | 0.5473 | 0.3544 | -0.0066 | 0.237 | 45.1% | 1.93× |
| xgboost | 0.5286 | 0.3521 | 0.0609 | 0.281 | 50.0% | 2.13× |

Bootstrap 95% intervals for R² on the test set:

- train_mean: -0.0294 [-0.0321, -0.0267]
- ridge: -0.0065 [-0.0355, 0.0192]
- xgboost: 0.0608 [0.0512, 0.0709]
- xgboost_minus_ridge: 0.0673 [0.0437, 0.0960]

What shortcuts would have reported for the same XGBoost model (R², for comparison only):

- random 80/20 split instead of a time split: 0.2797
- time split plus the catalogue's snapshot rating columns: 0.1672
- the honest setup above: 0.0609

Largest feature gain shares: text_len 28.5%, verified 12.2%, product_prior_reviews 7.5%, price 7.5%, image_count 7.1%

## Reviewer segments

18,959 reviewers. k = 6 by silhouette (0.348). Stability across seeds: mean ARI 0.997, min 0.994.

| segment | reviewers | avg rating | rating sd | avg log length | avg log helpful | verified | with images |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2,405 | 4.39 | 0.58 | 5.85 | 0.21 | 11.2% | 2.6% |
| 1 | 2,448 | 4.54 | 0.43 | 5.61 | 1.23 | 92.0% | 2.4% |
| 2 | 1,389 | 1.75 | 0.51 | 4.72 | 0.38 | 91.8% | 2.9% |
| 3 | 7,556 | 4.81 | 0.16 | 4.13 | 0.16 | 97.6% | 0.4% |
| 4 | 1,246 | 4.34 | 0.65 | 5.96 | 0.41 | 37.9% | 69.4% |
| 5 | 3,915 | 3.32 | 2.13 | 4.63 | 0.30 | 94.0% | 1.6% |
