# DSN Mart Sales Prediction : DSN AI Bootcamp Qualification Hackathon 2026 (ML Track)

This repository contains the notebook `dsn_mart_sales3.ipynb`, a sales-prediction solution built for the **DSN Bootcamp Qualification Hackathon 2026 — ML Track: Predict Sale**.

## Competition Overview

DSN Mart operates a chain of stores across Nigeria, ranging from small corner shops to flagship hypermarkets, spread across major urban centers, state capitals, and smaller towns. To plan stock, pricing, and store investment more intelligently, DSN Mart wants to understand what drives sales at the product level across its different store formats and locations.

The task: given historical product-store sales records, build a model that predicts **total sales** for a given product at a given store, based on what is known about the product and the outlet it is sold in.

This challenge served as the qualifying hackathon for the **DSN AI Bootcamp** — leaderboard rank was one of the inputs used to select participants for the bootcamp.

## Notebook Contents

The notebook is organized around a primary, fully engineered pipeline, followed by an appended secondary exploration that re-runs and extends an independent, simpler pipeline.

### Part 1 — Stacked Ensemble Solution (primary pipeline)

A custom stacked ensemble of gradient-boosted trees, nearest neighbors, and extra-trees models:

1. **Setup & Imports** — Loads data from Google Drive; uses scikit-learn, LightGBM, XGBoost, CatBoost, and Optuna.
2. **Target & ID Identification** — Determines the target and ID columns programmatically (by diffing train/test columns) rather than hard-coding names.
3. **Data Inspection** — Dtypes, missing values, duplicates, and cardinality checks.
4. **Exploratory Data Analysis** — Target distribution and transform diagnostics (skew/kurtosis for raw, log1p, and sqrt), categorical sanity checks, numeric correlations, categorical group means, and distribution/relationship visualizations.
5. **Preprocessing & Feature Engineering** — Missing-value handling for `product_weight_kg`; categorical interaction features (e.g. `category_x_format`, `category_x_size`, `format_x_size`); a custom out-of-fold **K-Fold target encoder** with smoothing, plus frequency encoding for high-cardinality columns.
6. **Baseline Modelling** — 5-fold cross-validated baselines for Ridge, Random Forest, Extra Trees, HistGradientBoosting, LightGBM, XGBoost, and CatBoost.
7. **Target Transform Analysis** — Compares raw, log1p, and sqrt response transformations.
8. **Hyperparameter Optimization** — Optuna-based tuning of CatBoost, XGBoost, LightGBM, and Extra Trees.
9. **Validation & Leakage Auditing** — Checks for duplicate rows and train/test key overlap before finalizing the modelling approach.
10. **Final Stacked Repeated-CV Bagging Ensemble** — Trains CatBoost, XGBoost, LightGBM, HistGB, KNN, and Extra Trees across 5 seeds × 5 folds, then combines their out-of-fold predictions with an **OLS linear stacking meta-learner**.
11. **Final Submission** — Generates bagged test predictions, applies the stacking weights, clips predictions at zero, and writes the submission in the exact format required by `sample_submission.csv`.



The bagged, stacked ensemble is the strongest configuration in this section, improving on every individual tuned baseline.

### Part 2 — Independent Exploration & External Data Augmentation

The remainder of the notebook re-enters the problem from scratch with a second, more manual pipeline (its own data loading, cleaning, EDA, and modelling cells), largely without narrative markdown headers. It:

- Re-cleans `product_category` casing and imputes `store_size`.
- Re-benchmarks Ridge, Random Forest, Extra Trees, LightGBM, and CatBoost on an 80/20 holdout.
- Augments the DSN training data with the external **Big Mart Sales** dataset (schema-mapped onto the competition's columns), then retrains a tuned Extra Trees model with engineered features (`price_per_kg`, `format_x_category`) and 5-fold cross-validation.
- Writes a final expanded-data submission (`Finalsubmission_.csv`).

## Data Sources

### Primary dataset
The competition-provided `train.csv`, `test.csv`, and `sample_submission.csv` files (DSN Mart product-store sales records), supplied by DSN for the hackathon.

### External dataset (cited)
Part 2 of the notebook incorporates the classic **Big Mart Sales** dataset to expand the training data, since it shares the same schema, product categories, and store-age structure as the competition data:

> **Big Mart Sales Data** — Kaggle notebook/dataset by *mragpavank*
> Source: [https://www.kaggle.com/code/mragpavank/big-mart-sales-data/input](https://www.kaggle.com/code/mragpavank/big-mart-sales-data/input)

The external rows were schema-mapped onto the competition's column names and categories before being concatenated with the original training data.



## Requirements

- Python 3
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- lightgbm, xgboost, catboost
- optuna

## How to Reproduce

1. Place `train.csv`, `test.csv`, and `sample_submission.csv` (from the DSN Mart competition) in the data directory referenced by `DATA_DIR` (the notebook mounts Google Drive by default; adjust the path for a local run).
2. Run the notebook top to bottom.
3. Part 1 produces the primary stacked-ensemble `submission` DataFrame/file. Part 2, if run, downloads the external Big Mart dataset and writes `Finalsubmission_.csv`.

## Notes & Limitations

- Target encoding is computed out-of-fold (via K-Fold) to reduce leakage risk from the categorical encodings.
- Part 2 of the notebook reuses several variable and column names from Part 1 (e.g. `train`, `TARGET`, `X`); if running the whole notebook sequentially, later cells will overwrite earlier variables — treat Parts 1 and 2 as two related but separately runnable pipelines rather than one continuous flow.
- As with any holdout-based validation, the reported CV RMSE values should be treated as estimates; performance on the live leaderboard may vary.
