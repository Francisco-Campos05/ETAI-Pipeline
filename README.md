# Francisco Campos, nº 20260654

## Overview
The task: predict two-year recidivism using ProPublica's COMPAS dataset -- the data behind a real 2016 investigation into a risk-assessment algorithm actually used by US courts to help inform bail and sentencing decisions.

## Pipeline progress

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Diagnostics & Preprocessing | Replaced hardcoded logic with a config-driven pipeline. Added systematic data diagnostics (invalid domain rules, placeholder token handling, exact/id duplicate checks). Tested missingness mechanisms (MCAR vs MNAR) using Cramer's V. Upgraded preprocessing to include median/most_frequent imputation, `<col>_was_missing` indicators for MNAR columns, Target Encoding for categoricals, and Standard Scaling for numerics based on an empirical grid search. Handled multicollinearity by dropping redundant columns. |
| 4 | Evaluation & Cross-Validation | Added a locked test set (20%). Models are now evaluated using stratified 5-fold cross-validation on the development set. Out-of-fold (OOF) predictions are used for classification and fairness reports. Shifted to scikit-learn's `TargetEncoder` for safe cross-fitting. Evaluated Dummy, unconstrained Decision Tree, Random Forest, and Logistic Regression baselines. | 
| 5 | Hyperparameter Tuning | Integrated **Optuna** for hyperparameter tuning. To estimate the tuning procedure honestly without leakage, implemented **Nested Cross-Validation** (tuning happens entirely inside the outer training folds). `build_pipeline()` now combines preprocessing and model into a single `Pipeline`. | 
| 6 | Ensembles in depth | Added three ensemble models: `random_forest`, `oblique_random_forest` (via `SparseRandomProjection`), and `adaboost`. Included a corrected paired t-test (Nadeau & Bengio) to strictly compare models over the same CV folds and distinguish genuine wins from noise. | 


## Preprocessing decisions
- **Placeholder Tokens**: Replaced missing value placeholders (e.g., `-`, `?`, `n/a`, `N/A`) with `NaN` before processing to ensure correct data typing.
- **Invalid Domain Values**: Applied validity bounds (e.g., `age` must be 18-100, `priors_count` max 60). Out-of-bound/impossible values were explicitly converted to `NaN`.
- **Missing Values (MCAR vs MNAR)**: Instead of dropping rows, imputed numericals with `median` and categoricals with `most_frequent`. For columns diagnosed as MNAR (`priors_count`, `c_charge_degree`), added explicit `<col>_was_missing` boolean indicator columns so the model learns the missingness pattern.
- **Categorical Features**: Upgraded from One-Hot Encoding to `TargetEncoder` (which won the empirical grid evaluation). 
- **Numeric Scaling**: Applied `StandardScaler` to all numeric features (selected via empirical grid search).
- **Multicollinearity/Redundancy**: Dropped `prior_offenses`, `age_in_months`, and `juvenile_total` based on VIF analysis to reduce redundancy.

### Model Comparison

#### *Model Benchmark across 5-Fold Stratified Cross-Validation (Week 5 & 6: Tuned Ensembles & Nested CV)*
In Week 5 & 6, hyperparameter tuning via Optuna and Nested Cross-Validation was introduced to honestely evaluate complex ensemble models without data leakage. 

Tuning successfully closed the massive overfitting gap observed in Week 4. The **AdaBoost** and **Random Forest** models emerged as the top performers, maintaining the highest validation scores (~0.68) with nearly zero overfitting (gaps of +0.004 and +0.009, respectively). While the Oblique Random Forest remained slightly behind, the tuned Decision Tree successfully mitigated its previous severe overfitting, stabilizing at a +0.013 gap.

| Week | Model | Optuna Tuning | Train Acc | Test Acc | Gap (Overfitting) | Verdict & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **6** | AdaBoost | Yes (220 trees, depth 4, lr 0.025) | 0.687 | 0.682 | +0.004 | **(Best Model - Tied)** Best generalization with virtually zero overfitting. |
| **6** | Random Forest | Yes (depth 22, min_samples 42) | 0.690 | 0.681 | +0.009 | **(Best Model - Tied)** Equivalent performance to AdaBoost, massive improvement from untuned state. |
| **6** | Oblique Random Forest | Yes (depth 6, min_samples 3) | 0.692 | 0.679 | +0.013 | Competes closely with standard Random Forest, but no significant advantage found. |
| **5** | Decision Tree | Yes (depth 6, min_samples 101) | 0.685 | 0.672 | +0.013 | Overfitting successfully contained compared to the untuned version (+0.088 gap in W4). |

#### *Model Benchmark across 5-Fold Stratified Cross-Validation (Week 4)*
In Week 4, we evaluated four distinct models using 5-fold stratified cross-validation on the development set, keeping the test set strictly locked. The Logistic Regression model achieved the highest validation performance with a mean validation accuracy of 0.672 and a weighted F1-score of 0.66, maintaining near-zero overfitting with a train-validation gap of only +0.003. 

The Random Forest achieved a higher training accuracy (0.733) but dropped to 0.650 in validation (a +0.083 gap) with a weighted F1-score of 0.65. The unconstrained Decision Tree continued to show clear signs of overfitting, reaching 0.696 in training but only 0.607 in validation (gap of +0.088) and a weighted F1-score of 0.61. Finally, the Dummy baseline confirmed the majority-class floor at 0.549 validation accuracy with an F1-score of 0.39, as it completely failed to recall positive recidivism cases.


| Week | Model | Train Acc | Test Acc | Gap (Overfitting) | F1-Score (Weighted) | Verdict & Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **2** | Logistic Regression | 0.679 | 0.680 | -0.001 | 0.63 | Stable baseline, good generalization. **(Best Model)** |
| **2** | Decision Tree | 0.829 | 0.628 | +0.201 | 0.54 | Severe overfitting on training data; poor test performance. |
| **3** | Logistic Regression | 0.676 | 0.657 | +0.019 | 0.65 | Maintained stable generalization with the new config-driven preprocessing (Target Encoding + Standard Scaler). **(Best Model)** |
| **3** | Decision Tree | 0.792 | 0.610 | +0.182 | 0.60 | Still suffers from strong overfitting, despite the updated data cleaning. |
| **4** | Dummy Classifier (5-Fold CV) | 0.549 (mean) | 0.549 (mean) | -0.000 (mean) | 0.39 | Majority class baseline. Predicts only non-recidivism (0 recall on class 1). |
| **4** | Decision Tree (5-Fold CV) | 0.696 (mean) | 0.607 (mean) | +0.088 (mean) | 0.61 | Underperformed; still suffers from clear overfitting with a high train-val gap. |
| **4** | Random Forest (5-Fold CV) | 0.733 (mean) | 0.650 (mean) | +0.083 (mean) | 0.65 | Stronger ensemble performance, but still shows moderate overfitting on training folds. |
| **4** | Logistic Regression (5-Fold CV) | 0.675 (mean) | 0.672 (mean) | +0.003 (mean) | 0.66 | **(Best Model)** Most robust performance across folds, highest validation accuracy, and negligible overfitting. |

*In week 4, the `logistic_regression model` was the best model, outperforming both tree-based and dummy baselines in cross-validation performance and generalization.*

---
<br>

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── data_diagnostics.py # MCAR/MNAR testing, validity bounds, duplicates
│   ├── preprocessing.py    # cleaning + train/test split + ColumnTransformer
│   ├── model.py            # model construction + build_pipeline
│   ├── evaluate.py         # accuracy metrics, paired t-test + fairness check
│   ├── tuning.py           # Optuna tuning, nested CV and tuning reports
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Environment setup

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:

**Windows**
```powershell
venv\Scripts\activate
python main.py
```

Activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.
To leave it later, run `deactivate`.

## Running the pipeline

With the environment active, from the project's root folder:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the models, tunes them via Nested CV (if enabled), and prints:
- **Nested Cross-Validation Table:** Honest estimation of the pipeline performance across outer folds.
- **Optimism Check & Best Hyperparameters.**
- a classification report on the out-of-fold predictions.
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/`, so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.