# Francisco Campos, nº 20260654

## Overview
The task: predict two-year recidivism using ProPublica's COMPAS dataset -- the data behind a real 2016 investigation into a risk-assessment algorithm actually used by US courts to help inform bail and sentencing decisions.

## Pipeline progress

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |
| 3 | Diagnostics & Preprocessing | Replaced hardcoded logic with a config-driven pipeline. Added systematic data diagnostics (invalid domain rules, placeholder token handling, exact/id duplicate checks). Tested missingness mechanisms (MCAR vs MNAR) using Cramer's V. Upgraded preprocessing to include median/most_frequent imputation, `<col>_was_missing` indicators for MNAR columns, Target Encoding for categoricals, and Standard Scaling for numerics based on an empirical grid search. Handled multicollinearity by dropping redundant columns. |


## Preprocessing decisions
- **Placeholder Tokens**: Replaced missing value placeholders (e.g., `-`, `?`, `n/a`, `N/A`) with `NaN` before processing to ensure correct data typing.
- **Invalid Domain Values**: Applied validity bounds (e.g., `age` must be 18-100, `priors_count` max 60). Out-of-bound/impossible values were explicitly converted to `NaN`.
- **Missing Values (MCAR vs MNAR)**: Instead of dropping rows, imputed numericals with `median` and categoricals with `most_frequent`. For columns diagnosed as MNAR (`priors_count`, `c_charge_degree`), added explicit `<col>_was_missing` boolean indicator columns so the model learns the missingness pattern.
- **Categorical Features**: Upgraded from One-Hot Encoding to `TargetEncoder` (which won the empirical grid evaluation). 
- **Numeric Scaling**: Applied `StandardScaler` to all numeric features (selected via empirical grid search).
- **Multicollinearity/Redundancy**: Dropped `prior_offenses`, `age_in_months`, and `juvenile_total` based on VIF analysis to reduce redundancy.

## Best Model
**Logistic Regression vs. Decision Tree**
The Logistic Regression model demonstrated better generalization, maintaining a stable accuracy of 0.679 in training and 0.680 in testing. Achieving a F1-score of 0.63. 
In contrast, the Decision Tree had a 0.829 accuracy in training but its performance degraded on the test set to 0.628, causing the F1-score to drop to 0.54. <br>
Therefore, the Logistic Regression model performed better than the Decision Tree.

---

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── data_diagnostics.py # MCAR/MNAR testing, validity bounds, duplicates
│   ├── preprocessing.py    # cleaning + train/test split + ColumnTransformer
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Environment setup

You only need to do this once per machine.

### Windows -- PowerShell
```powershell
python -m venv venv                  # creates an isolated Python environment in a folder called "venv"
venv\Scripts\activate                # activates it -- packages install here, not system-wide, and stay out of your other projects
pip install -r requirements.txt      # installs the exact packages this project needs, into that environment
```
If PowerShell blocks the activation script, run this once first:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Once the environment is active you'll see `(venv)` at the start of your prompt. To leave it later, run `deactivate` (same command on every OS).

### Every time after the first

Creating the environment and installing packages only needs to happen once, ever. Every other time you sit down to work -- a new terminal window, the next practical class, tomorrow -- you don't repeat any of the steps above. From the project's root folder, you just need to:


**Windows**
```powershell
venv\Scripts\activate
python main.py
```

That's it -- activate, then run. If you don't see `(venv)` at the start of your prompt, the environment isn't active and `python main.py` may use the wrong Python (or fail to find a package) entirely.

## Running the pipeline

With the environment active (see above), from the project's root
folder, on any OS:
```bash
python main.py
```

This loads `config.yaml`, loads and preprocesses the data, trains the model, and prints:
- **train accuracy and test accuracy, side by side.** Comparing the two is how you catch overfitting: if the model looks much better on the data it was trained on than on data it's never seen, it has memorised rather than learned something that generalises. 
- a classification report on the test set
- a false-positive-rate-by-race comparison between our model and
  COMPAS's own score

All of this is also saved to a timestamped file in `results/` (e.g.`results/run_20260916_143012.txt`), so it doesn't just scroll past in your terminal -- open it later, or change something in `config.yaml` (like the model type) and compare the new file to the last one.
`results/` is created automatically the first time you run the
pipeline, and isn't tracked in git (see `.gitignore`) since it's
generated output, not source.

You're free to improve on this structure or restructure it entirely -- what matters is that your project stays runnable end-to-end with a single command, and that each piece (data, preprocessing, model, evaluation) stays easy to find and change independently.

## Dataset

See `data/README.md`.
