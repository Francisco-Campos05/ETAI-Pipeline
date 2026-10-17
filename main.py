"""
Entry point for the baseline predictive pipeline.

Run with:
    python main.py
"""
import optuna
import yaml
from sklearn.model_selection import StratifiedKFold

from src.data import load_data
from src.preprocessing import clean_dataset, drop_duplicate_rows, split_features_target, split_train_test
from src.model import build_pipeline
from src.evaluate import cross_validate_pipeline, cv_report, oof_classification_report, fairness_report
from src.tuning import tune_pipeline, nested_cross_validate, tuning_report
from src.results import save_run


def load_config(path: str = "config.yaml") -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    config = load_config()

    # load + diagnose-and-clean (week 3): nothing here is learned from the data, so it's
    # safe to run on the whole dataset -- see src/preprocessing.py
    df_raw = load_data(config["data"]["path"])
    df_clean = clean_dataset(df_raw, config["diagnostics"])          # row-preserving: also safe for new data
    df_clean = drop_duplicate_rows(df_clean, config["diagnostics"].get("id_column"))

    mnar_sources = config["preprocessing"].get("mnar_indicator_sources", [])
    X, y, extras = split_features_target(df_clean, config["data"], mnar_sources)

    # week 4: the final test set is set aside HERE and never used again in this script.
    # Every decision from now on (preprocessing, model, hyperparameters) is made on the
    # development set only. The test set is used only for the final assessment.
    X_train, X_test, y_train, y_test, extras_train, extras_test = split_train_test(
        X, y, extras,
        test_size=config["test_set"]["size"],
        random_state=config["test_set"]["random_state"],
    )

    pipeline = build_pipeline(config["preprocessing"], config["model"])

    # a fixed random_state = the same folds on every run and for every model, so comparing
    # two models' fold scores is a like-for-like (paired) comparison
    cv_config = config["cv"]
    shuffle = cv_config.get("shuffle", True)
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=shuffle,
                         random_state=cv_config.get("random_state") if shuffle else None)
    scoring = cv_config.get("scoring", "accuracy")
    n_jobs = cv_config.get("n_jobs", 1)

    # fold scores + out-of-fold predictions (each row predicted by the fold model that did NOT train on it)
    fold_scores, y_oof = cross_validate_pipeline(
        pipeline, X_train, y_train, cv, scoring, n_jobs=n_jobs
    )
    header = f"Hyperparameters from config.yaml: {config['model'].get('params')}"
    print(header)
    report = header + "\n" + cv_report(fold_scores, scoring)

    # week 5: hyperparameter tuning with Optuna + nested cross-validation
    tuning_config = config.get("tuning", {})
    tuning_enabled = tuning_config.get("enabled", False)
    if tuning_enabled:
        model_type = config["model"]["type"]
        search_spaces = tuning_config.get("search_spaces") or {}
        if model_type not in search_spaces:
            raise ValueError(f"tuning is enabled but config.yaml has no tuning.search_spaces for "
                             f"'{model_type}'. Add one, or set tuning.enabled: false.")
        search_space = search_spaces[model_type]
        n_trials = tuning_config["n_trials"]
        tuning_seed = tuning_config["random_state"]
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        inner_cv = StratifiedKFold(n_splits=tuning_config["n_splits"], shuffle=True, random_state=tuning_seed)

        # Honest estimate on the same outer folds; its out-of-fold predictions feed the reports.
        print(f"\nNested cross-validation: {cv.get_n_splits()} outer folds x {n_trials} trials x "
              f"{inner_cv.get_n_splits()} inner folds ...")
        nested_scores, y_oof = nested_cross_validate(
            pipeline, X_train, y_train, cv, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        report += "\n\nNested cross-validation (the tuning procedure, estimated honestly):\n"
        report += cv_report(nested_scores, scoring)

        # The hyperparameters that are kept: the same procedure, once, on all development rows.
        pipeline, study = tune_pipeline(
            pipeline, X_train, y_train, inner_cv, scoring, search_space, n_trials, tuning_seed, n_jobs=n_jobs
        )
        print()
        report += "\n\n" + tuning_report(study, nested_scores, scoring)

    report += "\n\n" + oof_classification_report(y_train, y_oof)
    report += "\n" + fairness_report(
        y_train, y_oof, extras_train, sensitive_attr=config["data"]["sensitive_attr"]
    )

    # the model we'd actually use: same pipeline, refit on EVERY development row. CV above
    # estimated how well this recipe does; it didn't produce a model.
    final_model = pipeline.fit(X_train, y_train)
    refit = f"Final model: {config['model']['type']} refit on all {len(X_train)} development rows."
    if tuning_enabled:
        refit += f" Tuned hyperparameters: {study.best_params}"
    print(refit)
    report += "\n" + refit + "\n"

    locked = (f"Locked test set: {len(X_test)} rows set aside, not evaluated. "
              f"Development set: {len(X_train)} rows.")
    print(locked)
    report += "\n" + locked + "\n"

    results_dir = config.get("output", {}).get("results_dir", "results")
    path = save_run(results_dir, config, report)
    print(f"Full results saved to {path}")


if __name__ == "__main__":
    main()