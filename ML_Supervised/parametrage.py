from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    ParameterGrid,
    cross_validate,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import numpy as np

class LiveVerboseGridSearch(GridSearchCV):

    def _run_search(self, evaluate_candidates):
        
        param_sets = list(ParameterGrid(self.param_grid))
        print(f"\nNolbres de combinations: {len(param_sets)}")

        scoring = {
            "recall": "recall",
            "precision": "precision",
            "accuracy": "accuracy",
            "f1": "f1",
            "roc_auc": "roc_auc",
        }

        for i, params in enumerate(param_sets, start=1):
            print(f"\nCombination Num_{i}/{len(param_sets)}")
            print("Params:", params)
            print("-----------------------------------")

            estimator = self.estimator.set_params(**params)

            # ---------- FAST CROSS VALIDATION ----------
            cv_results = cross_validate(
                estimator,
                self.X_train_cv,
                self.y_train_cv,
                cv=self.cv,
                scoring=scoring,
                n_jobs=self.n_jobs,
                return_train_score=False,
            )

            print("Metriques de la Cross-validation :")
            for metric in ["accuracy", "precision", "recall", "f1"]:
                scores = cv_results[f"test_{metric}"]
                print(f"   • {metric}: {scores.mean():.4f} ")

            # ROC AUC (may fail for models without predict_proba)
            if "test_roc_auc" in cv_results:
                scores = cv_results["test_roc_auc"]
                print(f"   • roc_auc: {scores.mean():.4f}")
            else:
                print("   • roc_auc: Non dispo")

            print("-----------------------------------")

            # Now let GridSearchCV evaluate this param set normally
            evaluate_candidates([params])


# ============================================================
# Tune and evaluate function
# ============================================================
def tune_and_evaluate(
    balanced_train_df,
    test_df,
    estimator,
    param_grid,
    target_col="fire_count",
    cv_folds=5,
    random_state=42,
    scoring="recall",
):
    # ---------- Split data ----------
    X_train = balanced_train_df.drop(columns=[target_col])
    y_train = balanced_train_df[target_col].astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    # Store training data for the custom grid class
    LiveVerboseGridSearch.X_train_cv = X_train
    LiveVerboseGridSearch.y_train_cv = y_train

    # ---------- CV strategy ----------
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # ---------- Grid Search ----------
    grid = LiveVerboseGridSearch(
        estimator=estimator,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        verbose=0,
        refit=True,
        error_score="raise",
    )

    print("\n...GridSearchCV...\n")
    grid.fit(X_train, y_train)

    print("\nMeilleure combinaison de parameteres:", grid.best_params_)
    print(f"Meilleur Cross-Validation {scoring}: {grid.best_score_:.4f}\n")

    # ---------- Test Evaluation ----------
    y_pred = grid.best_estimator_.predict(X_test)
    try:
        y_proba = grid.best_estimator_.predict_proba(X_test)[:, 1]
    except:
        y_proba = None

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    print("\n...EVALUATION SUR X_TEST...\n")
    for k, v in test_metrics.items():
        if k not in ["confusion_matrix", "classification_report", "roc_auc"]:
            print(f"{k}: {v:.4f}")
    if test_metrics["roc_auc"] is not None:
        print(f"roc_auc: {test_metrics['roc_auc']:.4f}")

    print("\nMatrice de confusion:\n", test_metrics["confusion_matrix"])
    print("\nRapport de classification:\n", test_metrics["classification_report"])

    return {
        "best_model": grid.best_estimator_,
        "grid_search": grid,
        "test_metrics": test_metrics,
    }
