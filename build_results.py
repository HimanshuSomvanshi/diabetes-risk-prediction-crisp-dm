"""
Re-runs the diabetes CRISP-DM pipeline (same logic as the notebook) and dumps every
number the Streamlit app needs into results.json + saves model artifacts.
Single source of truth so the app never shows a number that doesn't match the notebook.
"""
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, precision_recall_curve,
)

np.random.seed(42)
RANDOM_STATE = 42

diabetes_data = pd.read_csv("data/diabetes.csv")
train_data, test_data = train_test_split(
    diabetes_data, test_size=0.2, random_state=RANDOM_STATE, stratify=diabetes_data["Outcome"]
)
feature_columns = [c for c in train_data.columns if c != "Outcome"]
X_train = train_data[feature_columns].copy()
y_train = train_data["Outcome"].copy()
X_test = test_data[feature_columns].copy()
y_test = test_data["Outcome"].copy()

heavy_missing_cols = ["SkinThickness", "Insulin"]
for col in heavy_missing_cols:
    X_train[f"{col}_was_missing"] = (X_train[col] == 0).astype(int)
    X_test[f"{col}_was_missing"] = (X_test[col] == 0).astype(int)

impute_columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
train_medians = X_train[impute_columns].replace(0, np.nan).median()
for col in impute_columns:
    X_train[col] = X_train[col].replace(0, np.nan).fillna(train_medians[col])
    X_test[col] = X_test[col].replace(0, np.nan).fillna(train_medians[col])

sc_X = StandardScaler()
final_feature_columns = X_train.columns.tolist()
X_train_scaled = pd.DataFrame(sc_X.fit_transform(X_train), columns=final_feature_columns, index=X_train.index)
X_test_scaled = pd.DataFrame(sc_X.transform(X_test), columns=final_feature_columns, index=X_test.index)
X_train, X_test = X_train_scaled, X_test_scaled

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
param_grids = {
    "Logistic Regression": {
        "estimator": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"C": [0.01, 0.1, 1, 10]},
    },
    "Decision Tree": {
        "estimator": DecisionTreeClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"max_depth": [3, 5, 7, None], "min_samples_leaf": [1, 5, 10]},
    },
    "Random Forest": {
        "estimator": RandomForestClassifier(random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"n_estimators": [200, 300], "max_depth": [5, 10, None], "min_samples_leaf": [1, 5]},
    },
    "Gradient Boosting": {
        "estimator": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "params": {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [2, 3]},
    },
    "SVM": {
        "estimator": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE, class_weight="balanced"),
        "params": {"C": [0.1, 1, 10], "gamma": ["scale", "auto"]},
    },
}

trained_models = {}
cv_summary = []
for name, cfg in param_grids.items():
    grid = GridSearchCV(cfg["estimator"], cfg["params"], scoring="roc_auc", cv=cv, n_jobs=-1)
    grid.fit(X_train, y_train)
    trained_models[name] = grid.best_estimator_
    cv_recall = cross_val_score(grid.best_estimator_, X_train, y_train, cv=cv, scoring="recall")
    cv_f1 = cross_val_score(grid.best_estimator_, X_train, y_train, cv=cv, scoring="f1")
    cv_summary.append({
        "model": name,
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "cv_roc_auc_mean": float(grid.best_score_),
        "cv_recall_mean": float(cv_recall.mean()),
        "cv_recall_std": float(cv_recall.std()),
        "cv_f1_mean": float(cv_f1.mean()),
        "cv_f1_std": float(cv_f1.std()),
    })

model_results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    model_results.append({
        "model": name,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    })
model_results_sorted = sorted(model_results, key=lambda r: r["recall"], reverse=True)
best_3 = [r["model"] for r in model_results_sorted[:3]]

ensemble = VotingClassifier(estimators=[(n, trained_models[n]) for n in best_3], voting="soft")
ensemble.fit(X_train, y_train)
ens_pred = ensemble.predict(X_test)
ens_prob = ensemble.predict_proba(X_test)[:, 1]
ensemble_result = {
    "model": "Soft Voting Ensemble",
    "base_models": best_3,
    "accuracy": float(accuracy_score(y_test, ens_pred)),
    "precision": float(precision_score(y_test, ens_pred, zero_division=0)),
    "recall": float(recall_score(y_test, ens_pred, zero_division=0)),
    "f1": float(f1_score(y_test, ens_pred, zero_division=0)),
    "roc_auc": float(roc_auc_score(y_test, ens_prob)),
}

best_model_name = model_results_sorted[0]["model"]
best_model = trained_models[best_model_name]

# Threshold analysis on best model
y_prob_best = best_model.predict_proba(X_test)[:, 1]
precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob_best)
f1_scores = np.divide(
    2 * precisions * recalls, precisions + recalls,
    out=np.zeros_like(precisions), where=(precisions + recalls) != 0,
)
best_f1_idx = int(np.argmax(f1_scores[:-1]))
best_f1_threshold = float(thresholds[best_f1_idx])

FN_COST, FP_COST = 4, 1
costs = []
for thresh in thresholds:
    y_pred_t = (y_prob_best >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    costs.append(FN_COST * fn + FP_COST * fp)
costs = np.array(costs)
cost_idx = int(np.argmin(costs))
cost_optimal_threshold = float(thresholds[cost_idx])

def metrics_at(thresh):
    y_pred_t = (y_prob_best >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_t).ravel()
    return {
        "threshold": float(thresh),
        "precision": float(precision_score(y_test, y_pred_t, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred_t, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred_t, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "weighted_cost": float(FN_COST * fn + FP_COST * fp),
    }

threshold_comparison = {
    "default": metrics_at(0.5),
    "f1_optimized": metrics_at(best_f1_threshold),
    "cost_weighted": metrics_at(cost_optimal_threshold),
}

# Feature importance / sensitivity (Insulin in/out) on best model
if hasattr(best_model, "feature_importances_"):
    importances = dict(zip(X_train.columns, [float(x) for x in best_model.feature_importances_]))
elif hasattr(best_model, "coef_"):
    importances = dict(zip(X_train.columns, [float(x) for x in best_model.coef_[0]]))
else:
    importances = {}

# Zero-count diagnostics (for Data Understanding tab)
zero_counts = {}
for col in ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction"]:
    zero_counts[col] = {
        "count": int((train_data[col] == 0).sum()),
        "pct": float((train_data[col] == 0).sum() / len(train_data) * 100),
    }

correlations = diabetes_data.corr(numeric_only=True)["Outcome"].sort_values(ascending=False)
correlations_dict = {k: float(v) for k, v in correlations.items()}

results = {
    "dataset": {
        "n_rows": int(diabetes_data.shape[0]),
        "n_features": len(feature_columns),
        "outcome_counts": {str(k): int(v) for k, v in diabetes_data["Outcome"].value_counts().items()},
        "outcome_pct": {str(k): float(v) for k, v in (diabetes_data["Outcome"].value_counts(normalize=True) * 100).items()},
        "n_train": int(len(train_data)),
        "n_test": int(len(test_data)),
        "zero_counts": zero_counts,
        "correlations_with_outcome": correlations_dict,
        "describe": {col: {k: float(v) for k, v in diabetes_data[col].describe().items()} for col in diabetes_data.columns},
    },
    "cv_summary": cv_summary,
    "model_results": model_results_sorted,
    "ensemble_result": ensemble_result,
    "best_model_name": best_model_name,
    "threshold_comparison": threshold_comparison,
    "feature_importances": importances,
    "feature_columns": list(X_train.columns),
    "train_medians": {k: float(v) for k, v in train_medians.items()},
    "cost_assumption": {"fn_cost": FN_COST, "fp_cost": FP_COST},
    "pr_curve": {
        "thresholds": [float(t) for t in thresholds],
        "precisions": [float(p) for p in precisions[:-1]],
        "recalls": [float(r) for r in recalls[:-1]],
        "f1_scores": [float(f) for f in f1_scores[:-1]],
    },
}

with open("models/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Save final deployment artifacts (best model by Recall, cost-weighted threshold)
joblib.dump(best_model, "models/final_model.joblib")
joblib.dump(sc_X, "models/scaler.joblib")
joblib.dump(list(X_train.columns), "models/feature_columns.joblib")

print(f"Best model: {best_model_name}")
print(f"Cost-weighted threshold: {cost_optimal_threshold:.4f}")
print(f"Saved results.json with {len(results)} top-level keys")
print("Feature importances:", importances)
