"""
train_model.py
XGBoost pipeline to predict infrastructure project cost overruns.
Outputs: model metrics, feature importances, predictions CSV for dashboard.
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score
)
from xgboost import XGBClassifier

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/projects.csv")

FEATURES = [
    "project_type", "region", "procurement_method", "complexity",
    "budget_million_usd", "planned_duration_months", "num_contractors",
    "change_orders", "soil_risk", "weather_delays_days",
    "design_complete_pct", "owner_experience_years", "prior_project_delays"
]
TARGET = "cost_overrun"

X = df[FEATURES].copy()
y = df[TARGET]

# ── Encode categoricals ────────────────────────────────────────────────────────
cat_cols = ["project_type", "region", "procurement_method", "complexity"]
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# ── Train / test split ─────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Model ──────────────────────────────────────────────────────────────────────
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# ── Evaluation ────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

print("=" * 50)
print(f"  Accuracy : {acc:.1%}")
print(f"  ROC-AUC  : {auc:.3f}")
print("=" * 50)
print(classification_report(y_test, y_pred, target_names=["No Overrun", "Overrun"]))

# ── Cross-validation ──────────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
print(f"  5-Fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

# ── Feature importances ───────────────────────────────────────────────────────
importance_df = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

readable_names = {
    "change_orders": "Change Orders",
    "design_complete_pct": "Design Completion %",
    "complexity": "Project Complexity",
    "procurement_method": "Procurement Method",
    "soil_risk": "Soil / Ground Risk",
    "budget_million_usd": "Project Budget ($M)",
    "prior_project_delays": "Prior Delay History",
    "weather_delays_days": "Weather Delays",
    "planned_duration_months": "Planned Duration",
    "owner_experience_years": "Owner Experience",
    "num_contractors": "Number of Contractors",
    "project_type": "Project Type",
    "region": "Region"
}
importance_df["label"] = importance_df["feature"].map(readable_names)
print("\nTop Risk Factors:")
print(importance_df[["label", "importance"]].to_string(index=False))

# ── Save outputs for dashboard ────────────────────────────────────────────────
# Predictions on full dataset
df["overrun_probability"] = model.predict_proba(X)[:, 1]
df["predicted_overrun"] = model.predict(X)

# Risk tier
df["risk_tier"] = pd.cut(
    df["overrun_probability"],
    bins=[0, 0.4, 0.65, 1.0],
    labels=["Low", "Medium", "High"]
)

df.to_csv("data/predictions.csv", index=False)

# Save metrics JSON for dashboard
metrics = {
    "accuracy": round(acc, 4),
    "roc_auc": round(auc, 4),
    "cv_auc_mean": round(cv_scores.mean(), 4),
    "cv_auc_std": round(cv_scores.std(), 4),
    "total_projects": len(df),
    "overrun_rate": round(df["cost_overrun"].mean(), 4),
    "high_risk_count": int((df["risk_tier"] == "High").sum()),
    "confusion_matrix": cm.tolist(),
    "feature_importances": [
        {"feature": row["label"], "importance": round(float(row["importance"]), 4)}
        for _, row in importance_df.iterrows()
    ]
}

with open("data/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n✓ Saved: data/predictions.csv, data/metrics.json")
