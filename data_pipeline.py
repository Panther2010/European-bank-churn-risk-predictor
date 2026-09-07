"""
data_pipeline.py
-----------------
Loading, cleaning, segmentation, KPI computation, and churn-prediction
model training for the European Bank Customer Segmentation & Churn
Pattern Analytics dashboard.

Kept separate from app.py so the Streamlit UI stays focused on layout,
and so this logic can be unit-tested or reused (e.g. in the research
paper notebook) independently of the dashboard.
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score, roc_auc_score,
                              roc_curve)
from xgboost import XGBClassifier

NUMERIC_FEATURES = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts", "EstimatedSalary"]
CATEGORICAL_FEATURES = ["Geography", "Gender", "HasCrCard", "IsActiveMember"]

AGE_BINS = [0, 30, 45, 60, 150]
AGE_LABELS = ["<30", "30-45", "46-60", "60+"]

CREDIT_BINS = [0, 580, 700, 1000]
CREDIT_LABELS = ["Low (<580)", "Medium (580-700)", "High (>700)"]


# ---------------------------------------------------------------------
# Loading & cleaning
# ---------------------------------------------------------------------

def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def tenure_group(t: int) -> str:
    if t <= 2:
        return "New (0-2 yrs)"
    elif t <= 6:
        return "Mid-term (3-6 yrs)"
    return "Long-term (7+ yrs)"


def balance_segment(b: float) -> str:
    if b == 0:
        return "Zero-balance"
    elif b < 100_000:
        return "Low-balance (<100K)"
    return "High-balance (>=100K)"


def clean_and_segment(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the cleaning/segmentation logic used in the analysis notebook."""
    out = df.copy()
    for col in ("Surname", "Year"):
        if col in out.columns:
            out = out.drop(columns=[col])

    out["Gender"] = out["Gender"].astype("category")
    out["Geography"] = out["Geography"].astype("category")
    out["HasCrCard_Label"] = out["HasCrCard"].map({1: "Has Card", 0: "No Card"})
    out["IsActiveMember_Label"] = out["IsActiveMember"].map({1: "Active", 0: "Inactive"})
    out["Exited_Label"] = out["Exited"].map({1: "Churned", 0: "Retained"})

    out["AgeGroup"] = pd.cut(out["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True)
    out["CreditScoreBand"] = pd.cut(out["CreditScore"], bins=CREDIT_BINS, labels=CREDIT_LABELS)

    out["TenureGroup"] = pd.Categorical(
        out["Tenure"].apply(tenure_group),
        categories=["New (0-2 yrs)", "Mid-term (3-6 yrs)", "Long-term (7+ yrs)"],
        ordered=True,
    )
    out["BalanceSegment"] = pd.Categorical(
        out["Balance"].apply(balance_segment),
        categories=["Zero-balance", "Low-balance (<100K)", "High-balance (>=100K)"],
        ordered=True,
    )

    balance_q75 = out["Balance"].quantile(0.75)
    salary_q75 = out["EstimatedSalary"].quantile(0.75)
    out["HighValue"] = (out["Balance"] >= balance_q75) | (out["EstimatedSalary"] >= salary_q75)

    return out


# ---------------------------------------------------------------------
# KPIs
# ---------------------------------------------------------------------

def segment_churn_table(df: pd.DataFrame, col: str) -> pd.DataFrame:
    tbl = df.groupby(col, observed=True).agg(
        Customers=("Exited", "count"),
        Churned=("Exited", "sum"),
        ChurnRate=("Exited", "mean"),
    )
    tbl["ShareOfBase"] = tbl["Customers"] / len(df)
    total_churners = df["Exited"].sum()
    tbl["ShareOfChurners"] = tbl["Churned"] / total_churners if total_churners else 0
    return tbl


def compute_kpis(df: pd.DataFrame) -> dict:
    overall_rate = df["Exited"].mean() if len(df) else np.nan
    active_rate = df.loc[df["IsActiveMember"] == 1, "Exited"].mean() if (df["IsActiveMember"] == 1).any() else np.nan
    inactive_rate = df.loc[df["IsActiveMember"] == 0, "Exited"].mean() if (df["IsActiveMember"] == 0).any() else np.nan
    hv_mask = df["HighValue"] if "HighValue" in df.columns else pd.Series(False, index=df.index)
    total_churners = df["Exited"].sum()
    hv_churn_ratio = (
        df.loc[hv_mask & (df["Exited"] == 1)].shape[0] / total_churners if total_churners else np.nan
    )
    geo_index = (segment_churn_table(df, "Geography")["ChurnRate"] / overall_rate).round(2) if len(df) else pd.Series(dtype=float)

    total_balance = df["Balance"].sum()
    churned_balance = df.loc[df["Exited"] == 1, "Balance"].sum()

    return {
        "overall_churn_rate": overall_rate,
        "n_customers": len(df),
        "n_churned": int(total_churners) if not np.isnan(total_churners) else 0,
        "active_churn_rate": active_rate,
        "inactive_churn_rate": inactive_rate,
        "engagement_drop_pp": (inactive_rate - active_rate) * 100 if pd.notna(active_rate) and pd.notna(inactive_rate) else np.nan,
        "high_value_churn_ratio": hv_churn_ratio,
        "geographic_risk_index": geo_index,
        "total_balance": total_balance,
        "churned_balance": churned_balance,
        "churned_balance_share": churned_balance / total_balance if total_balance else np.nan,
    }


# ---------------------------------------------------------------------
# Model training (tuned XGBoost — winner from the analysis notebook)
# ---------------------------------------------------------------------

def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def train_model(df: pd.DataFrame, random_state: int = 42):
    """Trains the tuned XGBoost churn classifier (best model from the analysis
    notebook's hyperparameter search) and returns the fitted pipeline plus
    test-set diagnostics for the Model Insights tab."""
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df["Exited"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    pipeline = Pipeline(steps=[
        ("preprocess", build_preprocessor()),
        ("model", XGBClassifier(
            n_estimators=180, max_depth=4, learning_rate=0.048,
            subsample=0.88, colsample_bytree=0.91, min_child_weight=3,
            scale_pos_weight=scale_pos_weight, eval_metric="logloss",
            random_state=random_state, n_jobs=-1,
        )),
    ])
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_proba),
    }
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_proba)

    feature_names = (
        NUMERIC_FEATURES
        + list(pipeline.named_steps["preprocess"].named_transformers_["cat"]
               .get_feature_names_out(CATEGORICAL_FEATURES))
    )
    importances = pipeline.named_steps["model"].feature_importances_
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(
        "Importance", ascending=False
    ).reset_index(drop=True)

    diagnostics = {
        "metrics": metrics,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "feature_importance": fi_df,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    return pipeline, diagnostics


def score_customers(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    out = df.copy()
    out["ChurnProbability"] = pipeline.predict_proba(X)[:, 1]
    out["ChurnRiskTier"] = pd.cut(
        out["ChurnProbability"], bins=[-0.01, 0.4, 0.7, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"],
    )
    return out


def predict_single(pipeline: Pipeline, inputs: dict) -> float:
    """Scores a single hypothetical customer for the What-If predictor tab."""
    row = pd.DataFrame([inputs])[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    return float(pipeline.predict_proba(row)[:, 1][0])
