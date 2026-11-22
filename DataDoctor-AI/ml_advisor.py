"""
DataDoctor AI - Machine Learning Readiness Advisor Module (Feature 7)
Advanced feature quality diagnostics, candidate target identification,
class imbalance auditing, leakage detection, preprocessing checklist,
and model family recommendations.
"""

import pandas as pd
import numpy as np


def evaluate_ml_readiness(df, selected_target=None, task_type="Auto-Detect"):
    """
    Evaluates ML feasibility and readiness for training machine learning models.
    
    Returns:
        dict containing:
            - "readiness_score": int 0-100
            - "task_type": "Classification" | "Regression" | "Clustering"
            - "target_col": candidate or selected target column
            - "checklist": list of preparation items with status (Pass/Warning/Fail)
            - "model_recommendations": list of suitable algorithms with explanations
            - "feature_breakdown": dict of column types
            - "leakage_warnings": list of potential data leakage indicators
            - "imbalance_info": dict of class balance metrics if classification
            - "limitations": str explanation of limitations
    """
    if df is None or df.empty:
        return {
            "readiness_score": 0,
            "task_type": "None",
            "target_col": None,
            "checklist": [],
            "model_recommendations": [],
            "feature_breakdown": {},
            "leakage_warnings": [],
            "imbalance_info": None,
            "limitations": "No dataset loaded for evaluation."
        }

    total_rows = len(df)
    total_cols = len(df.columns)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["date", "time", "day", "month", "year"])]

    # 1. AUTO-DETECT OR VALIDATE TARGET COLUMN
    target_col = selected_target
    if not target_col or target_col not in df.columns:
        # Heuristic search for probable targets
        candidate_names = [
            "target", "label", "revenue", "profit", "marks", "score",
            "status", "is_passed", "diagnosis", "outcome", "churn",
            "actual", "grade", "admitted", "completed"
        ]
        for cn in candidate_names:
            matched = [c for c in df.columns if cn in c.lower()]
            if matched:
                target_col = matched[0]
                break
        if not target_col and len(df.columns) > 1:
            target_col = df.columns[-1]

    # 2. DETERMINE TASK TYPE
    if task_type == "Auto-Detect":
        if target_col and target_col in df.columns:
            target_series = df[target_col].dropna()
            if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 10:
                inferred_task = "Regression"
            elif target_series.nunique() <= 20:
                inferred_task = "Classification"
            else:
                inferred_task = "Clustering"
        else:
            inferred_task = "Clustering"
    else:
        inferred_task = task_type

    # 3. AUDIT DIMENSIONS & SCORING
    score = 100
    checklist = []
    leakage_warnings = []
    imbalance_info = None

    # Check 1: Sample Size Adequacy
    if total_rows < 50:
        score -= 25
        checklist.append({
            "dimension": "Sample Size",
            "status": "Fail",
            "detail": f"Dataset has only {total_rows} records. Models trained on < 50 samples suffer from high variance and severe overfitting.",
            "action": "Acquire at least 200+ observations before training predictive models."
        })
    elif total_rows < 200:
        score -= 10
        checklist.append({
            "dimension": "Sample Size",
            "status": "Warning",
            "detail": f"Moderate dataset size ({total_rows} rows). Sufficient for simple linear/tree models, but may underfit complex ensembles.",
            "action": "Use k-fold cross-validation (5-10 folds) to prevent optimistic validation bias."
        })
    else:
        checklist.append({
            "dimension": "Sample Size",
            "status": "Pass",
            "detail": f"Sample volume is healthy ({total_rows:,} rows) for tabular statistical and machine learning estimators.",
            "action": "Standard 80/20 train/test split recommended."
        })

    # Check 2: Missing Value Burden
    total_nulls = int(df.isnull().sum().sum())
    null_ratio = total_nulls / max(1, total_rows * total_cols)
    if null_ratio > 0.20:
        score -= 20
        checklist.append({
            "dimension": "Missing Data",
            "status": "Fail",
            "detail": f"{round(null_ratio*100, 1)}% of values in the dataset are missing. Scikit-learn estimators cannot accept null entries directly.",
            "action": "Run AI Data Doctor median/mode imputation or drop columns with >60% missing rate."
        })
    elif null_ratio > 0.05:
        score -= 10
        checklist.append({
            "dimension": "Missing Data",
            "status": "Warning",
            "detail": f"{round(null_ratio*100, 1)}% missing values detected. Preprocessing imputation pipeline is mandatory.",
            "action": "Apply SimpleImputer(strategy='median') for numeric and SimpleImputer(strategy='most_frequent') for categorical."
        })
    else:
        checklist.append({
            "dimension": "Missing Data",
            "status": "Pass",
            "detail": f"Low missing value rate ({round(null_ratio*100, 2)}%). Missingness is minimal.",
            "action": "Minor median imputation prior to model fit."
        })

    # Check 3: Categorical Feature Cardinality
    high_card_cols = []
    for c in cat_cols:
        uniq = df[c].nunique()
        if uniq > 50 and uniq > 0.3 * total_rows:
            high_card_cols.append(c)
            
    if high_card_cols:
        score -= 15
        checklist.append({
            "dimension": "Categorical Encoding",
            "status": "Warning",
            "detail": f"High cardinality detected in {len(high_card_cols)} column(s): {', '.join(high_card_cols[:3])}. One-hot encoding will cause extreme dimensional explosion.",
            "action": "Use Target Encoding, Frequency Encoding, or exclude unique identifier strings."
        })
    else:
        checklist.append({
            "dimension": "Categorical Encoding",
            "status": "Pass",
            "detail": f"{len(cat_cols)} categorical feature(s) detected with manageable cardinality.",
            "action": "One-Hot Encoding or Ordinal Encoding is suitable."
        })

    # Check 4: Constant or Near-Constant Features
    constant_cols = [c for c in df.columns if df[c].nunique(dropna=False) == 1]
    if constant_cols:
        score -= 10
        checklist.append({
            "dimension": "Feature Variance",
            "status": "Warning",
            "detail": f"Found {len(constant_cols)} zero-variance column(s): {', '.join(constant_cols)}. They contain no predictive signal.",
            "action": "Drop zero-variance columns using VarianceThreshold(0.0)."
        })
    else:
        checklist.append({
            "dimension": "Feature Variance",
            "status": "Pass",
            "detail": "All feature columns exhibit informative variance across samples.",
            "action": "No variance filtering needed."
        })

    # Check 5: Class Imbalance (Classification)
    if inferred_task == "Classification" and target_col in df.columns:
        counts = df[target_col].value_counts()
        if len(counts) >= 2:
            min_c = counts.min()
            max_c = counts.max()
            ratio = round(max_c / max(1, min_c), 1)
            imbalance_info = {
                "classes": dict(zip(counts.index.astype(str), counts.values.tolist())),
                "imbalance_ratio": ratio,
                "is_imbalanced": ratio > 3.0
            }
            if ratio > 5.0:
                score -= 15
                checklist.append({
                    "dimension": "Class Balance",
                    "status": "Warning",
                    "detail": f"Severe class imbalance detected ({ratio}:1 ratio between majority and minority class). Standard accuracy metrics will be misleading.",
                    "action": "Use Balanced Class Weights, SMOTE resampling, and evaluate with PR-AUC / F1-Score instead of Accuracy."
                })
            else:
                checklist.append({
                    "dimension": "Class Balance",
                    "status": "Pass",
                    "detail": f"Class distribution is reasonably balanced ({ratio}:1 ratio).",
                    "action": "Standard cross-validation and ROC-AUC evaluation are suitable."
                })

    # Check 6: Data Leakage Detection
    id_like = [c for c in df.columns if any(kw in c.lower() for kw in ["id", "uuid", "roll_no", "txnid", "ssn", "patientid", "studentid"])]
    if id_like:
        leakage_warnings.append(
            f"Identifier columns detected ({', '.join(id_like)}). Including row/record IDs in model training causes memorization leakage. Exclude them from feature set X."
        )

    # 4. MODEL FAMILY RECOMMENDATIONS
    model_recs = []
    if inferred_task == "Classification":
        model_recs = [
            {
                "model": "Gradient Boosted Trees (XGBoost / LightGBM / HistGradientBoosting)",
                "suitability": "Top Recommendation",
                "why": "Robust to non-linear interactions, handles mixed numeric/categorical features natively, and robust to outliers."
            },
            {
                "model": "Random Forest Classifier",
                "suitability": "Strong Baseline",
                "why": "Ensemble bagging prevents overfitting on small-to-medium datasets without intensive hyperparameter tuning."
            },
            {
                "model": "Logistic Regression (with L2 Regularization)",
                "suitability": "Interpretable Benchmark",
                "why": "Provides transparent coefficient odds-ratios and fast inference, provided numeric features are standard scaled."
            }
        ]
    elif inferred_task == "Regression":
        model_recs = [
            {
                "model": "Gradient Boosted Regressor (LightGBM / XGBoost)",
                "suitability": "Top Recommendation",
                "why": "Captures non-linear price, revenue, or score curves without requiring strict normality assumptions."
            },
            {
                "model": "Random Forest Regressor",
                "suitability": "Strong Baseline",
                "why": "Resilient to extreme financial and performance outliers; outputs reliable feature importances."
            },
            {
                "model": "Ridge / ElasticNet Regressor",
                "suitability": "Linear Benchmark",
                "why": "Excellent when sample size is constrained, applying L1/L2 shrinkage to handle multicollinearity."
            }
        ]
    else: # Clustering
        model_recs = [
            {
                "model": "K-Means Clustering (with StandardScaler)",
                "suitability": "Top Recommendation",
                "why": "Partitions customers, patients, or students into distinct behavioral cohorts based on numeric distance."
            },
            {
                "model": "DBSCAN (Density-Based Clustering)",
                "suitability": "Outlier-Aware",
                "why": "Identifies arbitrary cluster shapes while automatically labeling noise and irregular records as outliers."
            }
        ]

    final_score = max(10, min(100, score))

    limitations = (
        "An ML Readiness Score reflects structural data hygiene (sample volume, completeness, balance, and encoding feasibility). "
        "A high score does NOT guarantee high predictive accuracy; domain feature engineering, cross-validation tuning, "
        "and validation against production drift remain required."
    )

    return {
        "readiness_score": final_score,
        "task_type": inferred_task,
        "target_col": target_col,
        "checklist": checklist,
        "model_recommendations": model_recs,
        "feature_breakdown": {
            "Numeric Features": len(numeric_cols),
            "Categorical Features": len(cat_cols),
            "Date/Time Features": len(date_cols)
        },
        "leakage_warnings": leakage_warnings,
        "imbalance_info": imbalance_info,
        "limitations": limitations
    }
