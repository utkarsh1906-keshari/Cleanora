"""
DataDoctor AI - Smart Anomaly Detective Module (Feature 2)
Multi-method statistical, temporal, and machine learning anomaly detection
with industry-specific contextual logic and interactive visualizations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Gracefully handle optional sklearn if blocked by Windows Application Control DLL policies
HAS_SKLEARN = False
try:
    from sklearn.ensemble import IsolationForest
    HAS_SKLEARN = True
except (ImportError, Exception):
    HAS_SKLEARN = False


def detect_smart_anomalies(df, workspace="Retail Store", mapping=None):
    """
    Executes multi-layered anomaly detection across statistical distributions,
    temporal sequences, multidimensional feature spaces (Isolation Forest),
    and domain-specific operational rules.

    Returns:
        dict containing:
            - "anomalies": list of anomaly records
            - "total_anomalies": int
            - "anomaly_table": pd.DataFrame of findings
            - "high_severity_count": int
            - "featured_col": best numeric column for visual plot
    """
    if df is None or df.empty:
        return {
            "anomalies": [],
            "total_anomalies": 0,
            "anomaly_table": pd.DataFrame(),
            "high_severity_count": 0,
            "featured_col": None
        }

    anomalies = []
    mapping = mapping or {}
    total_rows = len(df)

    # 1. ISOLATION FOREST MULTIVARIATE OUTLIER DETECTION
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    # Filter out columns with all NaN or zero variance
    valid_num_cols = [c for c in numeric_cols if df[c].nunique() > 1 and df[c].notnull().sum() > 10]
    
    if len(valid_num_cols) >= 2 and total_rows >= 20:
        try:
            num_data = df[valid_num_cols].fillna(df[valid_num_cols].median())
            if HAS_SKLEARN:
                iso = IsolationForest(contamination=0.04, random_state=42)
                preds = iso.fit_predict(num_data)
                iso_indices = np.where(preds == -1)[0]
                method_name = "Isolation Forest (Ensemble Tree)"
            else:
                # Pure NumPy Robust Multi-Feature Distance Outlier Detector
                z_mat = np.abs((num_data - num_data.mean()) / (num_data.std() + 1e-6))
                dist = np.sqrt((z_mat ** 2).sum(axis=1))
                threshold = np.percentile(dist, 96) # Top 4% anomalies
                iso_indices = np.where(dist > threshold)[0]
                method_name = "Multivariate Normalized Distance (> 96th percentile)"
            
            if len(iso_indices) > 0:
                first_col = valid_num_cols[0]
                med_val = float(df[first_col].median())
                for idx in iso_indices[:5]: # Take top 5 sample records
                    act_val = float(df.iloc[idx][first_col])
                    anomalies.append({
                        "anomaly_type": "Multivariate Pattern Anomaly",
                        "column_or_category": f"Row #{idx+1} ({', '.join(valid_num_cols[:2])})",
                        "actual_value": f"{first_col}={round(act_val, 2)}",
                        "expected_value": f"Cluster Median: {round(med_val, 2)}",
                        "detection_method": method_name,
                        "severity": "High" if abs(act_val - med_val) > 2 * (df[first_col].std() or 1) else "Medium",
                        "explanation": f"Record #{idx+1} exhibits an abnormal joint combination of numerical features compared to normal cluster behavior.",
                        "row_idx": int(idx)
                    })
        except Exception:
            pass

    # 2. STATISTICAL OUTLIERS (Z-SCORE & IQR) ON KEY NUMERIC METRICS
    for col in valid_num_cols[:5]:
        series = df[col].dropna()
        if len(series) > 10:
            mean = series.mean()
            std = series.std()
            if std > 0:
                z_scores = (series - mean) / std
                extreme_z = series[z_scores.abs() > 3.0]
                for idx, val in extreme_z.head(3).items():
                    anomalies.append({
                        "anomaly_type": "Statistical Distribution Outlier",
                        "column_or_category": f"Column '{col}' (Row #{idx+1})",
                        "actual_value": str(round(float(val), 2)),
                        "expected_value": f"Mean {round(mean, 2)} ± {round(3*std, 2)}",
                        "detection_method": "Z-Score (> 3.0 Standard Deviations)",
                        "severity": "High" if abs(z_scores.loc[idx]) > 4.0 else "Medium",
                        "explanation": f"Value {round(float(val), 2)} deviates by {round(abs(z_scores.loc[idx]), 1)} standard deviations from the dataset mean.",
                        "row_idx": int(idx)
                    })

    # 3. TEMPORAL SUDDEN SPIKES / DROPS
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["date", "time", "day", "month"])]
    if date_cols and valid_num_cols:
        d_col = date_cols[0]
        v_col = valid_num_cols[0]
        try:
            temp_df = df.dropna(subset=[d_col, v_col]).copy()
            temp_df["_dt"] = pd.to_datetime(temp_df[d_col], errors="coerce")
            temp_df = temp_df.dropna(subset=["_dt"]).sort_values("_dt")
            if len(temp_df) >= 7:
                temp_df["_roll_mean"] = temp_df[v_col].rolling(window=5, min_periods=2).mean()
                temp_df["_pct_change"] = ((temp_df[v_col] - temp_df["_roll_mean"]) / (temp_df["_roll_mean"] + 1e-5) * 100).abs()
                spikes = temp_df[temp_df["_pct_change"] > 150.0]
                for idx, row in spikes.head(3).iterrows():
                    anomalies.append({
                        "anomaly_type": "Abrupt Temporal Velocity Spike",
                        "column_or_category": f"'{v_col}' on {row['_dt'].strftime('%Y-%m-%d')}",
                        "actual_value": str(round(float(row[v_col]), 2)),
                        "expected_value": f"5-Period Rolling Mean: {round(float(row['_roll_mean']), 2)}",
                        "detection_method": "Rolling Average Velocity Deviation (>150%)",
                        "severity": "High",
                        "explanation": f"Abrupt spike of {round(float(row['_pct_change']), 1)}% above recent rolling average.",
                        "row_idx": int(idx)
                    })
        except Exception:
            pass

    # 4. INDUSTRY-SPECIFIC ANOMALIES
    if workspace == "Retail Store":
        # Price vs Quantity inverse check / Revenue zero with stock
        if "revenue" in mapping and "quantity" in mapping:
            r_col, q_col = mapping["revenue"], mapping["quantity"]
            if r_col in df.columns and q_col in df.columns:
                zero_rev_high_qty = df[(pd.to_numeric(df[r_col], errors="coerce") == 0) & (pd.to_numeric(df[q_col], errors="coerce") > 5)]
                for idx in zero_rev_high_qty.head(3).index:
                    anomalies.append({
                        "anomaly_type": "Zero-Revenue High Volume Sale",
                        "column_or_category": f"Retail Order (Row #{idx+1})",
                        "actual_value": f"Qty: {df.at[idx, q_col]}, Revenue: $0.00",
                        "expected_value": "Revenue > $0 for positive units sold",
                        "detection_method": "Commercial Logic Consistency Rule",
                        "severity": "High",
                        "explanation": "Transactions recorded units sold but registered exactly $0.00 in revenue (possible unrecorded promo/leakage).",
                        "row_idx": int(idx)
                    })

    elif workspace == "Healthcare":
        # Out of bounds age or stay length
        age_cols = [c for c in df.columns if "age" in c.lower()]
        if age_cols:
            a_col = age_cols[0]
            invalid_age = df[(pd.to_numeric(df[a_col], errors="coerce") < 0) | (pd.to_numeric(df[a_col], errors="coerce") > 115)]
            for idx in invalid_age.head(3).index:
                val = df.at[idx, a_col]
                anomalies.append({
                    "anomaly_type": "Physiological Age Anomaly",
                    "column_or_category": f"Patient Record (Row #{idx+1})",
                    "actual_value": f"Age: {val}",
                    "expected_value": "Valid Range [0, 115 years]",
                    "detection_method": "Clinical Boundary Guardrail",
                    "severity": "High",
                    "explanation": f"Patient record has biological age {val}, which is impossible or clinically abnormal.",
                    "row_idx": int(idx)
                })

    elif workspace == "Education":
        # Extreme marks or zero attendance with high marks
        marks_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["marks", "score", "grade"])]
        if marks_cols:
            m_col = marks_cols[0]
            extreme_m = df[(pd.to_numeric(df[m_col], errors="coerce") < 0) | (pd.to_numeric(df[m_col], errors="coerce") > 100)]
            for idx in extreme_m.head(3).index:
                val = df.at[idx, m_col]
                anomalies.append({
                    "anomaly_type": "Academic Score Out-of-Bounds",
                    "column_or_category": f"Student Evaluation (Row #{idx+1})",
                    "actual_value": f"Score: {val}",
                    "expected_value": "Standard Grading Scale [0, 100]",
                    "detection_method": "Curricular Bounding Constraint",
                    "severity": "High",
                    "explanation": f"Evaluation score {val} lies outside standard curriculum scoring boundaries.",
                    "row_idx": int(idx)
                })

    elif workspace == "Business & Finance":
        # Negative expenses or massive expense spikes
        rev_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["rev", "sale", "income"])]
        exp_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["exp", "cost", "spend"])]
        
        for c in rev_cols + exp_cols:
            neg_vals = df[pd.to_numeric(df[c], errors="coerce") < 0]
            for idx in neg_vals.head(3).index:
                val = df.at[idx, c]
                anomalies.append({
                    "anomaly_type": "Negative Financial Ledger Entry",
                    "column_or_category": f"Accounting Ledger '{c}' (Row #{idx+1})",
                    "actual_value": f"${round(float(val), 2)}",
                    "expected_value": "Financial Values >= $0.00",
                    "detection_method": "GAAP / Accrual Accounting Sanity Rule",
                    "severity": "High",
                    "explanation": f"Ledger entry recorded negative figure (${round(float(val), 2)}). Credit notes must be reconciled.",
                    "row_idx": int(idx)
                })

    table_df = pd.DataFrame(anomalies) if anomalies else pd.DataFrame(columns=[
        "anomaly_type", "column_or_category", "actual_value", "expected_value", "detection_method", "severity", "explanation"
    ])

    high_count = sum(1 for a in anomalies if a["severity"] == "High")
    featured = valid_num_cols[0] if valid_num_cols else None

    return {
        "anomalies": anomalies,
        "total_anomalies": len(anomalies),
        "anomaly_table": table_df,
        "high_severity_count": high_count,
        "featured_col": featured
    }


def create_anomaly_visual_chart(df, featured_col, anomalies):
    """
    Constructs an interactive Plotly scatter chart plotting all records
    and highlighting detected anomaly indices with pulsing red markers.
    """
    if df is None or df.empty or not featured_col or featured_col not in df.columns:
        return None

    series = pd.to_numeric(df[featured_col], errors="coerce").fillna(0)
    anom_indices = set(a["row_idx"] for a in anomalies if "row_idx" in a and a["row_idx"] < len(df))

    is_anomaly = [idx in anom_indices for idx in range(len(df))]
    
    plot_df = pd.DataFrame({
        "Index": range(1, len(df) + 1),
        "Value": series.values,
        "Classification": ["Detected Anomaly" if a else "Normal Pattern" for a in is_anomaly]
    })

    fig = px.scatter(
        plot_df,
        x="Index",
        y="Value",
        color="Classification",
        color_discrete_map={"Normal Pattern": "#38BDF8", "Detected Anomaly": "#F87171"},
        title=f"Anomaly Distribution Map across '{featured_col}'",
        labels={"Index": "Record Index", "Value": featured_col}
    )

    # Style anomalies with larger red diamond symbols
    fig.update_traces(
        marker=dict(size=7, opacity=0.75),
        selector=dict(name="Normal Pattern")
    )
    fig.update_traces(
        marker=dict(size=12, symbol="diamond", line=dict(width=2, color="#FFFFFF")),
        selector=dict(name="Detected Anomaly")
    )

    fig.update_layout(
        paper_bgcolor="#0B1120",
        plot_bgcolor="#16243A",
        font_color="#E2E8F0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig
