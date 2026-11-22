"""
DataDoctor AI - AI Data Doctor Module (Feature 1)
Intelligent dataset diagnosis, anomaly reasoning, actionable recommendations,
and interactive fix preview engine.
"""

import pandas as pd
import numpy as np


def diagnose_dataset(df, workspace="General", mapping=None):
    """
    Performs comprehensive diagnostic inspection of the dataset.
    Detects missing values, duplicates, statistical outliers, inconsistent dates,
    invalid data types, and suspicious columns.
    
    Returns:
        dict containing:
            - "issues": list of structured issue dictionaries
            - "issue_count": int total issues
            - "high_severity_count": int count of High severity issues
            - "health_impact": int estimated score deduction
            - "summary": string overview
    """
    if df is None or df.empty:
        return {
            "issues": [],
            "issue_count": 0,
            "high_severity_count": 0,
            "health_impact": 0,
            "summary": "No dataset provided for diagnosis."
        }

    issues = []
    total_records = len(df)
    mapping = mapping or {}

    # 1. DUPLICATE ROWS
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = round(dup_count / total_records * 100, 1)
        if workspace == "Healthcare":
            dup_impact = "Duplicate rows artificially inflate clinical metrics (like patient encounters, appointment counts, or hospital admissions)"
        elif workspace == "Retail Store":
            dup_impact = "Duplicate rows artificially inflate commercial metrics (like gross sales revenue, units sold, or order transactions)"
        elif workspace == "Education":
            dup_impact = "Duplicate rows artificially inflate academic metrics (like student enrollments, exam records, or attendance compliance)"
        elif workspace in ["Business & Finance", "Finance"]:
            dup_impact = "Duplicate rows artificially inflate financial metrics (like revenue, operating expenses, or transaction totals)"
        elif workspace in ["Employee & HR", "HR", "Employees"]:
            dup_impact = "Duplicate rows artificially inflate workforce metrics (like employee headcounts, payroll calculations, or staff counts)"
        else:
            dup_impact = "Duplicate rows artificially inflate record counts and introduce severe estimation bias in models"

        issues.append({
            "id": "fix_duplicate_rows",
            "problem_name": "Redundant Duplicate Records",
            "column": "Entire Row",
            "affected_records": dup_count,
            "severity": "High" if dup_pct > 5 else "Medium",
            "why_it_matters": (
                f"Found {dup_count} identical records ({dup_pct}% of dataset). "
                f"{dup_impact} and introduce severe estimation bias."
            ),
            "suggested_fix": "Deduplicate Dataset",
            "expected_impact": f"Removes {dup_count} redundant rows, restoring accurate record counts.",
            "fix_type": "remove_duplicates",
            "target_col": None
        })

    # 2. MISSING VALUES PER COLUMN
    for col in df.columns:
        null_count = int(df[col].isnull().sum())
        if null_count > 0:
            null_pct = round(null_count / total_records * 100, 1)
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            
            if null_pct > 60:
                severity = "High"
                fix_desc = "Drop Column or Flag as Incomplete"
                fix_type = "drop_column"
                why = (
                    f"Column '{col}' has {null_pct}% missing values. "
                    "Columns with more than 60% missing data are rarely reliable for statistical decision-making."
                )
            elif is_numeric:
                severity = "High" if null_pct > 15 else ("Medium" if null_pct > 3 else "Low")
                fix_desc = "Median Imputation"
                fix_type = "impute_median"
                why = (
                    f"Column '{col}' is missing {null_count} numeric values ({null_pct}%). "
                    "Null numeric values cause aggregations, sums, and machine learning estimators to fail or omit samples."
                )
            else:
                severity = "Medium" if null_pct > 15 else "Low"
                fix_desc = "Mode (Most Frequent) or 'Unknown' Tag"
                fix_type = "impute_mode"
                why = (
                    f"Categorical column '{col}' has {null_count} unrecorded entries ({null_pct}%). "
                    "Missing categories obscure demographic or cost-center breakdowns."
                )

            issues.append({
                "id": f"fix_missing_{col}",
                "problem_name": f"Missing Values in '{col}'",
                "column": col,
                "affected_records": null_count,
                "severity": severity,
                "why_it_matters": why,
                "suggested_fix": fix_desc,
                "expected_impact": f"Restores {null_count} records to usable state without dropping observation rows.",
                "fix_type": fix_type,
                "target_col": col
            })

    # 3. NUMERIC STORED AS STRING / INVALID DATA TYPES
    for col in df.select_dtypes(include=["object", "string"]).columns:
        # Check if values look like numbers with $, commas, or %
        sample_vals = df[col].dropna().astype(str).str.strip()
        if len(sample_vals) > 0:
            cleaned = sample_vals.str.replace("$", "", regex=False).str.replace(",", "", regex=False).str.replace("%", "", regex=False)
            numeric_test = pd.to_numeric(cleaned, errors="coerce")
            valid_num_ratio = numeric_test.notnull().mean()
            if valid_num_ratio > 0.85 and not pd.api.types.is_numeric_dtype(df[col]):
                issues.append({
                    "id": f"fix_type_{col}",
                    "problem_name": f"Numeric Field Stored as Text in '{col}'",
                    "column": col,
                    "affected_records": int(numeric_test.notnull().sum()),
                    "severity": "High",
                    "why_it_matters": (
                        f"Column '{col}' appears to contain financial or numerical figures formatted as strings "
                        "(e.g. '$1,200.00' or '85%'). String formatting prevents mathematical summation, averages, and charts."
                    ),
                    "suggested_fix": "Type Conversion to Float",
                    "expected_impact": "Strips currency/percentage symbols and converts column into native numeric float.",
                    "fix_type": "clean_numeric_type",
                    "target_col": col
                })

    # 4. STATISTICAL OUTLIERS (IQR METHOD)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 10:
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers_mask = (series < lower_bound) | (series > upper_bound)
                outlier_count = int(outliers_mask.sum())
                if outlier_count > 0:
                    outlier_pct = round(outlier_count / len(series) * 100, 1)
                    if outlier_pct > 1.0: # Ignore tiny 1-off noise
                        issues.append({
                            "id": f"fix_outlier_{col}",
                            "problem_name": f"Extreme Statistical Outliers in '{col}'",
                            "column": col,
                            "affected_records": outlier_count,
                            "severity": "Medium" if outlier_pct < 10 else "High",
                            "why_it_matters": (
                                f"Detected {outlier_count} values outside normal range [{round(lower_bound, 2)}, {round(upper_bound, 2)}]. "
                                "Extreme outliers heavily distort mean averages, budget forecasting, and statistical modeling."
                            ),
                            "suggested_fix": "Winsorization / Outlier Capping (99th/1st percentile)",
                            "expected_impact": "Caps extreme spikes at the 1st and 99th percentiles while preserving data volume.",
                            "fix_type": "cap_outliers",
                            "target_col": col
                        })

    # 5. SUSPICIOUS CONSTANT OR NEAR-CONSTANT COLUMNS
    for col in df.columns:
        unique_cnt = df[col].nunique(dropna=False)
        if unique_cnt == 1:
            issues.append({
                "id": f"fix_constant_{col}",
                "problem_name": f"Zero-Variance Constant Column '{col}'",
                "column": col,
                "affected_records": total_records,
                "severity": "Low",
                "why_it_matters": (
                    f"Column '{col}' has identical value for all {total_records} rows. "
                    "Zero-variance features provide zero informational value for analytics or predictive modeling."
                ),
                "suggested_fix": "Drop Redundant Feature",
                "expected_impact": "Eliminates redundant memory overhead.",
                "fix_type": "drop_column",
                "target_col": col
            })

    # 6. INCONSISTENT DATES
    date_cols = [c for c in df.columns if any(kw in c.lower() for kw in ["date", "time", "admit", "disch", "exam", "visit"])]
    if len(date_cols) >= 2:
        # Check start vs end dates (e.g. admission vs discharge)
        col1, col2 = date_cols[0], date_cols[1]
        try:
            d1 = pd.to_datetime(df[col1], errors="coerce")
            d2 = pd.to_datetime(df[col2], errors="coerce")
            inverted = (d1 > d2) & d1.notnull() & d2.notnull()
            inv_count = int(inverted.sum())
            if inv_count > 0:
                if workspace == "Healthcare":
                    date_impact = "Causes negative hospital durations, skewed length of stay (ALOS), or invalid clinical timeline reporting."
                elif workspace == "Retail Store":
                    date_impact = "Causes negative fulfillment durations or invalid order-to-delivery transaction timeline reporting."
                elif workspace == "Education":
                    date_impact = "Causes inverted academic term dates or invalid semester timeline reporting."
                elif workspace in ["Business & Finance", "Finance"]:
                    date_impact = "Causes negative payment cycles, skewed invoice-to-cash intervals, or invalid fiscal period reporting."
                elif workspace in ["Employee & HR", "HR", "Employees"]:
                    date_impact = "Leaving date occurs before joining date, creating negative employee tenure and invalid employment history."
                else:
                    date_impact = "Causes negative event durations or invalid timeline reporting."

                issues.append({
                    "id": f"fix_dates_{col1}_{col2}",
                    "problem_name": f"Inverted Chronological Dates ({col1} > {col2})",
                    "column": f"{col1} & {col2}",
                    "affected_records": inv_count,
                    "severity": "High",
                    "why_it_matters": (
                        f"Found {inv_count} records where '{col1}' occurs chronologically after '{col2}'. "
                        f"{date_impact}"
                    ),
                    "suggested_fix": "Chronological Transposition",
                    "expected_impact": "Swaps inverted start and end dates so all durations are strictly positive.",
                    "fix_type": "fix_dates",
                    "target_col": f"{col1}___{col2}"
                })
        except Exception:
            pass

    high_count = sum(1 for iss in issues if iss["severity"] == "High")
    
    return {
        "issues": issues,
        "issue_count": len(issues),
        "high_severity_count": high_count,
        "health_impact": min(60, len(issues) * 6 + high_count * 8),
        "summary": f"AI Data Doctor identified {len(issues)} data quality items ({high_count} critical severity) requiring attention."
    }


def preview_fix(df, fix_type, target_col=None):
    """
    Generates an interactive before-and-after preview of applying a specific fix
    WITHOUT modifying the original DataFrame.
    
    Returns:
        dict with:
            - "before_count": affected rows count before
            - "after_count": affected rows count after
            - "before_sample": DataFrame slice showing anomalies
            - "after_sample": DataFrame slice showing proposed corrected rows
            - "explanation": description of transformation
    """
    if df is None or df.empty:
        return None

    sample_df = df.copy()

    if fix_type == "remove_duplicates":
        dups = sample_df[sample_df.duplicated(keep=False)]
        before_cnt = len(dups)
        cleaned = sample_df.drop_duplicates()
        return {
            "before_count": before_cnt,
            "after_count": 0,
            "before_sample": dups.head(5),
            "after_sample": cleaned.head(5),
            "explanation": f"Found {before_cnt} duplicate rows. All redundant duplicates will be removed, preserving the first instance."
        }

    elif fix_type == "impute_median" and target_col in sample_df.columns:
        null_mask = sample_df[target_col].isnull()
        before_cnt = int(null_mask.sum())
        median_val = float(pd.to_numeric(sample_df[target_col], errors="coerce").median())
        before_slice = sample_df[null_mask].head(5)
        
        preview_cleaned = sample_df.copy()
        preview_cleaned[target_col] = preview_cleaned[target_col].fillna(median_val)
        after_slice = preview_cleaned.loc[before_slice.index]

        return {
            "before_count": before_cnt,
            "after_count": 0,
            "before_sample": before_slice,
            "after_sample": after_slice,
            "explanation": f"Imputes {before_cnt} null values in '{target_col}' with the column median ({round(median_val, 2)})."
        }

    elif fix_type == "impute_mode" and target_col in sample_df.columns:
        null_mask = sample_df[target_col].isnull()
        before_cnt = int(null_mask.sum())
        mode_series = sample_df[target_col].mode()
        mode_val = mode_series.iloc[0] if len(mode_series) > 0 else "Unknown"
        before_slice = sample_df[null_mask].head(5)

        preview_cleaned = sample_df.copy()
        preview_cleaned[target_col] = preview_cleaned[target_col].fillna(mode_val)
        after_slice = preview_cleaned.loc[before_slice.index]

        return {
            "before_count": before_cnt,
            "after_count": 0,
            "before_sample": before_slice,
            "after_sample": after_slice,
            "explanation": f"Imputes {before_cnt} unassigned categories in '{target_col}' with the most common value ('{mode_val}')."
        }

    elif fix_type == "clean_numeric_type" and target_col in sample_df.columns:
        before_slice = sample_df.head(5)
        preview_cleaned = sample_df.copy()
        cleaned_series = (
            preview_cleaned[target_col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        preview_cleaned[target_col] = pd.to_numeric(cleaned_series, errors="coerce")
        after_slice = preview_cleaned.head(5)

        return {
            "before_count": len(sample_df),
            "after_count": 0,
            "before_sample": before_slice,
            "after_sample": after_slice,
            "explanation": f"Standardizes string formatting in '{target_col}', casting it to a clean numeric float column."
        }

    elif fix_type == "cap_outliers" and target_col in sample_df.columns:
        s = pd.to_numeric(sample_df[target_col], errors="coerce").dropna()
        p01 = s.quantile(0.01)
        p99 = s.quantile(0.99)
        outliers = sample_df[(sample_df[target_col] < p01) | (sample_df[target_col] > p99)]
        before_slice = outliers.head(5)

        preview_cleaned = sample_df.copy()
        preview_cleaned[target_col] = preview_cleaned[target_col].clip(lower=p01, upper=p99)
        after_slice = preview_cleaned.loc[before_slice.index] if not before_slice.empty else preview_cleaned.head(5)

        return {
            "before_count": len(outliers),
            "after_count": 0,
            "before_sample": before_slice,
            "after_sample": after_slice,
            "explanation": f"Caps extreme outliers in '{target_col}' to the 1st percentile ({round(p01, 2)}) and 99th percentile ({round(p99, 2)})."
        }

    elif fix_type == "fix_dates" and target_col and "___" in target_col:
        col1, col2 = target_col.split("___")
        d1 = pd.to_datetime(sample_df[col1], errors="coerce")
        d2 = pd.to_datetime(sample_df[col2], errors="coerce")
        inverted = (d1 > d2) & d1.notnull() & d2.notnull()
        before_slice = sample_df[inverted].head(5)

        preview_cleaned = sample_df.copy()
        for idx in sample_df[inverted].index:
            val1 = preview_cleaned.at[idx, col1]
            val2 = preview_cleaned.at[idx, col2]
            preview_cleaned.at[idx, col1] = val2
            preview_cleaned.at[idx, col2] = val1
        after_slice = preview_cleaned.loc[before_slice.index]

        return {
            "before_count": int(inverted.sum()),
            "after_count": 0,
            "before_sample": before_slice,
            "after_sample": after_slice,
            "explanation": f"Transposes {int(inverted.sum())} chronological date inversions between '{col1}' and '{col2}'."
        }

    elif fix_type == "drop_column" and target_col in sample_df.columns:
        return {
            "before_count": len(sample_df),
            "after_count": len(sample_df),
            "before_sample": sample_df[[target_col]].head(5),
            "after_sample": sample_df.drop(columns=[target_col]).head(5),
            "explanation": f"Permanently drops uninformative column '{target_col}'."
        }

    return None


def apply_fix(df, fix_type, target_col=None):
    """
    Applies the specified fix to the dataset and returns the resulting DataFrame.
    """
    if df is None or df.empty:
        return df

    cleaned = df.copy()

    if fix_type == "remove_duplicates":
        cleaned = cleaned.drop_duplicates()

    elif fix_type == "impute_median" and target_col in cleaned.columns:
        med = pd.to_numeric(cleaned[target_col], errors="coerce").median()
        cleaned[target_col] = cleaned[target_col].fillna(med)

    elif fix_type == "impute_mode" and target_col in cleaned.columns:
        m = cleaned[target_col].mode()
        val = m.iloc[0] if len(m) > 0 else "Unknown"
        cleaned[target_col] = cleaned[target_col].fillna(val)

    elif fix_type == "clean_numeric_type" and target_col in cleaned.columns:
        cleaned_series = (
            cleaned[target_col].astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace("%", "", regex=False)
            .str.strip()
        )
        cleaned[target_col] = pd.to_numeric(cleaned_series, errors="coerce")

    elif fix_type == "cap_outliers" and target_col in cleaned.columns:
        s = pd.to_numeric(cleaned[target_col], errors="coerce").dropna()
        if len(s) > 0:
            p01 = s.quantile(0.01)
            p99 = s.quantile(0.99)
            cleaned[target_col] = cleaned[target_col].clip(lower=p01, upper=p99)

    elif fix_type == "fix_dates" and target_col and "___" in target_col:
        col1, col2 = target_col.split("___")
        d1 = pd.to_datetime(cleaned[col1], errors="coerce")
        d2 = pd.to_datetime(cleaned[col2], errors="coerce")
        inverted = (d1 > d2) & d1.notnull() & d2.notnull()
        for idx in cleaned[inverted].index:
            v1 = cleaned.at[idx, col1]
            v2 = cleaned.at[idx, col2]
            cleaned.at[idx, col1] = v2
            cleaned.at[idx, col2] = v1

    elif fix_type == "drop_column" and target_col in cleaned.columns:
        cleaned = cleaned.drop(columns=[target_col])

    return cleaned
