


import pandas as pd


# FEATURE 1: OUTLIER DETECTION
def detect_outliers(df):

    results = []

    numeric_columns = df.select_dtypes(include="number").columns

    for column in numeric_columns:

        # Skip ID columns
        if "id" in column.lower():
            continue

        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)

        IQR = Q3 - Q1

        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR

        outliers = df[
            (df[column] < lower_limit) |
            (df[column] > upper_limit)
        ][column]

        results.append({
            "Column": column,
            "Outliers": len(outliers),
            "Lower Limit": round(lower_limit, 2),
            "Upper Limit": round(upper_limit, 2)
        })

    return pd.DataFrame(results)


# FEATURE 2: CONSISTENCY GUARDIAN
def check_date_consistency(df):

    issues = []

    if (
        "Joining_Date" in df.columns
        and "Leaving_Date" in df.columns
    ):

        joining_dates = pd.to_datetime(
            df["Joining_Date"],
            errors="coerce"
        )

        leaving_dates = pd.to_datetime(
            df["Leaving_Date"],
            errors="coerce"
        )

        invalid_rows = df[
            leaving_dates.notna() &
            joining_dates.notna() &
            (leaving_dates < joining_dates)
        ]

        for index, row in invalid_rows.iterrows():

            issues.append({
                "Row": index + 1,
                "Issue": "Leaving date is before joining date",
                "Joining Date": row["Joining_Date"],
                "Leaving Date": row["Leaving_Date"]
            })

    return pd.DataFrame(issues)


# FEATURE 3: ML READINESS CHECKER
def check_ml_readiness(df):

    issues = []

    # Check missing values
    missing = df.isnull().sum()

    for column, count in missing.items():

        if count > 0:

            issues.append({
                "Column": column,
                "Issue": "Missing Values",
                "Count": int(count),
                "Recommendation": "Fill or remove missing values"
            })

    # Check duplicate rows
    duplicate_count = df.duplicated().sum()

    if duplicate_count > 0:

        issues.append({
            "Column": "Entire Dataset",
            "Issue": "Duplicate Rows",
            "Count": int(duplicate_count),
            "Recommendation": "Remove duplicate records"
        })

    # Check constant columns
    for column in df.columns:

        if df[column].nunique(dropna=False) <= 1:

            issues.append({
                "Column": column,
                "Issue": "Constant Column",
                "Count": 1,
                "Recommendation": "Consider removing this column"
            })

    # Check categorical columns
    categorical_columns = df.select_dtypes(
        include=["object", "category", "string"]
    ).columns

    for column in categorical_columns:

        issues.append({
            "Column": column,
            "Issue": "Categorical Data",
            "Count": int(df[column].nunique()),
            "Recommendation": "Consider encoding before ML training"
        })

    return pd.DataFrame(issues)
