
import pandas as pd


def calculate_health_score(
    df,
    outlier_results,
    consistency_results
):

    total_rows = len(df)
    total_columns = len(df.columns)

    if total_rows == 0 or total_columns == 0:
        return {
            "score": 0,
            "missing_penalty": 0,
            "duplicate_penalty": 0,
            "outlier_penalty": 0,
            "consistency_penalty": 0
        }

    # 1. MISSING VALUES
    total_cells = total_rows * total_columns

    missing_count = df.isnull().sum().sum()

    missing_rate = missing_count / total_cells

    missing_penalty = missing_rate * 40


    # 2. DUPLICATE ROWS
    duplicate_count = df.duplicated().sum()

    duplicate_rate = duplicate_count / total_rows

    duplicate_penalty = duplicate_rate * 25


    # 3. OUTLIERS
    numeric_columns = [
        column for column in df.select_dtypes(include="number").columns
        if "id" not in column.lower()
    ]

    if numeric_columns:

        total_numeric_cells = total_rows * len(numeric_columns)

        total_outliers = outlier_results["Outliers"].sum()

        outlier_rate = total_outliers / total_numeric_cells

    else:
        outlier_rate = 0

    outlier_penalty = outlier_rate * 20


    # 4. DATE CONSISTENCY
    consistency_count = len(consistency_results)

    consistency_rate = consistency_count / total_rows

    consistency_penalty = consistency_rate * 15


    # FINAL HEALTH SCORE
    total_penalty = (
        missing_penalty
        + duplicate_penalty
        + outlier_penalty
        + consistency_penalty
    )

    score = max(0, round(100 - total_penalty, 2))


    return {
        "score": score,
        "missing_penalty": round(missing_penalty, 2),
        "duplicate_penalty": round(duplicate_penalty, 2),
        "outlier_penalty": round(outlier_penalty, 2),
        "consistency_penalty": round(consistency_penalty, 2)
    }
