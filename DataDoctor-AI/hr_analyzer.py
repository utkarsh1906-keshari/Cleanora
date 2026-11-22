"""
DataDoctor AI - Human Resources & Employee Workforce Analyzer
Dedicated analytical engine for employee records, payroll, retention,
and HR data quality checks.
"""

import pandas as pd
import numpy as np


def is_hr_dataset(df):
    """
    Intelligently determines whether a dataset is an Employee / HR dataset
    based on signature column names and characteristics.
    """
    if df is None or df.empty:
        return False
        
    lower_cols = [c.lower().strip().replace(" ", "_") for c in df.columns]
    
    hr_signatures = [
        "employee_id", "empid", "emp_id", "staff_id", "worker_id",
        "salary", "wage", "compensation", "payroll", "hourly_rate",
        "joining_date", "hire_date", "start_date", "employment_date",
        "leaving_date", "exit_date", "termination_date", "resignation_date",
        "job_title", "designation", "role_title"
    ]
    
    matches = sum(1 for sig in hr_signatures if any(sig in c for c in lower_cols))
    
    # If 2 or more HR signatures match (e.g. employee_id and salary, or joining_date and salary)
    has_employee_id = any(sig in c for c in lower_cols for sig in ["employee_id", "empid", "emp_id", "staff_id"])
    has_salary = any(sig in c for c in lower_cols for sig in ["salary", "wage", "payroll"])
    
    return (has_employee_id and has_salary) or matches >= 3


def map_hr_columns(df):
    """
    Maps raw column names to standardized HR dictionary keys.
    """
    column_aliases = {
        "employee_id": [
            "employee_id", "empid", "emp_id", "employeeid", "staff_id",
            "worker_id", "badge_id", "personnel_id", "id"
        ],
        "name": [
            "name", "employee_name", "full_name", "staff_name", "worker_name",
            "emp_name", "first_name"
        ],
        "age": [
            "age", "employee_age", "emp_age", "years"
        ],
        "department": [
            "department", "dept", "division", "team", "business_unit", "group", "function"
        ],
        "salary": [
            "salary", "annual_salary", "monthly_salary", "wage", "compensation",
            "pay", "remuneration", "base_pay", "payroll"
        ],
        "joining_date": [
            "joining_date", "join_date", "hire_date", "date_of_joining", "start_date",
            "employment_date", "doj"
        ],
        "leaving_date": [
            "leaving_date", "leave_date", "exit_date", "termination_date",
            "date_of_leaving", "resignation_date", "end_date", "dol"
        ],
        "email": [
            "email", "email_address", "work_email", "mail", "contact_email"
        ]
    }

    mapping = {}
    normalized_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    for key, aliases in column_aliases.items():
        found_col = None
        for alias in aliases:
            if alias in normalized_cols:
                found_col = normalized_cols[alias]
                break
        if not found_col:
            for norm_name, orig_name in normalized_cols.items():
                for alias in aliases:
                    if alias in norm_name and orig_name not in mapping.values():
                        found_col = orig_name
                        break
                if found_col:
                    break
        if found_col:
            mapping[key] = found_col

    missing_critical = []
    if "employee_id" not in mapping and "name" not in mapping:
        missing_critical.append("Employee Identification (e.g. Employee_ID or Name)")

    return {
        "mapping": mapping,
        "detected": list(mapping.keys()),
        "missing_critical": missing_critical,
        "optional_detected": [k for k in ["salary", "department", "joining_date", "leaving_date", "age", "email"] if k in mapping],
        "optional_missing": [k for k in ["salary", "department", "joining_date", "leaving_date", "age", "email"] if k not in mapping]
    }


def prepare_hr_data(df, mapping=None):
    """
    Standardizes HR data, computes tenure, active status, and clean demographic fields.
    """
    if mapping is None:
        mapping = map_hr_columns(df)["mapping"]

    work_df = df.copy()

    # 1. Employee ID
    if "employee_id" in mapping:
        work_df["_clean_eid"] = work_df[mapping["employee_id"]].astype(str).str.strip()
    else:
        work_df["_clean_eid"] = [f"EMP-{i+1:04d}" for i in range(len(work_df))]

    # 2. Employee Name
    if "name" in mapping:
        work_df["_clean_name"] = work_df[mapping["name"]].astype(str).str.strip()
    else:
        work_df["_clean_name"] = work_df["_clean_eid"]

    # 3. Department
    if "department" in mapping:
        work_df["_clean_dept"] = work_df[mapping["department"]].astype(str).str.strip().replace(
            {"nan": "Unassigned", "None": "Unassigned", "": "Unassigned"}
        )
    else:
        work_df["_clean_dept"] = "General Staff"

    # 4. Salary
    if "salary" in mapping:
        work_df["_clean_salary"] = pd.to_numeric(
            work_df[mapping["salary"]].astype(str).str.replace("$", "", regex=False).str.replace(",", "", regex=False),
            errors="coerce"
        ).fillna(0.0)
    else:
        work_df["_clean_salary"] = 0.0

    # 5. Age
    has_age = False
    if "age" in mapping:
        work_df["_clean_age"] = pd.to_numeric(work_df[mapping["age"]], errors="coerce")
        has_age = True

    # 6. Dates & Status
    has_join = False
    has_leave = False

    if "joining_date" in mapping:
        work_df["_clean_join"] = pd.to_datetime(work_df[mapping["joining_date"]], errors="coerce")
        has_join = True
    else:
        work_df["_clean_join"] = pd.NaT

    if "leaving_date" in mapping:
        work_df["_clean_leave"] = pd.to_datetime(work_df[mapping["leaving_date"]], errors="coerce")
        has_leave = True
    else:
        work_df["_clean_leave"] = pd.NaT

    work_df["_is_departed"] = work_df["_clean_leave"].notna()
    work_df["_is_active"] = ~work_df["_is_departed"]

    # Compute tenure
    now = pd.Timestamp.now()
    end_date = work_df["_clean_leave"].fillna(now)
    tenure_days = (end_date - work_df["_clean_join"]).dt.days
    work_df["_tenure_years"] = (tenure_days / 365.25).clip(lower=0.0).round(1)

    metadata = {
        "has_salary": "salary" in mapping,
        "has_dept": "department" in mapping,
        "has_age": has_age,
        "has_join": has_join,
        "has_leave": has_leave,
        "mapping": mapping
    }

    return work_df, mapping, metadata


def summarize_hr(df):
    """
    Generates high-level HR and workforce metrics.
    """
    total_employees = len(df)
    active_employees = int(df["_is_active"].sum()) if "_is_active" in df.columns else total_employees
    departed_employees = int(df["_is_departed"].sum()) if "_is_departed" in df.columns else 0
    turnover_rate = round(departed_employees / total_employees * 100, 1) if total_employees > 0 else 0.0

    salaries = df["_clean_salary"][df["_clean_salary"] > 0] if "_clean_salary" in df.columns else pd.Series([], dtype=float)
    total_payroll = round(float(salaries.sum()), 2) if len(salaries) > 0 else 0.0
    avg_salary = round(float(salaries.mean()), 2) if len(salaries) > 0 else 0.0
    median_salary = round(float(salaries.median()), 2) if len(salaries) > 0 else 0.0

    valid_ages = df["_clean_age"].dropna() if "_clean_age" in df.columns else pd.Series([], dtype=float)
    avg_age = round(float(valid_ages.mean()), 1) if len(valid_ages) > 0 else None

    total_depts = int(df["_clean_dept"].nunique()) if "_clean_dept" in df.columns else 1

    valid_tenures = df["_tenure_years"].dropna() if "_tenure_years" in df.columns else pd.Series([], dtype=float)
    avg_tenure = round(float(valid_tenures.mean()), 1) if len(valid_tenures) > 0 else None

    return {
        "total_employees": total_employees,
        "active_employees": active_employees,
        "departed_employees": departed_employees,
        "turnover_rate": turnover_rate,
        "total_payroll": total_payroll,
        "avg_salary": avg_salary,
        "median_salary": median_salary,
        "avg_age": avg_age,
        "total_departments": total_depts,
        "avg_tenure": avg_tenure
    }


def get_hr_quality_check(df, mapping):
    """
    Detects HR-specific data anomalies:
    - Duplicate employee IDs
    - Leaving date before joining date
    - Invalid employee ages (<16 or >80)
    - Missing department assignments
    """
    issues = []
    
    # 1. Duplicates
    if "_clean_eid" in df.columns:
        dup_count = int(df.duplicated(subset=["_clean_eid"]).sum())
        if dup_count > 0:
            issues.append({
                "Category": "Duplicate Employee IDs",
                "Severity": "High",
                "Affected Records": dup_count,
                "Impact": "Duplicated employee IDs distort headcount and payroll auditing",
                "Recommendation": "Merge or re-assign unique Employee ID numbers"
            })

    # 2. Inconsistent Dates (Leaving before Joining)
    if "_clean_join" in df.columns and "_clean_leave" in df.columns:
        invalid_dates = df[df["_clean_leave"].notna() & df["_clean_join"].notna() & (df["_clean_leave"] < df["_clean_join"])]
        if len(invalid_dates) > 0:
            issues.append({
                "Category": "Inverted Employment Dates",
                "Severity": "High",
                "Affected Records": len(invalid_dates),
                "Impact": "Leaving date recorded chronologically before joining date creates negative tenure",
                "Recommendation": "Reconcile exit interview logs with HR hiring records"
            })

    # 3. Invalid Ages
    if "_clean_age" in df.columns:
        bad_ages = df[df["_clean_age"].notna() & ((df["_clean_age"] < 16) | (df["_clean_age"] > 85))]
        if len(bad_ages) > 0:
            issues.append({
                "Category": "Out-of-Range Employee Age",
                "Severity": "Medium",
                "Affected Records": len(bad_ages),
                "Impact": "Ages outside working range (<16 or >85) signal entry typos",
                "Recommendation": "Verify date of birth against official onboarding identification"
            })

    return {
        "issues_table": pd.DataFrame(issues) if issues else pd.DataFrame(columns=["Category", "Severity", "Affected Records", "Impact", "Recommendation"]),
        "issue_count": len(issues)
    }
