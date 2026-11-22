"""
DataDoctor AI - Business & Finance Analytics Module
Comprehensive financial analytics, revenue tracking, expense breakdowns,
profit & loss statements, budget analysis, cash flow, and data quality audits.
"""

import pandas as pd
import numpy as np


def map_business_finance_columns(df):
    """
    Maps raw DataFrame columns to standard financial keys using
    case-insensitive keyword and alias matching.
    """
    column_aliases = {
        "transaction_id": [
            "transaction_id", "invoice_id", "order_id", "txn_id",
            "trans_id", "receipt_no", "bill_id", "reference_no", "invoice_no"
        ],
        "date": [
            "date", "transaction_date", "trans_date", "invoice_date",
            "payment_date", "booking_date", "timestamp", "order_date"
        ],
        "revenue": [
            "revenue", "sales", "income", "amount", "total_amount",
            "gross_sales", "turnover", "inflow", "credit", "sales_revenue"
        ],
        "expense": [
            "expense", "expenses", "cost", "spending", "outflow",
            "expenditure", "total_cost", "debit", "cogs", "operating_cost"
        ],
        "customer": [
            "customer_id", "client_id", "customer_name", "client",
            "customer", "account_id", "buyer", "customer_code"
        ],
        "product": [
            "product", "service", "item", "product_name", "offering",
            "description", "item_name", "product_service", "sku"
        ],
        "category": [
            "category", "expense_category", "revenue_category", "type",
            "expense_type", "cost_center", "account_category", "item_category"
        ],
        "department": [
            "department", "dept", "business_unit", "division",
            "team", "cost_center", "branch"
        ],
        "region": [
            "region", "territory", "location", "market", "country",
            "city", "zone", "geography"
        ],
        "budget": [
            "budget", "planned_budget", "allocated_budget", "target_budget",
            "budget_amount", "target", "plan"
        ],
        "actual": [
            "actual", "actual_amount", "actual_cost", "actual_spend",
            "actual_expense", "spent", "actual_value"
        ],
        "status": [
            "status", "transaction_status", "payment_status", "state", "result"
        ],
        "payment_method": [
            "payment_method", "payment_type", "mode", "payment_mode",
            "channel", "method"
        ]
    }

    mapping = {}
    normalized_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    excluded_from_trans = ["employee_id", "empid", "emp_id", "staff_id", "patient_id", "student_id"]
    excluded_from_date = ["joining_date", "leaving_date", "hire_date", "exit_date", "birth_date"]

    for key, aliases in column_aliases.items():
        found_col = None
        for alias in aliases:
            if alias in normalized_cols:
                norm_cand = alias
                if key == "transaction_id" and any(exc in norm_cand for exc in excluded_from_trans):
                    continue
                if key == "date" and any(exc in norm_cand for exc in excluded_from_date):
                    continue
                found_col = normalized_cols[alias]
                break
        if not found_col:
            for norm_name, original_name in normalized_cols.items():
                if key == "transaction_id" and any(exc in norm_name for exc in excluded_from_trans):
                    continue
                if key == "date" and any(exc in norm_name for exc in excluded_from_date):
                    continue
                for alias in aliases:
                    if len(alias) >= 3 and alias in norm_name and original_name not in mapping.values():
                        found_col = original_name
                        break
                if found_col:
                    break
        if found_col:
            mapping[key] = found_col

    detected = list(mapping.keys())
    missing_critical = []
    if "revenue" not in mapping and "expense" not in mapping:
        missing_critical.append("financial figures (e.g. Revenue, Sales, Expense, or Amount)")

    return {
        "mapping": mapping,
        "detected": detected,
        "missing_critical": missing_critical,
        "optional_detected": [k for k in ["date", "customer", "product", "category", "department", "region", "budget", "actual", "status", "payment_method"] if k in mapping],
        "optional_missing": [k for k in ["date", "customer", "product", "category", "department", "region", "budget", "actual", "status", "payment_method"] if k not in mapping]
    }


def prepare_business_finance_data(df, mapping=None):
    """
    Standardizes business and finance records into clean internal columns.
    Computes Net Profit, Margins, Cash Inflow/Outflow, and variance safely.
    """
    if mapping is None:
        mapping_info = map_business_finance_columns(df)
        mapping = mapping_info["mapping"]

    work_df = df.copy()

    # 1. Transaction ID
    if "transaction_id" in mapping:
        work_df["_clean_txnid"] = work_df[mapping["transaction_id"]].astype(str).str.strip().replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    else:
        work_df["_clean_txnid"] = [f"TXN-{i+1:05d}" for i in range(len(work_df))]

    # 2. Date
    has_date = False
    if "date" in mapping:
        work_df["_clean_date"] = pd.to_datetime(work_df[mapping["date"]], errors="coerce")
        if work_df["_clean_date"].notna().sum() > 0:
            has_date = True
            work_df["_year_month"] = work_df["_clean_date"].dt.strftime("%Y-%m")
            work_df["_month_label"] = work_df["_clean_date"].dt.strftime("%b %Y")
            work_df["_year"] = work_df["_clean_date"].dt.year
            work_df["_date_only"] = work_df["_clean_date"].dt.date
    else:
        work_df["_clean_date"] = pd.NaT

    # 3. Revenue
    has_revenue = False
    if "revenue" in mapping:
        raw_rev = pd.to_numeric(work_df[mapping["revenue"]], errors="coerce")
        work_df["_clean_revenue"] = raw_rev.fillna(0.0)
        work_df["_is_negative_rev"] = raw_rev < 0
        has_revenue = True
    else:
        work_df["_clean_revenue"] = 0.0
        work_df["_is_negative_rev"] = False

    # 4. Expense
    has_expense = False
    if "expense" in mapping:
        raw_exp = pd.to_numeric(work_df[mapping["expense"]], errors="coerce")
        work_df["_clean_expense"] = raw_exp.fillna(0.0)
        work_df["_is_negative_exp"] = raw_exp < 0
        has_expense = True
    else:
        work_df["_clean_expense"] = 0.0
        work_df["_is_negative_exp"] = False

    # 5. Profit & Margin (Accrual basis)
    work_df["_clean_profit"] = work_df["_clean_revenue"] - work_df["_clean_expense"]

    # 6. Customer
    has_customer = False
    if "customer" in mapping:
        has_customer = True
        work_df["_clean_customer"] = work_df[mapping["customer"]].astype(str).str.strip().replace({"nan": "General Client", "None": "General Client", "": "General Client"})
    else:
        work_df["_clean_customer"] = "General Client"

    # 7. Product / Service
    has_product = False
    if "product" in mapping:
        has_product = True
        work_df["_clean_product"] = work_df[mapping["product"]].astype(str).str.strip().replace({"nan": "General Offering", "None": "General Offering", "": "General Offering"})
    else:
        work_df["_clean_product"] = "General Offering"

    # 8. Category
    has_category = False
    if "category" in mapping:
        has_category = True
        work_df["_clean_category"] = work_df[mapping["category"]].astype(str).str.strip().replace({"nan": "General", "None": "General", "": "General"})
    else:
        work_df["_clean_category"] = "General Operations"

    # 9. Department
    has_dept = False
    if "department" in mapping:
        has_dept = True
        work_df["_clean_dept"] = work_df[mapping["department"]].astype(str).str.strip().replace({"nan": "Corporate", "None": "Corporate", "": "Corporate"})
    else:
        work_df["_clean_dept"] = "Corporate"

    # 10. Region
    has_region = False
    if "region" in mapping:
        has_region = True
        work_df["_clean_region"] = work_df[mapping["region"]].astype(str).str.strip().replace({"nan": "Domestic", "None": "Domestic", "": "Domestic"})
    else:
        work_df["_clean_region"] = "All Regions"

    # 11. Budget & Actual
    has_budget = False
    if "budget" in mapping:
        work_df["_clean_budget"] = pd.to_numeric(work_df[mapping["budget"]], errors="coerce").fillna(0.0)
        has_budget = True
    else:
        work_df["_clean_budget"] = np.nan

    if "actual" in mapping:
        work_df["_clean_actual"] = pd.to_numeric(work_df[mapping["actual"]], errors="coerce").fillna(0.0)
    elif has_expense:
        work_df["_clean_actual"] = work_df["_clean_expense"]
    else:
        work_df["_clean_actual"] = np.nan

    # 12. Transaction Status
    has_status = False
    if "status" in mapping:
        has_status = True
        def norm_status(s):
            if pd.isna(s) or str(s).strip() in ["", "nan", "None"]:
                return "Completed"
            val = str(s).strip().capitalize()
            if "comp" in val.lower() or "succ" in val.lower() or "paid" in val.lower():
                return "Completed"
            elif "fail" in val.lower() or "decl" in val.lower() or "reject" in val.lower():
                return "Failed"
            elif "pend" in val.lower() or "proc" in val.lower():
                return "Pending"
            return val
        work_df["_clean_status"] = work_df[mapping["status"]].apply(norm_status)
    else:
        work_df["_clean_status"] = "Completed"

    # 13. Payment Method
    has_payment_method = False
    if "payment_method" in mapping:
        has_payment_method = True
        work_df["_clean_pm"] = work_df[mapping["payment_method"]].astype(str).str.strip().replace({"nan": "Direct", "None": "Direct", "": "Direct"})
    else:
        work_df["_clean_pm"] = "Direct / Electronic"

    metadata = {
        "has_date": has_date,
        "has_revenue": has_revenue,
        "has_expense": has_expense,
        "has_customer": has_customer,
        "has_product": has_product,
        "has_category": has_category,
        "has_dept": has_dept,
        "has_region": has_region,
        "has_budget": has_budget,
        "has_status": has_status,
        "has_payment_method": has_payment_method
    }

    return work_df, mapping, metadata


def summarize_business_finance(df):
    """
    Computes overarching executive business and financial KPIs.
    """
    total_records = len(df)
    total_revenue = float(df["_clean_revenue"].sum()) if "_clean_revenue" in df.columns else 0.0
    total_expenses = float(df["_clean_expense"].sum()) if "_clean_expense" in df.columns else 0.0
    net_profit = total_revenue - total_expenses
    
    profit_margin = round((net_profit / total_revenue * 100.0), 1) if total_revenue > 0 else 0.0
    total_transactions = total_records
    
    # Average transaction value (for revenue transactions)
    rev_txns = df[df["_clean_revenue"] > 0] if "_clean_revenue" in df.columns else pd.DataFrame()
    atv = round(total_revenue / max(1, len(rev_txns)), 2) if len(rev_txns) > 0 else 0.0
    
    total_customers = df["_clean_customer"].nunique() if "_clean_customer" in df.columns else 0

    # Monthly Growth Rate (MoM Revenue Growth)
    mom_growth = None
    if "_clean_date" in df.columns and "_year_month" in df.columns and total_revenue > 0:
        monthly_rev = df.groupby("_year_month")["_clean_revenue"].sum().reset_index()
        if len(monthly_rev) >= 2:
            prev_m = monthly_rev["_clean_revenue"].iloc[-2]
            curr_m = monthly_rev["_clean_revenue"].iloc[-1]
            if prev_m > 0:
                mom_growth = round(((curr_m - prev_m) / prev_m * 100.0), 1)

    return {
        "total_revenue": total_revenue,
        "total_expenses": total_expenses,
        "net_profit": net_profit,
        "profit_margin": profit_margin,
        "total_transactions": total_transactions,
        "atv": atv,
        "total_customers": total_customers,
        "mom_growth": mom_growth
    }


def get_revenue_analytics(df, freq="monthly"):
    """
    Computes revenue trends, growth rates, product/service share, department share, and regional sales.
    """
    res = {}

    # 1. Revenue Over Time
    if "_clean_date" in df.columns and df["_clean_date"].notna().sum() > 0:
        valid_df = df.dropna(subset=["_clean_date"]).copy()
        if freq == "monthly" and "_year_month" in valid_df.columns:
            rev_trend = valid_df.groupby("_year_month").agg(
                Revenue=("_clean_revenue", "sum"),
                Transactions=("_clean_txnid", "count")
            ).reset_index().rename(columns={"_year_month": "Period"}).sort_values("Period")
        else:
            rev_trend = valid_df.groupby("_date_only").agg(
                Revenue=("_clean_revenue", "sum"),
                Transactions=("_clean_txnid", "count")
            ).reset_index().rename(columns={"_date_only": "Period"}).sort_values("Period")
            rev_trend["Period"] = pd.to_datetime(rev_trend["Period"])
        res["revenue_trend"] = rev_trend
    else:
        res["revenue_trend"] = pd.DataFrame()

    # 2. Revenue by Product / Service
    if "_clean_product" in df.columns and "_clean_revenue" in df.columns:
        prod_rev = df[df["_clean_revenue"] > 0].groupby("_clean_product").agg(
            Revenue=("_clean_revenue", "sum"),
            Transactions=("_clean_txnid", "count")
        ).reset_index().rename(columns={"_clean_product": "Product / Service"}).sort_values("Revenue", ascending=False)
        total_rev = prod_rev["Revenue"].sum()
        prod_rev["Share (%)"] = (prod_rev["Revenue"] / max(1.0, total_rev) * 100.0).round(1)
        res["product_revenue"] = prod_rev
    else:
        res["product_revenue"] = pd.DataFrame()

    # 3. Revenue by Department
    if "_clean_dept" in df.columns and "_clean_revenue" in df.columns:
        dept_rev = df[df["_clean_revenue"] > 0].groupby("_clean_dept").agg(
            Revenue=("_clean_revenue", "sum"),
            Transactions=("_clean_txnid", "count")
        ).reset_index().rename(columns={"_clean_dept": "Department"}).sort_values("Revenue", ascending=False)
        res["dept_revenue"] = dept_rev
    else:
        res["dept_revenue"] = pd.DataFrame()

    # 4. Revenue by Region
    if "_clean_region" in df.columns and "_clean_revenue" in df.columns:
        reg_rev = df[df["_clean_revenue"] > 0].groupby("_clean_region").agg(
            Revenue=("_clean_revenue", "sum"),
            Transactions=("_clean_txnid", "count")
        ).reset_index().rename(columns={"_clean_region": "Region"}).sort_values("Revenue", ascending=False)
        res["region_revenue"] = reg_rev
    else:
        res["region_revenue"] = pd.DataFrame()

    return res


def get_expense_analytics(df, freq="monthly"):
    """
    Computes expense category distribution, monthly expense trends,
    department expenditures, and expense rankings.
    """
    res = {}
    exp_df = df[df["_clean_expense"] > 0].copy() if "_clean_expense" in df.columns else pd.DataFrame()

    if not exp_df.empty:
        total_exp = exp_df["_clean_expense"].sum()
        res["total_expense"] = total_exp

        # 1. Expense by Category
        if "_clean_category" in exp_df.columns:
            cat_exp = exp_df.groupby("_clean_category").agg(
                Expense=("_clean_expense", "sum"),
                Transactions=("_clean_txnid", "count")
            ).reset_index().rename(columns={"_clean_category": "Expense Category"}).sort_values("Expense", ascending=False)
            cat_exp["Share (%)"] = (cat_exp["Expense"] / max(1.0, total_exp) * 100.0).round(1)
            res["category_expense"] = cat_exp
        else:
            res["category_expense"] = pd.DataFrame()

        # 2. Expense Trends over time
        if "_clean_date" in exp_df.columns and exp_df["_clean_date"].notna().sum() > 0:
            date_col = "_year_month" if (freq == "monthly" and "_year_month" in exp_df.columns) else "_date_only"
            trend_exp = exp_df.groupby(date_col).agg(
                Expense=("_clean_expense", "sum"),
                Transactions=("_clean_txnid", "count")
            ).reset_index().rename(columns={date_col: "Period"}).sort_values("Period")
            res["expense_trend"] = trend_exp
        else:
            res["expense_trend"] = pd.DataFrame()

        # 3. Department-wise expenses
        if "_clean_dept" in exp_df.columns:
            dept_exp = exp_df.groupby("_clean_dept").agg(
                Expense=("_clean_expense", "sum"),
                Transactions=("_clean_txnid", "count")
            ).reset_index().rename(columns={"_clean_dept": "Department"}).sort_values("Expense", ascending=False)
            res["dept_expense"] = dept_exp
        else:
            res["dept_expense"] = pd.DataFrame()
    else:
        res["total_expense"] = 0.0
        res["category_expense"] = pd.DataFrame()
        res["expense_trend"] = pd.DataFrame()
        res["dept_expense"] = pd.DataFrame()

    return res


def get_profit_loss_analytics(df, freq="monthly"):
    """
    Computes Net Profit, Margins, Monthly P&L statements,
    department profitability, and product profitability.
    """
    res = {}
    tot_rev = float(df["_clean_revenue"].sum()) if "_clean_revenue" in df.columns else 0.0
    tot_exp = float(df["_clean_expense"].sum()) if "_clean_expense" in df.columns else 0.0
    net_profit = tot_rev - tot_exp
    margin = round((net_profit / tot_rev * 100.0), 1) if tot_rev > 0 else 0.0

    res["total_revenue"] = tot_rev
    res["total_expense"] = tot_exp
    res["net_profit"] = net_profit
    res["profit_margin"] = margin

    # Monthly P&L
    if "_clean_date" in df.columns and "_year_month" in df.columns:
        p_l = df.groupby("_year_month").agg(
            Revenue=("_clean_revenue", "sum"),
            Expense=("_clean_expense", "sum"),
            Net_Profit=("_clean_profit", "sum")
        ).reset_index().rename(columns={"_year_month": "Period"}).sort_values("Period")
        p_l["Margin (%)"] = np.where(p_l["Revenue"] > 0, (p_l["Net_Profit"] / p_l["Revenue"] * 100.0).round(1), 0.0)
        res["monthly_pl"] = p_l
    else:
        res["monthly_pl"] = pd.DataFrame()

    # Department Profitability
    if "_clean_dept" in df.columns:
        dept_pl = df.groupby("_clean_dept").agg(
            Revenue=("_clean_revenue", "sum"),
            Expense=("_clean_expense", "sum"),
            Net_Profit=("_clean_profit", "sum")
        ).reset_index().rename(columns={"_clean_dept": "Department"}).sort_values("Net_Profit", ascending=False)
        dept_pl["Margin (%)"] = np.where(dept_pl["Revenue"] > 0, (dept_pl["Net_Profit"] / dept_pl["Revenue"] * 100.0).round(1), 0.0)
        res["dept_profitability"] = dept_pl
    else:
        res["dept_profitability"] = pd.DataFrame()

    # Product Profitability
    if "_clean_product" in df.columns:
        prod_pl = df.groupby("_clean_product").agg(
            Revenue=("_clean_revenue", "sum"),
            Expense=("_clean_expense", "sum"),
            Net_Profit=("_clean_profit", "sum")
        ).reset_index().rename(columns={"_clean_product": "Product / Service"}).sort_values("Net_Profit", ascending=False)
        prod_pl["Margin (%)"] = np.where(prod_pl["Revenue"] > 0, (prod_pl["Net_Profit"] / prod_pl["Revenue"] * 100.0).round(1), 0.0)
        res["product_profitability"] = prod_pl
    else:
        res["product_profitability"] = pd.DataFrame()

    return res


def get_transaction_analytics(df):
    """
    Computes transaction success rates, payment method distribution,
    average order sizes, and customer spending leaderboards.
    """
    res = {}
    tot_txns = len(df)
    res["total_transactions"] = tot_txns

    # Status distribution
    if "_clean_status" in df.columns:
        st_counts = df["_clean_status"].value_counts().reset_index()
        st_counts.columns = ["Status", "Count"]
        st_counts["Percentage"] = (st_counts["Count"] / max(1, tot_txns) * 100.0).round(1)
        res["status_summary"] = st_counts
        status_dict = dict(zip(st_counts["Status"], st_counts["Count"]))
        completed_n = status_dict.get("Completed", tot_txns)
        failed_n = status_dict.get("Failed", 0)
        res["success_rate"] = round(completed_n / max(1, tot_txns) * 100.0, 1)
        res["failed_count"] = failed_n
    else:
        res["status_summary"] = pd.DataFrame()
        res["success_rate"] = 100.0
        res["failed_count"] = 0

    # Payment Methods
    if "_clean_pm" in df.columns:
        pm_counts = df["_clean_pm"].value_counts().reset_index()
        pm_counts.columns = ["Payment Method", "Transactions"]
        pm_counts["Percentage"] = (pm_counts["Transactions"] / max(1, tot_txns) * 100.0).round(1)
        res["payment_methods"] = pm_counts
    else:
        res["payment_methods"] = pd.DataFrame()

    # Top Customers
    if "_clean_customer" in df.columns and "_clean_revenue" in df.columns:
        top_cust = df[df["_clean_revenue"] > 0].groupby("_clean_customer").agg(
            Total_Spent=("_clean_revenue", "sum"),
            Orders=("_clean_txnid", "count")
        ).reset_index().rename(columns={"_clean_customer": "Customer ID / Name", "Total_Spent": "Total Revenue ($)"}).sort_values("Total Revenue ($)", ascending=False).head(10)
        res["top_customers"] = top_cust
    else:
        res["top_customers"] = pd.DataFrame()

    return res


def get_budget_analytics(df):
    """
    Compares budgeted vs actual spending, calculates budget utilization,
    and identifies over-budget departments and variance.
    """
    res = {}
    if "_clean_budget" in df.columns and df["_clean_budget"].notna().sum() > 0:
        budget_df = df[df["_clean_budget"] > 0].copy()
        if not budget_df.empty:
            total_b = float(budget_df["_clean_budget"].sum())
            total_a = float(budget_df["_clean_actual"].sum()) if "_clean_actual" in budget_df.columns else float(budget_df["_clean_expense"].sum())
            variance = total_b - total_a # Positive = Under budget, Negative = Over budget
            utilization = round((total_a / max(1.0, total_b) * 100.0), 1)

            res["has_budget_data"] = True
            res["total_budget"] = total_b
            res["total_actual"] = total_a
            res["variance"] = variance
            res["utilization_pct"] = utilization

            # Department Budget Breakdown
            if "_clean_dept" in budget_df.columns:
                dept_b = budget_df.groupby("_clean_dept").agg(
                    Budget=("_clean_budget", "sum"),
                    Actual=("_clean_actual", "sum")
                ).reset_index().rename(columns={"_clean_dept": "Department"})
                dept_b["Variance ($)"] = dept_b["Budget"] - dept_b["Actual"]
                dept_b["Utilization (%)"] = np.where(dept_b["Budget"] > 0, (dept_b["Actual"] / dept_b["Budget"] * 100.0).round(1), 0.0)
                dept_b["Status"] = np.where(dept_b["Actual"] > dept_b["Budget"], "Over Budget", "Within Budget")
                res["dept_budget_table"] = dept_b.sort_values("Variance ($)", ascending=True)
            else:
                res["dept_budget_table"] = pd.DataFrame()
            return res

    res["has_budget_data"] = False
    return res


def get_cash_flow_analytics(df, freq="monthly"):
    """
    Computes cash inflow (revenue), cash outflow (expenses), net cash flow,
    and temporal cash flow trends.
    """
    res = {}
    inflow = float(df["_clean_revenue"].sum()) if "_clean_revenue" in df.columns else 0.0
    outflow = float(df["_clean_expense"].sum()) if "_clean_expense" in df.columns else 0.0
    net_cf = inflow - outflow

    res["cash_inflow"] = inflow
    res["cash_outflow"] = outflow
    res["net_cash_flow"] = net_cf

    # Cash Flow Trend Over Time
    if "_clean_date" in df.columns and "_year_month" in df.columns:
        cf_trend = df.groupby("_year_month").agg(
            Inflow=("_clean_revenue", "sum"),
            Outflow=("_clean_expense", "sum"),
            Net_Cash_Flow=("_clean_profit", "sum")
        ).reset_index().rename(columns={"_year_month": "Period"}).sort_values("Period")
        res["cash_flow_trend"] = cf_trend
    else:
        res["cash_flow_trend"] = pd.DataFrame()

    return res


def check_business_finance_quality(df, mapping=None):
    """
    Financial dataset quality auditor:
    Detects duplicate transaction IDs, negative revenue/expense entries,
    missing financial amounts, unassigned categories, and computes a financial health score.
    """
    if mapping is None:
        mapping_info = map_business_finance_columns(df)
        mapping = mapping_info["mapping"]

    issues = []
    total_records = len(df)

    # 1. Duplicate Transactions
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "Category": "Duplicate Transaction Records",
            "Severity": "Critical",
            "Affected Records": dup_count,
            "Impact": f"{round(dup_count / total_records * 100, 1)}% of rows are duplicate ledger entries",
            "Recommendation": "Eliminate duplicate transactions to avoid double-counting revenue/expenses"
        })

    # 2. Missing Transaction ID
    if "_clean_txnid" in df.columns:
        unknown_txn = int((df["_clean_txnid"] == "Unknown").sum())
        if unknown_txn > 0:
            issues.append({
                "Category": "Missing Transaction / Invoice ID",
                "Severity": "High",
                "Affected Records": unknown_txn,
                "Impact": "Unidentified transactions cannot be audited against bank statements",
                "Recommendation": "Reconcile with gateway receipt numbers"
            })

    # 3. Negative Financial Amounts
    neg_rev = int(df["_is_negative_rev"].sum()) if "_is_negative_rev" in df.columns else 0
    neg_exp = int(df["_is_negative_exp"].sum()) if "_is_negative_exp" in df.columns else 0
    if neg_rev > 0 or neg_exp > 0:
        issues.append({
            "Category": "Negative Financial Amounts",
            "Severity": "High",
            "Affected Records": neg_rev + neg_exp,
            "Impact": "Negative revenue or expense values skew ledger summation",
            "Recommendation": "Verify credit notes / refunds or take absolute values"
        })

    # 4. Missing Categories or Departments
    if "_clean_category" in df.columns and "category" in mapping:
        missing_cat = int((df["_clean_category"] == "General Operations").sum())
        if missing_cat > 0:
            issues.append({
                "Category": "Uncategorized Financial Entries",
                "Severity": "Low",
                "Affected Records": missing_cat,
                "Impact": "Limits granular cost-center and P&L departmental attribution",
                "Recommendation": "Assign chart-of-accounts expense/revenue category"
            })

    # Health Score
    penalty = 0.0
    if dup_count > 0:
        penalty += (dup_count / total_records) * 30.0
    if neg_rev > 0 or neg_exp > 0:
        penalty += ((neg_rev + neg_exp) / total_records) * 25.0
    if "_clean_txnid" in df.columns:
        penalty += ((df["_clean_txnid"] == "Unknown").sum() / total_records) * 25.0

    score = max(0.0, round(100.0 - penalty, 1))

    return {
        "health_score": score,
        "issues_table": pd.DataFrame(issues) if issues else pd.DataFrame(columns=["Category", "Severity", "Affected Records", "Impact", "Recommendation"]),
        "issue_count": len(issues),
        "dup_count": dup_count
    }


def clean_business_finance_dataset(df, mapping, fill_missing=True, remove_duplicates=True, fix_negatives=True):
    """
    Cleans business and finance datasets:
    - Removes duplicate ledger entries
    - Rectifies negative numbers by applying absolute value (or refund normalization)
    - Imputes missing numerical values with median, categoricals with mode.
    """
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    # Fix negative revenue / expense
    if fix_negatives:
        for key in ["revenue", "expense", "budget", "actual"]:
            col = mapping.get(key)
            if col and col in cleaned.columns:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce").abs()

    # Fill missing values
    if fill_missing:
        for col in cleaned.columns:
            if cleaned[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                else:
                    mode_val = cleaned[col].mode()
                    if not mode_val.empty:
                        cleaned[col] = cleaned[col].fillna(mode_val[0])
                    else:
                        cleaned[col] = cleaned[col].fillna("General")

    return cleaned
