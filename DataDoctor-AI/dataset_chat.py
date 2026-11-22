"""
DataDoctor AI - Chat With Your Dataset Module (Feature 4)
Conversational analytics engine with natural language intent recognition,
zero-hallucination data calculations, tabular answers, and Plotly visualizations.
Works 100% offline via a rule-based analytical parser, with optional LLM enhancement.
"""

import os
import pandas as pd
import numpy as np
import plotly.express as px


def ask_dataset(query, df, workspace="Retail Store", mapping=None):
    """
    Parses natural language user queries and resolves them strictly against
    the active DataFrame.
    
    Returns:
        dict containing:
            - "answer": str textual response
            - "calculation": str breakdown of how the metric was computed
            - "table": pd.DataFrame (optional tabular result)
            - "chart": Plotly Figure (optional visual chart)
            - "status": "success" | "clarification_needed"
    """
    if df is None or df.empty:
        return {
            "answer": "No dataset is currently loaded. Please upload a CSV file or load a sample dataset first.",
            "calculation": "No data source available.",
            "table": None,
            "chart": None,
            "status": "clarification_needed"
        }

    q = query.lower().strip()
    mapping = mapping or {}
    total_records = len(df)

    # 1. DATA QUALITY & HEALTH SCORE QUERIES
    if any(k in q for k in ["health score", "quality", "clean", "score", "diagnosis"]):
        missing_total = int(df.isnull().sum().sum())
        dup_total = int(df.duplicated().sum())
        ans = (
            f"The dataset contains **{total_records:,} total records** and **{len(df.columns)} columns**.\n"
            f"- Missing Values: **{missing_total:,}**\n"
            f"- Duplicate Rows: **{dup_total:,}**\n"
            f"Navigate to **AI Data Doctor** or **Reports** to view the full 0-100 score breakdown and proposed fixes."
        )
        col_miss = df.isnull().sum()[lambda x: x > 0].reset_index()
        col_miss.columns = ["Column", "Missing Count"]
        return {
            "answer": ans,
            "calculation": f"Calculated df.isnull().sum().sum() = {missing_total} and df.duplicated().sum() = {dup_total}",
            "table": col_miss if not col_miss.empty else None,
            "chart": None,
            "status": "success"
        }

    # 2. MISSING VALUES QUERY
    if "missing" in q:
        null_s = df.isnull().sum()
        miss_df = pd.DataFrame({
            "Column": null_s.index,
            "Missing Values": null_s.values,
            "Percentage (%)": (null_s.values / total_records * 100).round(2)
        })
        miss_df = miss_df[miss_df["Missing Values"] > 0].sort_values("Missing Values", ascending=False)
        if miss_df.empty:
            return {
                "answer": f"Good news! None of the {len(df.columns)} columns contain missing values.",
                "calculation": "Checked df.isnull().sum() across all features.",
                "table": None,
                "chart": None,
                "status": "success"
            }
        else:
            return {
                "answer": f"Found missing values across **{len(miss_df)} column(s)**. See the table below:",
                "calculation": "Computed missing counts and percentages per column.",
                "table": miss_df,
                "chart": None,
                "status": "success"
            }

    # 3. REVENUE / SALES QUERIES
    if any(k in q for k in ["revenue", "sales", "earnings", "income", "turnover"]):
        rev_col = mapping.get("revenue")
        # Try finding revenue column dynamically if not mapped
        if not rev_col or rev_col not in df.columns:
            for c in df.columns:
                if any(w in c.lower() for w in ["rev", "sale", "amount", "total"]):
                    rev_col = c
                    break

        if rev_col and rev_col in df.columns:
            rev_series = pd.to_numeric(df[rev_col], errors="coerce").fillna(0.0)
            total_rev = float(rev_series.sum())
            avg_rev = float(rev_series.mean())

            # Check if user asked for product or category breakdown
            prod_col = mapping.get("product") or next((c for c in df.columns if any(p in c.lower() for p in ["product", "item", "service", "item_name"])), None)
            chart = None
            tbl = None

            if prod_col and prod_col in df.columns and any(k in q for k in ["product", "top", "item", "highest", "best"]):
                top_rev = df.groupby(prod_col)[rev_col].sum().reset_index().sort_values(rev_col, ascending=False).head(10)
                top_rev[rev_col] = top_rev[rev_col].round(2)
                tbl = top_rev
                chart = px.bar(top_rev, x=rev_col, y=prod_col, orientation="h", title=f"Top Products by Revenue ({rev_col})", color=rev_col, color_continuous_scale="Teal")
                chart.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0", yaxis=dict(autorange="reversed"))
                best_item = top_rev.iloc[0][prod_col]
                best_amt = top_rev.iloc[0][rev_col]
                ans = f"Total revenue across all transactions is **${total_rev:,.2f}**.\n\nThe top-performing offering is **'{best_item}'**, generating **${best_amt:,.2f}**."
            else:
                ans = f"Total recorded revenue is **${total_rev:,.2f}**, with an average transaction value of **${avg_rev:,.2f}** across {len(rev_series):,} records."

            return {
                "answer": ans,
                "calculation": f"Computed df['{rev_col}'].sum() = ${total_rev:,.2f}",
                "table": tbl,
                "chart": chart,
                "status": "success"
            }
        else:
            return {
                "answer": "Could not identify a revenue or sales column in the current dataset. Please check column mappings.",
                "calculation": "Inspected mapping and scanned column names for 'revenue', 'sales', 'amount'.",
                "table": None,
                "chart": None,
                "status": "clarification_needed"
            }

    # 4. EXPENSE / SPENDING / PROFIT QUERIES
    if any(k in q for k in ["expense", "spending", "cost", "profit", "margin", "p&l"]):
        exp_col = mapping.get("expense") or next((c for c in df.columns if any(w in c.lower() for w in ["exp", "cost", "spend"])), None)
        rev_col = mapping.get("revenue") or next((c for c in df.columns if any(w in c.lower() for w in ["rev", "sale"])), None)

        tot_rev = float(pd.to_numeric(df[rev_col], errors="coerce").sum()) if rev_col and rev_col in df.columns else 0.0
        tot_exp = float(pd.to_numeric(df[exp_col], errors="coerce").sum()) if exp_col and exp_col in df.columns else 0.0
        net_prof = tot_rev - tot_exp
        margin = round(net_prof / tot_rev * 100.0, 1) if tot_rev > 0 else 0.0

        tbl = None
        chart = None
        cat_col = mapping.get("category") or mapping.get("department") or next((c for c in df.columns if any(w in c.lower() for w in ["cat", "dept", "cost_center"])), None)
        
        if exp_col and cat_col and cat_col in df.columns:
            exp_by_cat = df.groupby(cat_col)[exp_col].sum().reset_index().sort_values(exp_col, ascending=False).head(10)
            exp_by_cat[exp_col] = exp_by_cat[exp_col].round(2)
            tbl = exp_by_cat
            chart = px.pie(exp_by_cat, names=cat_col, values=exp_col, title="Expense Distribution", hole=0.45, color_discrete_sequence=px.colors.sequential.Blues)
            chart.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")

        ans = (
            f"- **Total Revenue**: ${tot_rev:,.2f}\n"
            f"- **Total Expenses**: ${tot_exp:,.2f}\n"
            f"- **Net Profit**: **${net_prof:,.2f}** (Margin: **{margin}%**)\n"
        )
        return {
            "answer": ans,
            "calculation": f"Computed Revenue (${tot_rev:,.2f}) - Expenses (${tot_exp:,.2f}) = Net Profit (${net_prof:,.2f})",
            "table": tbl,
            "chart": chart,
            "status": "success"
        }

    # 5. HEALTHCARE: APPOINTMENTS, PATIENTS, ADMISSIONS
    if any(k in q for k in ["patient", "appointment", "doctor", "admission", "discharge", "physician", "department", "hospital", "stay"]):
        status_col = mapping.get("status") or next((c for c in df.columns if any(w in c.lower() for w in ["status", "appt_status", "visit_type"])), None)
        dept_col = mapping.get("department") or next((c for c in df.columns if any(w in c.lower() for w in ["dept", "department", "specialty"])), None)
        
        tbl = None
        chart = None
        ans = f"Analyzed **{total_records:,} patient records**.\n"

        if status_col and status_col in df.columns:
            st_counts = df[status_col].value_counts().reset_index()
            st_counts.columns = ["Status", "Count"]
            tbl = st_counts
            chart = px.pie(st_counts, names="Status", values="Count", title="Appointment Status Breakdown", hole=0.45, color_discrete_sequence=px.colors.sequential.Teal)
            chart.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")
            ans += f"- **Appointment Outcomes**: {', '.join(f'{r.Status}: {r.Count}' for _, r in st_counts.iterrows())}\n"

        if dept_col and dept_col in df.columns:
            top_dept = df[dept_col].value_counts().head(5).reset_index()
            top_dept.columns = ["Department", "Patient Encounters"]
            ans += f"- **Busiest Department**: **'{top_dept.iloc[0]['Department']}'** with {top_dept.iloc[0]['Patient Encounters']} visits.\n"

        return {
            "answer": ans,
            "calculation": f"Grouped records by status and department across {total_records} encounters.",
            "table": tbl,
            "chart": chart,
            "status": "success"
        }

    # 6. EDUCATION: MARKS, ATTENDANCE, GRADES, STUDENTS
    if any(k in q for k in ["student", "marks", "grade", "pass", "attendance", "teacher", "subject", "class"]):
        marks_col = mapping.get("marks") or next((c for c in df.columns if any(w in c.lower() for w in ["marks", "score", "grade_point", "result"])), None)
        att_col = mapping.get("attendance") or next((c for c in df.columns if any(w in c.lower() for w in ["att", "attendance", "presence"])), None)
        subj_col = mapping.get("subject") or next((c for c in df.columns if any(w in c.lower() for w in ["subj", "course", "topic"])), None)
        
        tbl = None
        chart = None
        ans = f"Analyzed academic dataset with **{total_records:,} student evaluations**.\n"

        if marks_col and marks_col in df.columns:
            valid_m = pd.to_numeric(df[marks_col], errors="coerce").dropna()
            avg_m = round(float(valid_m.mean()), 1)
            pass_p = round(float((valid_m >= 40.0).mean() * 100.0), 1)
            ans += f"- **Average Marks**: **{avg_m}%**\n- **Overall Pass Percentage**: **{pass_p}%** (at >=40% pass cutoff)\n"

        if att_col and att_col in df.columns:
            valid_att = pd.to_numeric(df[att_col], errors="coerce").dropna()
            avg_att = round(float(valid_att.mean()), 1)
            low_att = int((valid_att < 75.0).sum())
            ans += f"- **Average Attendance**: **{avg_att}%**\n- **Students with Low Attendance (<75%)**: **{low_att}** students\n"

        if subj_col and subj_col in df.columns and marks_col and marks_col in df.columns:
            subj_perf = df.groupby(subj_col)[marks_col].agg(["mean", "count"]).reset_index().rename(columns={"mean": "Average Marks", "count": "Evaluations"}).sort_values("Average Marks", ascending=False)
            subj_perf["Average Marks"] = subj_perf["Average Marks"].round(1)
            tbl = subj_perf
            chart = px.bar(subj_perf, x=subj_col, y="Average Marks", title="Subject Average Marks", color="Average Marks", color_continuous_scale="Teal")
            chart.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")

        return {
            "answer": ans,
            "calculation": f"Computed statistical averages on marks and attendance for {total_records} records.",
            "table": tbl,
            "chart": chart,
            "status": "success"
        }

    # 7. GENERAL SUMMARY / COLUMNS FALLBACK
    col_summary = pd.DataFrame({
        "Column Name": df.columns,
        "Data Type": df.dtypes.astype(str).values,
        "Unique Values": df.nunique().values,
        "Missing Count": df.isnull().sum().values
    })

    return {
        "answer": (
            f"Here is an executive summary of your dataset:\n"
            f"- **Records (Rows)**: {total_records:,}\n"
            f"- **Features (Columns)**: {len(df.columns)}\n"
            f"- **Available Columns**: {', '.join(f'`{c}`' for c in df.columns[:10])}{'...' if len(df.columns) > 10 else ''}\n\n"
            f"You can ask me specific questions like *'What is the total revenue?'*, *'Show missing values'*, *'Which product generated the most revenue?'*, or *'What is the average attendance?'*."
        ),
        "calculation": "Inspected DataFrame schema and calculated dimension properties.",
        "table": col_summary.head(10),
        "chart": None,
        "status": "success"
    }
