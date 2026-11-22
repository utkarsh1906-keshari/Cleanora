"""
DataDoctor AI - What-If Scenario Simulator Module (Feature 3)
Interactive scenario modeling and sensitive stress-testing across all four
industry domains without mutating baseline dataset records.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go


def simulate_retail_scenario(df, mapping, price_pct=0.0, cost_pct=0.0, volume_pct=0.0):
    """
    Simulates retail pricing, cost optimization, and volume elasticity scenarios.
    """
    mapping = mapping or {}
    rev_col = mapping.get("revenue")
    cost_col = mapping.get("cost")
    qty_col = mapping.get("quantity")

    orig_rev = float(pd.to_numeric(df[rev_col], errors="coerce").sum()) if rev_col in df.columns else 1000.0
    orig_qty = float(pd.to_numeric(df[qty_col], errors="coerce").sum()) if qty_col in df.columns else len(df)
    
    if cost_col in df.columns:
        orig_cost = float(pd.to_numeric(df[cost_col], errors="coerce").sum())
    else:
        orig_cost = orig_rev * 0.55 # Estimate 55% COGS if unmapped

    orig_profit = orig_rev - orig_cost
    orig_margin = round(orig_profit / orig_rev * 100.0, 1) if orig_rev > 0 else 0.0

    # Model Simulated Values
    vol_multiplier = 1.0 + (volume_pct / 100.0)
    price_multiplier = 1.0 + (price_pct / 100.0)
    cost_multiplier = 1.0 + (cost_pct / 100.0)

    sim_rev = orig_rev * price_multiplier * vol_multiplier
    sim_cost = orig_cost * cost_multiplier * vol_multiplier
    sim_profit = sim_rev - sim_cost
    sim_margin = round(sim_profit / sim_rev * 100.0, 1) if sim_rev > 0 else 0.0

    rev_delta_pct = round((sim_rev - orig_rev) / orig_rev * 100.0, 1) if orig_rev > 0 else 0.0
    profit_delta_pct = round((sim_profit - orig_profit) / max(1.0, abs(orig_profit)) * 100.0, 1)

    return {
        "domain": "Retail Store",
        "orig_revenue": orig_rev,
        "sim_revenue": sim_rev,
        "rev_delta_pct": rev_delta_pct,
        "orig_profit": orig_profit,
        "sim_profit": sim_profit,
        "profit_delta_pct": profit_delta_pct,
        "orig_margin": orig_margin,
        "sim_margin": sim_margin,
        "margin_delta": round(sim_margin - orig_margin, 1)
    }


def simulate_healthcare_scenario(df, mapping, cancel_reduction_pct=20.0, capacity_change_pct=10.0):
    """
    Simulates clinical appointment fulfillment and bed capacity changes.
    """
    mapping = mapping or {}
    status_col = mapping.get("status", "Status")
    
    total_appts = len(df)
    if status_col in df.columns:
        cancelled = int((df[status_col].astype(str).str.lower() == "cancelled").sum())
        completed = int((df[status_col].astype(str).str.lower() == "completed").sum())
    else:
        cancelled = int(total_appts * 0.18)
        completed = total_appts - cancelled

    # Reduction in cancellations converts cancellations to completed
    recovered_visits = int(cancelled * (cancel_reduction_pct / 100.0))
    capacity_expansion = int(completed * (capacity_change_pct / 100.0))
    
    sim_completed = completed + recovered_visits + capacity_expansion
    sim_cancelled = max(0, cancelled - recovered_visits)
    completion_rate_orig = round(completed / max(1, total_appts) * 100.0, 1)
    completion_rate_sim = round(sim_completed / max(1, total_appts + capacity_expansion) * 100.0, 1)

    return {
        "domain": "Healthcare",
        "orig_completed": completed,
        "sim_completed": sim_completed,
        "recovered_visits": recovered_visits,
        "orig_cancelled": cancelled,
        "sim_cancelled": sim_cancelled,
        "orig_completion_rate": completion_rate_orig,
        "sim_completion_rate": completion_rate_sim,
        "delta_patients": sim_completed - completed
    }


def simulate_education_scenario(df, mapping, marks_boost_pct=5.0, attendance_boost_pct=5.0, pass_threshold=40.0):
    """
    Simulates student marks improvements, tutoring interventions, and attendance recovery.
    """
    mapping = mapping or {}
    marks_col = mapping.get("marks", "Marks")
    att_col = mapping.get("attendance", "Attendance")

    if marks_col in df.columns:
        orig_marks = pd.to_numeric(df[marks_col], errors="coerce").dropna()
    else:
        orig_marks = pd.Series([65.0, 72.0, 45.0, 88.0, 35.0, 90.0])

    if att_col in df.columns:
        orig_att = pd.to_numeric(df[att_col], errors="coerce").dropna()
    else:
        orig_att = pd.Series([78.0, 85.0, 68.0, 92.0, 62.0])

    orig_avg_marks = round(float(orig_marks.mean()), 1)
    orig_pass_pct = round(float((orig_marks >= pass_threshold).mean() * 100.0), 1)
    orig_support_needed = int((orig_marks < 50.0).sum())
    orig_avg_att = round(float(orig_att.mean()), 1)

    # Simulated Values
    sim_marks = (orig_marks * (1.0 + marks_boost_pct / 100.0)).clip(upper=100.0)
    sim_att = (orig_att * (1.0 + attendance_boost_pct / 100.0)).clip(upper=100.0)

    sim_avg_marks = round(float(sim_marks.mean()), 1)
    sim_pass_pct = round(float((sim_marks >= pass_threshold).mean() * 100.0), 1)
    sim_support_needed = int((sim_marks < 50.0).sum())
    sim_avg_att = round(float(sim_att.mean()), 1)

    return {
        "domain": "Education",
        "orig_avg_marks": orig_avg_marks,
        "sim_avg_marks": sim_avg_marks,
        "orig_pass_pct": orig_pass_pct,
        "sim_pass_pct": sim_pass_pct,
        "orig_support_needed": orig_support_needed,
        "sim_support_needed": sim_support_needed,
        "students_lifted": orig_support_needed - sim_support_needed,
        "orig_avg_att": orig_avg_att,
        "sim_avg_att": sim_avg_att
    }


def simulate_finance_scenario(df, mapping, rev_change_pct=5.0, exp_reduction_pct=5.0, budget_cut_pct=0.0):
    """
    Simulates corporate financial scenarios: revenue acceleration, cost cutting, and budget allocation.
    """
    mapping = mapping or {}
    rev_col = mapping.get("revenue")
    exp_col = mapping.get("expense")

    orig_rev = float(pd.to_numeric(df[rev_col], errors="coerce").sum()) if rev_col in df.columns else 500000.0
    orig_exp = float(pd.to_numeric(df[exp_col], errors="coerce").sum()) if exp_col in df.columns else 350000.0
    orig_profit = orig_rev - orig_exp
    orig_margin = round(orig_profit / orig_rev * 100.0, 1) if orig_rev > 0 else 0.0

    sim_rev = orig_rev * (1.0 + rev_change_pct / 100.0)
    sim_exp = orig_exp * (1.0 - exp_reduction_pct / 100.0)
    sim_profit = sim_rev - sim_exp
    sim_margin = round(sim_profit / sim_rev * 100.0, 1) if sim_rev > 0 else 0.0

    profit_delta_pct = round((sim_profit - orig_profit) / max(1.0, abs(orig_profit)) * 100.0, 1)

    return {
        "domain": "Business & Finance",
        "orig_revenue": orig_rev,
        "sim_revenue": sim_rev,
        "orig_expenses": orig_exp,
        "sim_expenses": sim_exp,
        "orig_profit": orig_profit,
        "sim_profit": sim_profit,
        "profit_delta_pct": profit_delta_pct,
        "orig_margin": orig_margin,
        "sim_margin": sim_margin,
        "margin_delta": round(sim_margin - orig_margin, 1)
    }


def create_simulation_comparison_chart(metric_labels, orig_values, sim_values, chart_title="Scenario Comparison"):
    """
    Generates a grouped bar chart visualizing Baseline vs Simulated results.
    """
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=metric_labels,
        y=orig_values,
        name="Baseline (Actual)",
        marker_color="#38BDF8"
    ))
    fig.add_trace(go.Bar(
        x=metric_labels,
        y=sim_values,
        name="Simulated Outcome",
        marker_color="#34D399"
    ))

    fig.update_layout(
        barmode="group",
        title=chart_title,
        paper_bgcolor="#0B1120",
        plot_bgcolor="#16243A",
        font_color="#E2E8F0",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig
