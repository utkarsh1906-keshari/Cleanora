"""
DataDoctor AI - Executive PDF Report Generator Module (Feature 5)
Automated generation of publication-grade C-level PDF audit reports
with domain KPI scorecards, anomaly audit tables, and data governance roadmaps.
"""

import io
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors


def generate_pdf_report(
    df,
    workspace="Retail Store",
    dataset_name="dataset.csv",
    summary_dict=None,
    health_score=85.0,
    anomalies_list=None,
    issues_list=None,
    cleaned_df=None
):
    """
    Generates a publication-grade PDF executive report as bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom Color Palette
    PRIMARY_COLOR = colors.HexColor("#0F172A")    # Deep Navy
    ACCENT_COLOR = colors.HexColor("#0284C7")     # Cyan / Sky
    SUCCESS_COLOR = colors.HexColor("#059669")    # Emerald
    WARNING_COLOR = colors.HexColor("#D97706")    # Amber
    MUTED_TEXT = colors.HexColor("#64748B")       # Slate
    LIGHT_BG = colors.HexColor("#F8FAFC")         # Crisp light grey
    BORDER_COLOR = colors.HexColor("#E2E8F0")

    # Custom Typography Styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=26,
        textColor=PRIMARY_COLOR,
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        textColor=MUTED_TEXT,
        spaceAfter=15
    )
    h2_style = ParagraphStyle(
        "Heading2Custom",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=17,
        textColor=PRIMARY_COLOR,
        spaceBefore=12,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1E293B")
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        parent=body_style,
        fontName="Helvetica-Bold"
    )

    story = []

    # 1. HEADER
    story.append(Paragraph("DataDoctor AI &mdash; Executive Dataset Audit", title_style))
    gen_time = datetime.now().strftime("%B %d, %Y &bull; %H:%M:%S UTC")
    story.append(Paragraph(f"Workspace: <b>{workspace}</b> | Dataset: <b>{dataset_name}</b> | Generated: {gen_time}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=ACCENT_COLOR, spaceAfter=14))

    # 2. EXECUTIVE SCORECARD
    total_rows = len(df) if df is not None else 0
    total_cols = len(df.columns) if df is not None else 0
    total_nulls = int(df.isnull().sum().sum()) if df is not None else 0
    total_dups = int(df.duplicated().sum()) if df is not None else 0
    anomaly_cnt = len(anomalies_list) if anomalies_list else 0

    score_text = f"<b>{health_score}/100</b>"
    status_label = "Optimal" if health_score >= 80 else ("Moderate" if health_score >= 50 else "Critical")

    scorecard_data = [
        [
            Paragraph("<b>Overall Health Score</b>", body_style),
            Paragraph("<b>Total Records</b>", body_style),
            Paragraph("<b>Total Features</b>", body_style),
            Paragraph("<b>Missing Values</b>", body_style),
            Paragraph("<b>Duplicate Rows</b>", body_style),
            Paragraph("<b>Anomalies</b>", body_style)
        ],
        [
            Paragraph(f"<font size=13 color='{ACCENT_COLOR.hexval()}'>{score_text}</font><br/><font size=7 color='{MUTED_TEXT.hexval()}'>{status_label}</font>", body_style),
            Paragraph(f"<font size=11><b>{total_rows:,}</b></font>", body_style),
            Paragraph(f"<font size=11><b>{total_cols}</b></font>", body_style),
            Paragraph(f"<font size=11><b>{total_nulls:,}</b></font>", body_style),
            Paragraph(f"<font size=11><b>{total_dups:,}</b></font>", body_style),
            Paragraph(f"<font size=11 color='{WARNING_COLOR.hexval()}'><b>{anomaly_cnt}</b></font>", body_style)
        ]
    ]

    scorecard_table = Table(scorecard_data, colWidths=[90, 88, 88, 88, 88, 88])
    scorecard_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_BG),
        ("BOX", (0, 0), (-1, -1), 1, BORDER_COLOR),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(scorecard_table)
    story.append(Spacer(1, 14))

    # 3. INDUSTRY-SPECIFIC PERFORMANCE METRICS
    story.append(Paragraph(f"Industry Key Performance Indicators ({workspace})", h2_style))
    summary_dict = summary_dict or {}

    kpi_rows = [["Metric Name", "Value", "Benchmark / Context"]]
    if workspace == "Retail Store":
        kpi_rows.append(["Total Revenue", f"${summary_dict.get('total_revenue', 0):,.2f}", "Total gross top-line volume"])
        kpi_rows.append(["Units Sold", f"{summary_dict.get('total_units', 0):,}", "Inventory sales volume"])
        kpi_rows.append(["Transactions", f"{summary_dict.get('total_transactions', 0):,}", "Completed customer checkouts"])
        kpi_rows.append(["Total Profit", f"${summary_dict.get('total_profit', 0):,.2f}", f"Gross Margin: {summary_dict.get('profit_margin', 0)}%"])
    elif workspace == "Healthcare":
        kpi_rows.append(["Total Unique Patients", f"{summary_dict.get('unique_patients', total_rows):,}", "Distinct clinical patient encounters"])
        kpi_rows.append(["Total Appointments", f"{summary_dict.get('total_appointments', total_rows):,}", "Scheduled clinical visits"])
        kpi_rows.append(["Total Admissions", f"{summary_dict.get('total_admissions', 0):,}", "Inpatient bed admissions"])
        kpi_rows.append(["Average Length of Stay", f"{summary_dict.get('alos', 0)} days", "Inpatient bed turnover velocity"])
    elif workspace == "Education":
        kpi_rows.append(["Total Students", f"{summary_dict.get('total_students', total_rows):,}", "Enrolled academic cohort"])
        kpi_rows.append(["Average Marks", f"{summary_dict.get('avg_marks', 0)}%", "Cohort grade average"])
        kpi_rows.append(["Pass Percentage", f"{summary_dict.get('pass_pct', 0)}%", "Students meeting standard cutoff"])
        kpi_rows.append(["Average Attendance", f"{summary_dict.get('avg_attendance', 0)}%", "Presence compliance rate"])
    else: # Business & Finance
        kpi_rows.append(["Total Revenue", f"${summary_dict.get('total_revenue', 0):,.2f}", "Total commercial top-line inflow"])
        kpi_rows.append(["Total Expenses", f"${summary_dict.get('total_expenses', 0):,.2f}", "Operational & capital outflow"])
        kpi_rows.append(["Net Profit", f"${summary_dict.get('net_profit', 0):,.2f}", f"Net Margin: {summary_dict.get('profit_margin', 0)}%"])
        kpi_rows.append(["Avg Transaction Value", f"${summary_dict.get('atv', 0):,.2f}", "Average revenue per order"])

    kpi_table_data = []
    for r in kpi_rows:
        kpi_table_data.append([
            Paragraph(f"<b>{r[0]}</b>", body_style),
            Paragraph(f"<b>{r[1]}</b>", body_style),
            Paragraph(r[2], body_style)
        ])

    kpi_table = Table(kpi_table_data, colWidths=[180, 150, 200])
    kpi_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E293B")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    story.append(kpi_table)
    story.append(Spacer(1, 14))

    # 4. DETECTED ANOMALIES & AUDIT FINDINGS
    story.append(Paragraph("Smart Anomaly Audit & Unusual Patterns", h2_style))
    if anomalies_list and len(anomalies_list) > 0:
        anom_rows = [["Type", "Affected Entity", "Actual vs Expected", "Severity"]]
        for a in anomalies_list[:6]:
            anom_rows.append([
                Paragraph(a.get("anomaly_type", "Anomaly"), body_style),
                Paragraph(a.get("column_or_category", "N/A"), body_style),
                Paragraph(f"{a.get('actual_value', '')} (Exp: {a.get('expected_value', '')})", body_style),
                Paragraph(f"<b>{a.get('severity', 'Medium')}</b>", body_style)
            ])
        anom_table = Table(anom_rows, colWidths=[140, 140, 170, 80])
        anom_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(anom_table)
    else:
        story.append(Paragraph("<i>No critical anomalies or pattern outliers detected in current sample.</i>", body_style))

    story.append(Spacer(1, 14))

    # 5. BEFORE VS AFTER CLEANING (IF CLEANED)
    if cleaned_df is not None:
        story.append(Paragraph("Smart Data Cleaning Impact Summary", h2_style))
        clean_rows = [
            ["Data Quality Metric", "Before Cleaning", "After Cleaning", "Status"],
            ["Total Observations (Rows)", f"{total_rows:,}", f"{len(cleaned_df):,}", "Deduplicated"],
            ["Missing Values (Nulls)", f"{total_nulls:,}", f"{int(cleaned_df.isnull().sum().sum()):,}", "Imputed / Resolved"],
            ["Duplicate Records", f"{total_dups:,}", f"{int(cleaned_df.duplicated().sum()):,}", "Zero Duplicates"],
        ]
        clean_table_data = []
        for r in clean_rows:
            clean_table_data.append([
                Paragraph(f"<b>{r[0]}</b>", body_style),
                Paragraph(r[1], body_style),
                Paragraph(f"<b>{r[2]}</b>", body_style),
                Paragraph(f"<font color='{SUCCESS_COLOR.hexval()}'><b>{r[3]}</b></font>", body_style)
            ])
        clean_table = Table(clean_table_data, colWidths=[180, 110, 110, 130])
        clean_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#065F46")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.5, BORDER_COLOR),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT_BG]),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(clean_table)
        story.append(Spacer(1, 14))

    # 6. INDUSTRY-SPECIFIC STRATEGIC RECOMMENDATIONS
    story.append(Paragraph(f"Strategic Industry Recommendations ({workspace})", h2_style))
    try:
        from recommendation_engine import get_industry_recommendations
        ind_recs = get_industry_recommendations(df, workspace=workspace, summary=summary_dict)
    except Exception:
        ind_recs = []

    if ind_recs:
        for idx, rec in enumerate(ind_recs[:4], 1):
            story.append(Paragraph(f"<b>{idx}. {rec['title']}</b> <font color='{PRIMARY_COLOR.hexval()}'>({rec['severity'].upper()})</font>", body_style))
            story.append(Paragraph(f"&bull; <i>Insight:</i> {rec['insight']}", body_style))
            story.append(Paragraph(f"&bull; <i>Action:</i> {rec['recommendation']}", body_style))
            story.append(Spacer(1, 4))
    else:
        if workspace == "Retail Store":
            safe_recs = [
                "1. <b>Inventory Replenishment:</b> Configure automated reorder points on fast-moving retail SKUs.",
                "2. <b>Basket Optimization:</b> Implement checkout bundling and cross-merchandising promotions to elevate AOV.",
                "3. <b>Margin Protection:</b> Audit supplier cost changes to prevent retail gross margin erosion."
            ]
        elif workspace == "Healthcare":
            safe_recs = [
                "1. <b>Appointment Adherence:</b> Deploy automated appointment reminders to minimize clinic cancellations.",
                "2. <b>Inpatient Flow:</b> Standardize multidisciplinary clinical pathways to optimize Average Length of Stay.",
                "3. <b>EHR Integrity:</b> Deduplicate master patient indexes to ensure complete longitudinal patient records."
            ]
        elif workspace == "Education":
            safe_recs = [
                "1. <b>Academic Remediation:</b> Deploy tutoring cohorts for students scoring below grade proficiency.",
                "2. <b>Attendance Early-Warning:</b> Enforce alerts for students falling below statutory attendance thresholds.",
                "3. <b>Registry Hygiene:</b> Reconcile student roll numbers with central academic enrollment registries."
            ]
        elif workspace in ["Employee & HR", "HR", "Employees"]:
            safe_recs = [
                "1. <b>Workforce Retention:</b> Address turnover drivers in high-attrition departments through stay interviews.",
                "2. <b>Compensation Equity:</b> Benchmark salary bands against market medians to maintain pay fairness.",
                "3. <b>HR Registry Integrity:</b> Audit onboarding and exit dates to eliminate negative employee tenure records."
            ]
        else: # Business & Finance
            safe_recs = [
                "1. <b>Cost Rationalization:</b> Audit departmental operating expenditure allocations and eliminate redundant subscriptions.",
                "2. <b>Budget Guardrails:</b> Enforce pre-approval variance workflows on line items exceeding planned envelopes.",
                "3. <b>Receivables Acceleration:</b> Institute automated aging collection workflows to optimize cash flow liquidity."
            ]
        for r in safe_recs:
            story.append(Paragraph(r, body_style))
            story.append(Spacer(1, 3))

    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=0.5, color=MUTED_TEXT, spaceAfter=8))
    story.append(Paragraph("Report generated by DataDoctor AI &bull; Intelligent Dataset Health & Multi-Industry Analytics Platform", subtitle_style))

    doc.build(story)
    return buf.getvalue()
