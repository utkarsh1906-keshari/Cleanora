
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import io
from datetime import datetime

# Core Legacy Engines (Protected)
from analyzer import (
    detect_outliers,
    check_date_consistency,
    check_ml_readiness
)
from cleaner import clean_dataset
from scoring import calculate_health_score

# Industry Modules
from store_analyzer import (
    map_store_columns,
    prepare_sales_data,
    summarize_sales,
    get_top_products,
    get_sales_trend,
    get_category_summary,
    get_customer_summary,
    get_inventory_summary
)
from healthcare_analyzer import (
    map_healthcare_columns,
    prepare_healthcare_data,
    summarize_healthcare,
    get_patient_analytics,
    get_appointment_analytics,
    get_doctor_department_analytics,
    get_admission_discharge_analytics,
    get_diagnosis_treatment_trends,
    check_healthcare_quality,
    clean_healthcare_dataset
)
from education_analyzer import (
    map_education_columns,
    prepare_education_data,
    summarize_education,
    get_performance_analytics,
    get_attendance_analytics,
    get_class_department_analytics,
    get_education_trends,
    check_education_quality,
    clean_education_dataset
)
from business_finance_analyzer import (
    map_business_finance_columns,
    prepare_business_finance_data,
    summarize_business_finance,
    get_revenue_analytics,
    get_expense_analytics,
    get_profit_loss_analytics,
    get_transaction_analytics,
    get_budget_analytics,
    get_cash_flow_analytics,
    check_business_finance_quality,
    clean_business_finance_dataset
)

# Seven Advanced AI Features
from ai_data_doctor import diagnose_dataset, preview_fix, apply_fix
from anomaly_detector import detect_smart_anomalies, create_anomaly_visual_chart
from what_if_simulator import (
    simulate_retail_scenario,
    simulate_healthcare_scenario,
    simulate_education_scenario,
    simulate_finance_scenario,
    create_simulation_comparison_chart
)
from dataset_chat import ask_dataset
from report_generator import generate_pdf_report
from history_manager import (
    save_health_session,
    get_health_history,
    clear_health_history,
    create_history_trend_chart
)
from ml_advisor import evaluate_ml_readiness

# Industry-Specific Recommendation Engine
from recommendation_engine import (
    generate_retail_recommendations,
    generate_healthcare_recommendations,
    generate_education_recommendations,
    generate_finance_recommendations,
    generate_hr_recommendations,
    get_industry_recommendations
)

# Human Resources & Workforce Analyzer
from hr_analyzer import (
    is_hr_dataset,
    map_hr_columns,
    prepare_hr_data,
    summarize_hr,
    get_hr_quality_check
)


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="DataDoctor AI - Multi-Industry Intelligence",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================================
# 2. MODERN SAAS FRONTEND STYLING (CSS)
# ==========================================

st.markdown("""
<style>
/* Main Dark SaaS Container */
.stApp {
    background-color: #0B1120;
    color: #F8FAFC;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

/* Header Box */
.header-box {
    background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 14px;
    padding: 30px;
    margin-bottom: 25px;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
}

.header-title {
    font-size: 32px;
    font-weight: 800;
    color: #F8FAFC;
    letter-spacing: -0.025em;
    margin-bottom: 8px;
}

.header-subtitle {
    font-size: 15px;
    color: #94A3B8;
    line-height: 1.5;
}

/* Card Containers */
.section-box {
    background-color: #16243A;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 22px;
    margin-bottom: 20px;
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.section-box:hover {
    border-color: #38BDF8;
}

/* KPI Cards */
.retail-kpi-card {
    background: linear-gradient(145deg, #16243A 0%, #1E293B 100%);
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 15px;
    text-align: left;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

.retail-kpi-label {
    font-size: 12px;
    font-weight: 600;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.retail-kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #38BDF8;
    letter-spacing: -0.02em;
    margin-bottom: 4px;
}

.retail-kpi-sub {
    font-size: 12px;
    color: #64748B;
}

/* Status Badges */
.pill-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-size: 11px;
    font-weight: 600;
}

.pill-success {
    background-color: rgba(52, 211, 153, 0.15);
    color: #34D399;
    border: 1px solid rgba(52, 211, 153, 0.3);
}

.pill-warning {
    background-color: rgba(251, 191, 36, 0.15);
    color: #FBBF24;
    border: 1px solid rgba(251, 191, 36, 0.3);
}

.pill-danger {
    background-color: rgba(248, 113, 113, 0.15);
    color: #F87171;
    border: 1px solid rgba(248, 113, 113, 0.3);
}

.pill-info {
    background-color: rgba(56, 189, 248, 0.15);
    color: #38BDF8;
    border: 1px solid rgba(56, 189, 248, 0.3);
}

/* Diagnostic Card */
.diag-card {
    background-color: #131E32;
    border-left: 4px solid #38BDF8;
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}
.diag-card-high {
    border-left-color: #F87171;
}
.diag-card-medium {
    border-left-color: #FBBF24;
}
.diag-card-low {
    border-left-color: #34D399;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0F172A;
    border-right: 1px solid #1E293B;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def get_outlier_count(results):
    if "Outliers" in results.columns:
        return int(results["Outliers"].sum())
    return 0

def create_health_gauge(score):
    if score >= 80:
        gauge_color = "#34D399"
    elif score >= 50:
        gauge_color = "#FBBF24"
    else:
        gauge_color = "#F87171"

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "/100", "font": {"size": 36, "color": "#E2E8F0"}},
            title={"text": "Dataset Health Score", "font": {"size": 16, "color": "#94A3B8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                "bar": {"color": gauge_color},
                "bgcolor": "#1E293B",
                "steps": [
                    {"range": [0, 50], "color": "rgba(248, 113, 113, 0.15)"},
                    {"range": [50, 80], "color": "rgba(251, 191, 36, 0.15)"},
                    {"range": [80, 100], "color": "rgba(52, 211, 153, 0.15)"}
                ]
            }
        )
    )
    fig.update_layout(
        paper_bgcolor="#0B1120",
        font_color="#E2E8F0",
        height=260,
        margin=dict(t=40, b=20, l=30, r=30)
    )
    return fig


def render_recommendation_cards(recommendations, industry_name):
    """
    Renders clean, modern, context-aware industry recommendation cards
    with priority badges, metric callouts, and strategic action buttons.
    """
    st.markdown(f"#### 💡 Strategic & Operational Recommendations ({industry_name})")
    st.caption(f"Context-aware recommendations generated from your uploaded {industry_name} dataset.")

    if not recommendations:
        st.info(f"No priority recommendations flagged for {industry_name} based on the current dataset metrics.")
        return

    for idx, rec in enumerate(recommendations):
        sev = rec.get("severity", "Info")
        if sev == "Critical":
            card_border = "#EF4444"
            badge_class = "pill-danger"
        elif sev == "Warning":
            card_border = "#F59E0B"
            badge_class = "pill-warning"
        elif sev == "Opportunity":
            card_border = "#10B981"
            badge_class = "pill-success"
        else:
            card_border = "#38BDF8"
            badge_class = "pill-info"

        with st.expander(f"{sev.upper()}: {rec['title']}", expanded=(idx < 2)):
            st.markdown(f"""
                <div style="background-color:#1E293B; border-left:4px solid {card_border}; border-radius:8px; padding:16px; margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                        <span class="pill-badge {badge_class}">{sev} Priority</span>
                        <span style="color:#94A3B8; font-size:12px; font-weight:600;">{rec['category']}</span>
                    </div>
                    <h4 style="color:#F8FAFC; margin-top:4px; margin-bottom:8px;">{rec['title']}</h4>
                    <p style="color:#E2E8F0; font-size:13px; margin-bottom:8px;">
                        <b>Key Metric / Observation:</b> <code style="color:#38BDF8;">{rec.get('metric_value', 'N/A')}</code>
                    </p>
                    <p style="color:#CBD5E1; font-size:13px; margin-bottom:12px; line-height:1.5;">
                        <b>Why This Matters:</b> {rec['insight']}
                    </p>
                    <div style="background-color:#0F172A; border-radius:6px; padding:10px 14px; margin-top:8px;">
                        <p style="color:#34D399; font-size:13px; margin:0; font-weight:500;">
                            <b>Recommended Strategic Action:</b> {rec['recommendation']}
                        </p>
                    </div>
                </div>
            """, unsafe_allow_html=True)


# ==========================================
# 4. INDUSTRY WORKSPACE SELECTION (LANDING PAGE)
# ==========================================

if "workspace" not in st.session_state:
    st.session_state["workspace"] = None

if st.session_state["workspace"] is None:
    st.markdown("""
        <div class="header-box" style="text-align:center; padding: 45px 25px;">
            <span class="pill-badge pill-info" style="margin-bottom:15px; font-size:12px;">
                ENTERPRISE MULTI-INDUSTRY DATA INTELLIGENCE
            </span>
            <div class="header-title" style="font-size:38px; margin-top:10px;">DataDoctor AI Platform</div>
            <div class="header-subtitle" style="font-size:17px; max-width:800px; margin: 0 auto 20px auto;">
                Next-generation dataset health diagnosis, smart anomaly detection, what-if scenario simulations, 
                conversational analytics, and executive reporting for modern data-driven enterprises.
            </div>
            <p style="color:#64748B; font-size:13px; margin:0;">
                Select your industry workspace below to access customized analytical engines and specialized intelligence workflows.
            </p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="section-box">
                <div style="font-size:32px; margin-bottom:8px;">🛒</div>
                <h3 style="color:#38BDF8; margin-top:0;">Retail Store Intelligence</h3>
                <p style="color:#94A3B8; font-size:14px; min-height:55px;">
                    Analyze sales velocity, gross margins, product profitability, customer buying patterns, and stock-out alerts.
                </p>
                <span class="pill-badge pill-success">Full Suite Active</span>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Retail Store Workspace", key="enter_retail", use_container_width=True):
            st.session_state["workspace"] = "Retail Store"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="section-box">
                <div style="font-size:32px; margin-bottom:8px;">🏥</div>
                <h3 style="color:#38BDF8; margin-top:0;">Healthcare Clinical Operations</h3>
                <p style="color:#94A3B8; font-size:14px; min-height:55px;">
                    Track patient flow, appointment fulfillment, bed occupancy, physician workloads, and EHR data integrity.
                </p>
                <span class="pill-badge pill-success">Full Suite Active</span>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Healthcare Workspace", key="enter_healthcare", use_container_width=True):
            st.session_state["workspace"] = "Healthcare"
            st.rerun()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
            <div class="section-box">
                <div style="font-size:32px; margin-bottom:8px;">🎓</div>
                <h3 style="color:#38BDF8; margin-top:0;">Education & Academic Analytics</h3>
                <p style="color:#94A3B8; font-size:14px; min-height:55px;">
                    Monitor student performance, grade distribution, attendance compliance, faculty workloads, and tutoring alerts.
                </p>
                <span class="pill-badge pill-success">Full Suite Active</span>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Education Workspace", key="enter_education", use_container_width=True):
            st.session_state["workspace"] = "Education"
            st.rerun()

    with col4:
        st.markdown("""
            <div class="section-box">
                <div style="font-size:32px; margin-bottom:8px;">💼</div>
                <h3 style="color:#38BDF8; margin-top:0;">Business & Corporate Finance</h3>
                <p style="color:#94A3B8; font-size:14px; min-height:55px;">
                    Audit revenue streams, expense cost centers, net profit margins, cash flow liquidity, and budget variances.
                </p>
                <span class="pill-badge pill-success">Full Suite Active</span>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Business & Finance Workspace", key="enter_finance", use_container_width=True):
            st.session_state["workspace"] = "Business & Finance"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
        <div style="background-color:#16243A; border:1px solid #334155; border-radius:12px; padding:20px; text-align:center;">
            <p style="color:#94A3B8; font-size:13px; font-weight:600; text-transform:uppercase; letter-spacing:0.05em; margin-bottom:12px;">
                Platform Capabilities Included Across All Workspaces
            </p>
            <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:10px;">
                <span class="pill-badge pill-info">🩺 AI Data Doctor</span>
                <span class="pill-badge pill-info">🔍 Anomaly Detective</span>
                <span class="pill-badge pill-info">🎛️ What-If Simulator</span>
                <span class="pill-badge pill-info">💬 Chat With Dataset</span>
                <span class="pill-badge pill-info">📑 Executive PDF Reports</span>
                <span class="pill-badge pill-info">📜 Health History Tracking</span>
                <span class="pill-badge pill-info">🤖 ML Readiness Advisor</span>
                <span class="pill-badge pill-info">🧹 Smart Cleaning Engine</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.stop()


# ==========================================
# 5. SIDEBAR & NAVIGATION
# ==========================================

with st.sidebar:
    st.markdown(
        "<h2 style='color:#38BDF8; margin-bottom:2px;'>DataDoctor AI</h2>"
        "<p style='color:#64748B; font-size:12px; margin-top:0;'>Intelligent Multi-Industry Analytics</p>",
        unsafe_allow_html=True
    )

    active_workspace = st.session_state.get("workspace", "Retail Store")
    st.info(f"Active Workspace: **{active_workspace}**")

    if st.button("🔄 Switch Workspace", use_container_width=True):
        st.session_state["workspace"] = None
        st.session_state.pop("cleaned_df", None)
        st.rerun()

    st.divider()

    st.subheader("Platform Navigation")

    nav_options = [
        f"📊 {active_workspace} Analytics",
        "🩺 AI Data Doctor",
        "🔍 Anomaly Detective",
        "🎛️ What-If Simulator",
        "💬 Chat With Dataset",
        "📑 Executive PDF Report",
        "📜 Health History",
        "🤖 ML Readiness Advisor",
        "🧹 Smart Cleaning & Data Export"
    ]

    selected_page = st.radio(
        "Navigation",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("Dataset Ingestion")

    if active_workspace == "Healthcare":
        sample_file_path = os.path.join(os.path.dirname(__file__), "healthcare_sample.csv")
        sample_btn_label = "📂 Load Sample Healthcare Data"
        sample_flag_key = "use_sample_healthcare"
    elif active_workspace == "Education":
        sample_file_path = os.path.join(os.path.dirname(__file__), "education_sample.csv")
        sample_btn_label = "📂 Load Sample Education Data"
        sample_flag_key = "use_sample_education"
    elif active_workspace == "Business & Finance":
        sample_file_path = os.path.join(os.path.dirname(__file__), "business_finance_sample.csv")
        sample_btn_label = "📂 Load Sample Financial Data"
        sample_flag_key = "use_sample_finance"
    else:
        sample_file_path = os.path.join(os.path.dirname(__file__), "retail_store_sample.csv")
        sample_btn_label = "📂 Load Sample Retail Data"
        sample_flag_key = "use_sample_retail"

    if os.path.exists(sample_file_path):
        if st.button(sample_btn_label, key="load_sample_btn", use_container_width=True):
            st.session_state[sample_flag_key] = True
            st.session_state["use_sample_employees"] = False
            st.session_state.pop("cleaned_df", None)
            st.rerun()

    emp_sample_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "datadoctor_sample_employees.csv")
    if not os.path.exists(emp_sample_path):
        emp_sample_path = os.path.join(os.path.dirname(__file__), "datadoctor_sample_employees.csv")

    if os.path.exists(emp_sample_path):
        if st.button("👥 Load Sample Employee HR Data", key="load_emp_sample_btn", use_container_width=True):
            st.session_state["use_sample_employees"] = True
            st.session_state["use_sample_healthcare"] = False
            st.session_state["use_sample_education"] = False
            st.session_state["use_sample_finance"] = False
            st.session_state["use_sample_retail"] = False
            st.session_state.pop("cleaned_df", None)
            st.rerun()

    uploaded_file = st.file_uploader(
        "Upload Custom CSV File",
        type=["csv"]
    )

    if uploaded_file is not None:
        st.session_state[sample_flag_key] = False
        st.session_state["use_sample_employees"] = False

    st.divider()
    st.markdown(
        f"<p style='color:#64748B; font-size:11px; text-align:center;'>"
        f"DataDoctor AI &bull; {active_workspace} Edition<br>Unified Analytics Platform"
        f"</p>",
        unsafe_allow_html=True
    )


# ==========================================
# 6. MAIN WORKSPACE HEADER
# ==========================================

workspace_headers = {
    "Healthcare": (
        "Healthcare Operations & Clinical Intelligence",
        "Patient Flow, Appointments, Admissions, Inpatient Tracking & EHR Diagnostics"
    ),
    "Education": (
        "Academic & Institutional Performance Intelligence",
        "Student Marks, Grade Distributions, Attendance Rates, Faculty Allocations & Curricular Health"
    ),
    "Business & Finance": (
        "Corporate Finance & Business Performance Intelligence",
        "P&L Statements, Operating Expenses, Net Margins, Cash Flow Liquidity & Budget Tracking"
    ),
    "Retail Store": (
        "Retail Store Operations & Commercial Intelligence",
        "Sales Velocity, Gross Margins, Product Profitability, Inventory Stock-Outs & Basket Analytics"
    )
}

header_info = workspace_headers.get(active_workspace, ("Enterprise Analytics", "Dataset Health & Decision Intelligence"))

st.markdown(f"""
    <div class="header-box">
        <div class="header-title">{header_info[0]}</div>
        <div class="header-subtitle">{header_info[1]}</div>
    </div>
""", unsafe_allow_html=True)


# ==========================================
# 7. CHECK DATASET UPLOAD & EMPTY STATE
# ==========================================

has_sample = (
    (st.session_state.get(sample_flag_key, False) and os.path.exists(sample_file_path))
    or (st.session_state.get("use_sample_employees", False) and os.path.exists(emp_sample_path))
)

if uploaded_file is None and not has_sample:
    st.info(f"Upload your {active_workspace} CSV dataset from the sidebar, or load the pre-configured sample dataset to begin.")
    
    col_e1, col_e2, col_e3 = st.columns(3)
    with col_e1:
        st.markdown(f"""
            <div class="section-box">
                <h4 style="color:#38BDF8; margin-top:0;">Domain Intelligence</h4>
                <p style="color:#94A3B8; font-size:13px;">
                    Dedicated KPIs, aggregations, and category attribution tailored for {active_workspace}.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_e2:
        st.markdown("""
            <div class="section-box">
                <h4 style="color:#38BDF8; margin-top:0;">AI Diagnostic Doctor</h4>
                <p style="color:#94A3B8; font-size:13px;">
                    Automated root cause identification, severity ranking, and interactive fix preview.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with col_e3:
        st.markdown("""
            <div class="section-box">
                <h4 style="color:#38BDF8; margin-top:0;">What-If & Executive Reports</h4>
                <p style="color:#94A3B8; font-size:13px;">
                    Simulate price/volume/capacity scenarios and download publication-ready executive PDFs.
                </p>
            </div>
        """, unsafe_allow_html=True)

    if os.path.exists(sample_file_path):
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button(f"🚀 Load Sample {active_workspace} Dataset to Explore", key="load_sample_hero", use_container_width=True):
            st.session_state[sample_flag_key] = True
            st.rerun()

    st.stop()


# ==========================================
# 8. READ DATASET & PREPARE DATA
# ==========================================

try:
    use_emp_sample = st.session_state.get("use_sample_employees", False) and os.path.exists(emp_sample_path)
    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        file_id = (uploaded_file.name, uploaded_file.size)
    elif use_emp_sample:
        df = pd.read_csv(emp_sample_path)
        dataset_name = "datadoctor_sample_employees.csv (Sample HR Dataset)"
        file_id = ("sample_employees", len(df))
    else:
        df = pd.read_csv(sample_file_path)
        dataset_name = os.path.basename(sample_file_path) + " (Sample Dataset)"
        file_id = (f"sample_{active_workspace.lower().replace(' ', '_').replace('&', 'and')}", len(df))

except Exception as error:
    st.error(f"Unable to read CSV file: {error}")
    st.stop()

if df.empty or len(df.columns) == 0:
    st.error("The selected dataset is empty.")
    st.stop()

# Base diagnostics
outlier_results = detect_outliers(df)
consistency_results = check_date_consistency(df)
ml_results = check_ml_readiness(df)
health_result = calculate_health_score(df, outlier_results, consistency_results)
original_score = health_result["score"]
original_missing = int(df.isnull().sum().sum())
original_duplicates = int(df.duplicated().sum())
original_outliers = get_outlier_count(outlier_results)
original_consistency = len(consistency_results)

# Domain Specific Preparation
store_summary = {}
health_summary = {}
edu_summary = {}
bf_summary = {}
hr_summary = {}
domain_summary = {}

is_hr_detected = is_hr_dataset(df)

if is_hr_detected:
    hr_mapping_info = map_hr_columns(df)
    current_mapping = hr_mapping_info["mapping"]
    hr_df, hr_mapping, hr_meta = prepare_hr_data(df, current_mapping)
    domain_summary = hr_summary = summarize_hr(hr_df)
    hr_quality = get_hr_quality_check(hr_df, hr_mapping)
elif active_workspace == "Healthcare":
    health_mapping_info = map_healthcare_columns(df)
    current_mapping = health_mapping_info["mapping"]
    health_df, health_mapping, health_meta = prepare_healthcare_data(df, current_mapping)
    domain_summary = health_summary = summarize_healthcare(health_df)
    health_quality = check_healthcare_quality(health_df, health_mapping)
elif active_workspace == "Education":
    edu_mapping_info = map_education_columns(df)
    current_mapping = edu_mapping_info["mapping"]
    edu_df, edu_mapping, edu_meta = prepare_education_data(df, current_mapping)
    domain_summary = edu_summary = summarize_education(edu_df)
    edu_quality = check_education_quality(edu_df, edu_mapping)
elif active_workspace == "Business & Finance":
    bf_mapping_info = map_business_finance_columns(df)
    current_mapping = bf_mapping_info["mapping"]
    bf_df, bf_mapping, bf_meta = prepare_business_finance_data(df, current_mapping)
    domain_summary = bf_summary = summarize_business_finance(bf_df)
    bf_quality = check_business_finance_quality(bf_df, bf_mapping)
else: # Retail Store
    mapping_info = map_store_columns(df)
    current_mapping = mapping_info["mapping"]
    store_df, store_mapping, store_meta = prepare_sales_data(df, current_mapping)
    domain_summary = store_summary = summarize_sales(store_df)

cleaned_df = st.session_state.get("cleaned_df", None)


# ==========================================
# 9B. HUMAN RESOURCES & WORKFORCE DASHBOARD (WHEN HR DATASET DETECTED)
# ==========================================

if is_hr_detected and (selected_page in ["Store Analytics", "Clinical Analytics", "Academic Analytics", "Financial Analytics", "Overview"] or "Analytics" in selected_page):
    st.info(
        "💼 **Workforce & HR Dataset Detected**: This dataset contains employee records (Employee IDs, Salaries, Hiring/Leaving dates). "
        "DataDoctor AI is analyzing it with the dedicated Human Resources & Workforce Intelligence Engine."
    )
    st.subheader("Human Resources & Workforce Intelligence")
    st.caption(f"Currently analyzing workforce records: {dataset_name} ({len(df):,} employees)")

    # HR KPI CARDS
    hr_k1, hr_k2, hr_k3, hr_k4 = st.columns(4)
    with hr_k1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Headcount</div>
                <div class="retail-kpi-value">{hr_summary['total_employees']:,}</div>
                <div class="retail-kpi-sub">Active: {hr_summary['active_employees']} | Departed: {hr_summary['departed_employees']}</div>
            </div>
        """, unsafe_allow_html=True)
    with hr_k2:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Annualized Payroll</div>
                <div class="retail-kpi-value">${hr_summary['total_payroll']:,.2f}</div>
                <div class="retail-kpi-sub">Avg Salary: ${hr_summary['avg_salary']:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
    with hr_k3:
        st.markdown(f"""
            <div class="retail-kpi-card profit">
                <div class="retail-kpi-label">Turnover Rate</div>
                <div class="retail-kpi-value">{hr_summary['turnover_rate']}%</div>
                <div class="retail-kpi-sub">Avg Tenure: {hr_summary['avg_tenure']} yrs</div>
            </div>
        """, unsafe_allow_html=True)
    with hr_k4:
        age_str = f"{hr_summary['avg_age']} yrs" if hr_summary['avg_age'] is not None else "N/A"
        st.markdown(f"""
            <div class="retail-kpi-card purple">
                <div class="retail-kpi-label">Average Employee Age</div>
                <div class="retail-kpi-value">{age_str}</div>
                <div class="retail-kpi-sub">Departments: {hr_summary['total_departments']}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    hr_tab1, hr_tab2, hr_tab3, hr_tab4, hr_tab5, hr_tab6 = st.tabs([
        "👥 Headcount & Departments",
        "💵 Compensation & Payroll",
        "🔄 Retention & Attrition",
        "🛡️ HR Data Quality",
        "💡 Workforce Recommendations",
        "📋 Employee Dataset Preview"
    ])

    with hr_tab1:
        st.markdown("#### Department Headcount & Staffing")
        dept_counts = hr_df["_clean_dept"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Headcount"]
        fig_dept = px.bar(dept_counts, x="Department", y="Headcount", text="Headcount", color="Headcount", color_continuous_scale="Tealgrn")
        fig_dept.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#0B1120", font_color="#E2E8F0")
        st.plotly_chart(fig_dept, use_container_width=True)

    with hr_tab2:
        st.markdown("#### Department Compensation & Payroll Breakdown")
        dept_sal = hr_df.groupby("_clean_dept")["_clean_salary"].agg(["mean", "sum", "count"]).reset_index()
        dept_sal.columns = ["Department", "Average Salary", "Total Payroll", "Staff Count"]
        fig_sal = px.bar(dept_sal, x="Department", y="Average Salary", text_auto=",.2f", color="Department")
        fig_sal.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#0B1120", font_color="#E2E8F0")
        st.plotly_chart(fig_sal, use_container_width=True)
        st.dataframe(dept_sal, use_container_width=True)

    with hr_tab3:
        st.markdown("#### Staff Retention & Attrition Analysis")
        ret_c1, ret_c2 = st.columns(2)
        with ret_c1:
            ret_df = pd.DataFrame({
                "Status": ["Active Staff", "Departed Staff"],
                "Count": [hr_summary["active_employees"], hr_summary["departed_employees"]]
            })
            fig_ret = px.pie(ret_df, names="Status", values="Count", color="Status", color_discrete_map={"Active Staff": "#10B981", "Departed Staff": "#EF4444"}, hole=0.4)
            fig_ret.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#0B1120", font_color="#E2E8F0")
            st.plotly_chart(fig_ret, use_container_width=True)
        with ret_c2:
            st.metric("Turnover Rate", f"{hr_summary['turnover_rate']}%")
            st.metric("Average Tenure", f"{hr_summary['avg_tenure']} years" if hr_summary['avg_tenure'] else "N/A")
            if hr_summary["departed_employees"] > 0 and "_clean_leave" in hr_df:
                st.markdown("**Departed Personnel:**")
                st.dataframe(hr_df[hr_df["_is_departed"]][["_clean_eid", "_clean_name", "_clean_dept", "_clean_salary"]], use_container_width=True)

    with hr_tab4:
        st.markdown("#### Human Resources Data Quality Audit")
        if not hr_quality["issues_table"].empty:
            st.dataframe(hr_quality["issues_table"], use_container_width=True)
        else:
            st.success("Zero HR data quality defects detected in employee records.")

    with hr_tab5:
        hr_recs = generate_hr_recommendations(hr_df, hr_mapping, hr_summary)
        render_recommendation_cards(hr_recs, "Human Resources & Workforce")

    with hr_tab6:
        st.markdown("#### Raw Employee Dataset Preview")
        st.dataframe(df, use_container_width=True)


# ==========================================
# 10. RETAIL STORE DASHBOARD
# ==========================================

elif not is_hr_detected and active_workspace == "Retail Store" and (selected_page in ["Store Analytics", "Overview"] or "Analytics" in selected_page):

    st.subheader("Retail Store Analytics Dashboard")
    st.caption(f"Currently analyzing: {dataset_name} ({len(df):,} records)")

    # COLUMN MAPPING & READINESS STATUS
    missing_crit = mapping_info["missing_critical"]
    if missing_crit:
        st.warning(
            f"⚠️ **Notice: Missing Required Column(s)**: {', '.join(missing_crit)}. "
            "Some store metrics may be estimated or incomplete. Check column names or view Data Diagnosis."
        )

    with st.expander("🔍 View Detected Store Columns & Data Readiness"):
        col_map_cols = st.columns(4)
        with col_map_cols[0]:
            st.markdown("**Sales Fields**")
            st.markdown(f"- Product: `{store_mapping.get('product', 'Not detected')}`")
            st.markdown(f"- Quantity: `{store_mapping.get('quantity', 'Defaulted to 1')}`")
            st.markdown(f"- Price: `{store_mapping.get('price', 'Derived')}`")
            st.markdown(f"- Revenue: `{store_mapping.get('revenue', 'Derived')}`")
        with col_map_cols[1]:
            st.markdown("**Financial Fields**")
            st.markdown(f"- Cost: `{store_mapping.get('cost', 'Not detected (Profit disabled)')}`")
            st.markdown(f"- Profit Mode: `{'Active' if store_meta['has_cost'] else 'Unavailable'}`")
        with col_map_cols[2]:
            st.markdown("**Segmentation**")
            st.markdown(f"- Category: `{store_mapping.get('category', 'Not detected (General)')}`")
            st.markdown(f"- Customer: `{store_mapping.get('customer', 'Not detected')}`")
        with col_map_cols[3]:
            st.markdown("**Operations**")
            st.markdown(f"- Date: `{store_mapping.get('date', 'Not detected')}`")
            st.markdown(f"- Stock: `{store_mapping.get('stock', 'Not detected')}`")

    # 1. SALES OVERVIEW KPI CARDS
    st.markdown("### Executive Sales & Business Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Revenue</div>
                <div class="retail-kpi-value">${store_summary['total_revenue']:,.2f}</div>
                <div class="retail-kpi-sub">Avg Order: ${store_summary['avg_order_value']:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Units Sold</div>
                <div class="retail-kpi-value">{store_summary['total_units']:,}</div>
                <div class="retail-kpi-sub">Avg Unit Price: ${store_summary['avg_unit_price']:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Transactions</div>
                <div class="retail-kpi-value">{store_summary['total_transactions']:,}</div>
                <div class="retail-kpi-sub">Unique Products: {store_summary['unique_products']:,}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        if store_summary['has_cost']:
            st.markdown(f"""
                <div class="retail-kpi-card profit">
                    <div class="retail-kpi-label">Gross Profit</div>
                    <div class="retail-kpi-value">${store_summary['total_profit']:,.2f}</div>
                    <div class="retail-kpi-sub">Margin: {store_summary['profit_margin']}% | Cost: ${store_summary['total_cost']:,.2f}</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="retail-kpi-card warning">
                    <div class="retail-kpi-label">Dataset Health</div>
                    <div class="retail-kpi-value">{original_score}/100</div>
                    <div class="retail-kpi-sub">Add cost column for profit analytics</div>
                </div>
            """, unsafe_allow_html=True)

    st.divider()


    # 2. SPECIALIZED RETAIL ANALYSIS TABS
    tab_trends, tab_profit, tab_products, tab_categories, tab_inventory, tab_customers, tab_quality, tab_recs = st.tabs([
        "📈 Sales Trends",
        "💰 Profit Analysis",
        "🏆 Top-Selling Products",
        "🏷️ Category Performance",
        "📦 Inventory Health",
        "👥 Customer Insights",
        "🛡️ Data Quality Check",
        "💡 Strategic Recommendations"
    ])

    # TAB 1: SALES TRENDS
    with tab_trends:
        st.markdown("#### Sales & Revenue Trends Over Time")
        if store_meta["has_date"]:
            trend_ctrl1, trend_ctrl2 = st.columns([1, 2])
            with trend_ctrl1:
                trend_view = st.radio(
                    "Trend Frequency:",
                    ["Daily Performance", "Monthly Performance"],
                    horizontal=True,
                    key="trend_freq_radio"
                )
            freq_arg = "monthly" if "Monthly" in trend_view else "daily"
            trend_df = get_sales_trend(store_df, store_mapping, freq=freq_arg)

            if not trend_df.empty:
                fig_trend = px.line(
                    trend_df,
                    x="Period",
                    y="Revenue",
                    markers=True,
                    title=f"Store Revenue Over Time ({freq_arg.capitalize()})",
                    color_discrete_sequence=["#38BDF8"]
                )
                if store_summary["has_cost"] and "Profit" in trend_df.columns:
                    fig_trend.add_scatter(
                        x=trend_df["Period"],
                        y=trend_df["Profit"],
                        mode="lines+markers",
                        name="Gross Profit",
                        line=dict(color="#34D399", dash="dot")
                    )
                fig_trend.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    hovermode="x unified",
                    xaxis_title="Date",
                    yaxis_title="Revenue ($)"
                )
                st.plotly_chart(fig_trend, use_container_width=True)

                col_t1, col_t2 = st.columns(2)
                with col_t1:
                    fig_vol = px.bar(
                        trend_df,
                        x="Period_Label",
                        y="Units Sold",
                        title="Units Sold Volume",
                        color_discrete_sequence=["#06B6D4"]
                    )
                    fig_vol.update_layout(
                        paper_bgcolor="#0B1120",
                        plot_bgcolor="#16243A",
                        font_color="#E2E8F0",
                        xaxis_title=""
                    )
                    st.plotly_chart(fig_vol, use_container_width=True)

                with col_t2:
                    if "_day_name" in store_df.columns:
                        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
                        dow_df = store_df.groupby("_day_name")["_clean_revenue"].sum().reindex(dow_order).dropna().reset_index()
                        fig_dow = px.bar(
                            dow_df,
                            x="_day_name",
                            y="_clean_revenue",
                            title="Revenue by Day of Week",
                            color_discrete_sequence=["#6366F1"]
                        )
                        fig_dow.update_layout(
                            paper_bgcolor="#0B1120",
                            plot_bgcolor="#16243A",
                            font_color="#E2E8F0",
                            xaxis_title="",
                            yaxis_title="Revenue ($)"
                        )
                        st.plotly_chart(fig_dow, use_container_width=True)
            else:
                st.info("No valid date records could be parsed from the date column.")
        else:
            st.info("📅 **Date column not detected**: Upload a dataset with a date column (e.g. `Date`, `Order_Date`, `Sale_Date`) to unlock daily and monthly sales trends.")

    # TAB 2: PROFIT ANALYSIS
    with tab_profit:
        st.markdown("#### Store Profitability & Margins")
        if store_summary["has_cost"]:
            p1, p2, p3 = st.columns(3)
            p1.metric("Total Revenue", f"${store_summary['total_revenue']:,.2f}")
            p2.metric("Total Cost (COGS)", f"${store_summary['total_cost']:,.2f}")
            p3.metric("Gross Profit", f"${store_summary['total_profit']:,.2f}", f"{store_summary['profit_margin']}% Margin")

            waterfall_df = pd.DataFrame({
                "Component": ["Revenue", "Total Cost", "Gross Profit"],
                "Amount": [store_summary['total_revenue'], store_summary['total_cost'], store_summary['total_profit']]
            })
            fig_profit = px.bar(
                waterfall_df,
                x="Component",
                y="Amount",
                color="Component",
                title="Revenue vs Cost vs Profit Breakdown",
                color_discrete_map={"Revenue": "#38BDF8", "Total Cost": "#F87171", "Gross Profit": "#34D399"}
            )
            fig_profit.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#16243A",
                font_color="#E2E8F0",
                showlegend=False
            )
            st.plotly_chart(fig_profit, use_container_width=True)

            if store_meta["has_category"]:
                cat_prof_df = get_category_summary(store_df)
                if not cat_prof_df.empty and "Margin %" in cat_prof_df.columns:
                    st.markdown("##### Category Profitability Ranking")
                    fig_cat_prof = px.bar(
                        cat_prof_df,
                        x="Category",
                        y="Margin %",
                        color="Margin %",
                        title="Profit Margin % by Department",
                        color_continuous_scale="Tealgrn"
                    )
                    fig_cat_prof.update_layout(
                        paper_bgcolor="#0B1120",
                        plot_bgcolor="#16243A",
                        font_color="#E2E8F0"
                    )
                    st.plotly_chart(fig_cat_prof, use_container_width=True)
        else:
            st.info("💡 **Cost Data Unavailable**: To analyze profit and margins, include a cost column in your dataset (such as `Unit_Cost`, `Cost`, `Cost_Price`, or `COGS`).")
            st.write("When available, DataDoctor AI will compute per-item cost, total expenses, profit margins, and identify your most and least profitable products.")

    # TAB 3: BEST-SELLING PRODUCTS
    with tab_products:
        st.markdown("#### Top-Selling Products Performance")
        prod_c1, prod_c2 = st.columns([1, 1])
        with prod_c1:
            sort_metric = st.selectbox(
                "Rank Products By:",
                ["Total Revenue ($)", "Units Sold (Volume)"],
                key="prod_sort_select"
            )
        with prod_c2:
            top_limit = st.slider("Show Top Products:", 5, 25, 10, key="prod_limit_slider")

        sort_arg = "revenue" if "Revenue" in sort_metric else "quantity"
        top_prods_df = get_top_products(store_df, store_mapping, top_n=top_limit, sort_by=sort_arg)

        if not top_prods_df.empty:
            chart_y_metric = "Total Revenue" if sort_arg == "revenue" else "Units Sold"
            fig_top = px.bar(
                top_prods_df,
                x=chart_y_metric,
                y="Product",
                orientation="h",
                title=f"Top {top_limit} Products Ranked by {chart_y_metric}",
                color=chart_y_metric,
                color_continuous_scale="Viridis"
            )
            fig_top.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#16243A",
                font_color="#E2E8F0",
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig_top, use_container_width=True)

            st.markdown("##### Detailed Product Performance")
            st.dataframe(top_prods_df, use_container_width=True)
        else:
            st.warning("No products could be extracted. Please check the Product column.")

    # TAB 4: CATEGORY PERFORMANCE
    with tab_categories:
        st.markdown("#### Category & Department Breakdown")
        if store_meta["has_category"]:
            cat_df = get_category_summary(store_df)
            if not cat_df.empty:
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    fig_cat_pie = px.pie(
                        cat_df,
                        names="Category",
                        values="Revenue",
                        hole=0.45,
                        title="Revenue Contribution by Category",
                        color_discrete_sequence=px.colors.qualitative.Prism
                    )
                    fig_cat_pie.update_layout(
                        paper_bgcolor="#0B1120",
                        plot_bgcolor="#16243A",
                        font_color="#E2E8F0"
                    )
                    st.plotly_chart(fig_cat_pie, use_container_width=True)

                with col_c2:
                    fig_cat_bar = px.bar(
                        cat_df,
                        x="Category",
                        y="Units Sold",
                        color="Units Sold",
                        title="Units Sold by Category",
                        color_continuous_scale="Blues"
                    )
                    fig_cat_bar.update_layout(
                        paper_bgcolor="#0B1120",
                        plot_bgcolor="#16243A",
                        font_color="#E2E8F0",
                        xaxis_title=""
                    )
                    st.plotly_chart(fig_cat_bar, use_container_width=True)

                st.markdown("##### Department Summary Table")
                st.dataframe(cat_df, use_container_width=True)
            else:
                st.info("No category records available.")
        else:
            st.info("🏷️ **Category column not detected**: Include a column such as `Category`, `Department`, or `Product_Category` to see category revenue distribution.")

    # TAB 5: INVENTORY HEALTH
    with tab_inventory:
        st.markdown("#### Stock & Inventory Health Management")
        if store_meta["has_stock"]:
            inv_summary = get_inventory_summary(store_df, store_mapping)
            if inv_summary is not None:
                inv_kpi1, inv_kpi2, inv_kpi3, inv_kpi4 = st.columns(4)
                inv_kpi1.metric("Total Stock Units", f"{inv_summary['total_stock_units']:,}")
                inv_kpi2.metric("Estimated Stock Value", f"${inv_summary['total_stock_value']:,.2f}")
                inv_kpi3.metric("Low-Stock Alerts (<=10)", inv_summary['low_stock_count'], delta="Reorder needed" if inv_summary['low_stock_count'] > 0 else None, delta_color="inverse")
                inv_kpi4.metric("Out of Stock (0)", inv_summary['out_of_stock_count'], delta="Urgent" if inv_summary['out_of_stock_count'] > 0 else "All available", delta_color="inverse")

                if inv_summary['out_of_stock_count'] > 0:
                    st.error(f"🚨 **Critical Stock Alert**: {inv_summary['out_of_stock_count']} product(s) are completely out of stock!")
                    st.dataframe(inv_summary['out_of_stock_products'], use_container_width=True)

                if inv_summary['low_stock_count'] > 0:
                    st.warning(f"⚠️ **Low Stock Warning**: {inv_summary['low_stock_count']} product(s) have 10 or fewer units remaining.")
                    st.dataframe(inv_summary['low_stock_products'], use_container_width=True)

                st.markdown("##### Full Store Inventory Status")
                st.dataframe(inv_summary['inventory_table'], use_container_width=True)
        else:
            st.info("📦 **Stock column not detected**: Include an inventory column (e.g. `Stock_Level`, `Inventory`, `Stock`, `Quantity_In_Stock`) to enable low-stock alerts, out-of-stock monitoring, and inventory valuation.")

    # TAB 6: CUSTOMER INSIGHTS
    with tab_customers:
        st.markdown("#### Customer Purchasing Patterns")
        if store_meta["has_customer"]:
            cust_summary = get_customer_summary(store_df, store_mapping)
            if cust_summary is not None:
                ckpi1, ckpi2, ckpi3, ckpi4 = st.columns(4)
                ckpi1.metric("Unique Customers", f"{cust_summary['total_customers']:,}")
                ckpi2.metric("Repeat Customers (>1 order)", f"{cust_summary['repeat_customers']:,}")
                ckpi3.metric("Repeat Customer Rate", f"{cust_summary['repeat_rate']}%")
                ckpi4.metric("Avg Customer Value", f"${cust_summary['avg_customer_value']:,.2f}")

                st.markdown("##### Top Customers Leaderboard")
                st.dataframe(cust_summary["top_customers"], use_container_width=True)
        else:
            st.info("👥 **Customer column not detected**: Include a customer identifier column (e.g. `Customer_ID`, `Customer_Name`, `Client_ID`) to analyze customer frequency, repeat purchasing rates, and top buyers.")

    # TAB 7: DATA QUALITY CHECK
    with tab_quality:
        st.markdown("#### Dataset Quality Snapshot (DataDoctor Engine)")
        dq1, dq2 = st.columns([1, 1])
        with dq1:
            gauge = create_health_gauge(original_score)
            st.plotly_chart(gauge, use_container_width=True)
        with dq2:
            st.markdown("##### Detected Quality Issues")
            st.markdown(f"- **Missing Values**: `{original_missing:,}` cells")
            st.markdown(f"- **Duplicate Rows**: `{original_duplicates:,}` rows")
            st.markdown(f"- **Numeric Outliers**: `{original_outliers:,}` detected points")
            st.markdown(f"- **Date Inconsistencies**: `{original_consistency:,}` issues")

            if original_score >= 80:
                st.success("Good Dataset Health: Few quality issues detected.")
            elif original_score >= 50:
                st.warning("Moderate Dataset Health: Cleaning recommended for reliable analytics.")
            else:
                st.error("Poor Dataset Health: Significant issues detected. Cleaning required.")

            st.caption("Use 'Data Diagnosis' or 'Smart Cleaning' from the sidebar for in-depth quality correction.")

    # TAB 8: STRATEGIC RETAIL RECOMMENDATIONS
    with tab_recs:
        retail_recs = generate_retail_recommendations(store_df, store_mapping, store_summary)
        render_recommendation_cards(retail_recs, "Retail Store")

    st.divider()


    # RAW DATA PREVIEW
    st.subheader("Store Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    with st.expander("View Column Types & Summary"):
        column_info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str).values,
            "Unique Values": df.nunique().values,
            "Missing Values": df.isnull().sum().values
        })
        st.dataframe(column_info, use_container_width=True)


# ==========================================
# 10B. HEALTHCARE WORKSPACE DASHBOARD
# ==========================================

elif not is_hr_detected and active_workspace == "Healthcare" and (selected_page in ["Clinical Analytics", "Overview"] or "Analytics" in selected_page):

    st.subheader("Healthcare Operations & Clinical Analytics")
    st.caption(f"Currently analyzing: {dataset_name} ({len(df):,} patient records)")

    # 1. COLUMN MAPPING & READINESS
    missing_crit = health_mapping_info["missing_critical"]
    if missing_crit:
        st.warning(
            f"⚠️ **Notice: Missing Required Clinical Column(s)**: {', '.join(missing_crit)}. "
            "Some healthcare indicators may be estimated or incomplete. Check column names or view Data Diagnosis."
        )

    with st.expander("🔍 View Detected Healthcare Columns & Interactive Mapping"):
        h_col1, h_col2, h_col3, h_col4 = st.columns(4)
        with h_col1:
            st.markdown("**Patient Demographics**")
            st.markdown(f"- Patient ID: `{health_mapping.get('patient_id', 'Auto-generated')}`")
            st.markdown(f"- Patient Name: `{health_mapping.get('patient_name', 'Not detected')}`")
            st.markdown(f"- Age: `{health_mapping.get('age', 'Not detected')}`")
            st.markdown(f"- Gender: `{health_mapping.get('gender', 'Not detected')}`")
        with h_col2:
            st.markdown("**Clinical Context**")
            st.markdown(f"- Department: `{health_mapping.get('department', 'General')}`")
            st.markdown(f"- Doctor: `{health_mapping.get('doctor', 'Unassigned')}`")
            st.markdown(f"- Diagnosis: `{health_mapping.get('diagnosis', 'Not detected')}`")
            st.markdown(f"- Treatment: `{health_mapping.get('treatment', 'Not detected')}`")
        with h_col3:
            st.markdown("**Appointments & Visits**")
            st.markdown(f"- Appointment Date: `{health_mapping.get('appointment_date', 'Not detected')}`")
            st.markdown(f"- Visit Status: `{health_mapping.get('appointment_status', 'Defaulted Completed')}`")
        with h_col4:
            st.markdown("**Admissions & Financials**")
            st.markdown(f"- Admission Date: `{health_mapping.get('admission_date', 'Not detected')}`")
            st.markdown(f"- Discharge Date: `{health_mapping.get('discharge_date', 'Not detected')}`")
            st.markdown(f"- Billing / Cost: `{health_mapping.get('billing', 'Not detected')}`")

    # 2. SECTION A: HEALTHCARE OVERVIEW KPI CARDS
    st.markdown("### Executive Healthcare & Clinical Overview")
    hkpi1, hkpi2, hkpi3, hkpi4 = st.columns(4)

    with hkpi1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Patients</div>
                <div class="retail-kpi-value">{health_summary['unique_patients']:,}</div>
                <div class="retail-kpi-sub">Total Encounters: {health_summary['total_records']:,}</div>
            </div>
        """, unsafe_allow_html=True)

    with hkpi2:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Appointments</div>
                <div class="retail-kpi-value">{health_summary['total_appointments']:,}</div>
                <div class="retail-kpi-sub">Doctors: {health_summary['total_doctors']} | Depts: {health_summary['total_departments']}</div>
            </div>
        """, unsafe_allow_html=True)

    with hkpi3:
        st.markdown(f"""
            <div class="retail-kpi-card profit">
                <div class="retail-kpi-label">Hospital Admissions</div>
                <div class="retail-kpi-value">{health_summary['total_admissions']:,}</div>
                <div class="retail-kpi-sub">Discharges: {health_summary['total_discharges']:,} | Inpatients: {health_summary['current_inpatients']}</div>
            </div>
        """, unsafe_allow_html=True)

    with hkpi4:
        age_label = f"{health_summary['avg_age']} yrs" if health_summary['avg_age'] is not None else "N/A"
        alos_label = f"{health_summary['alos']} days" if health_summary['alos'] is not None else "N/A"
        st.markdown(f"""
            <div class="retail-kpi-card purple">
                <div class="retail-kpi-label">Avg Patient Age</div>
                <div class="retail-kpi-value">{age_label}</div>
                <div class="retail-kpi-sub">Avg Stay (ALOS): {alos_label}</div>
            </div>
        """, unsafe_allow_html=True)

    st.divider()


    # 3. SPECIALIZED HEALTHCARE ANALYSIS TABS
    tab_pat, tab_apt, tab_doc, tab_adm, tab_diag, tab_qual, tab_recs, tab_prev = st.tabs([
        "👥 Patient Analytics",
        "📅 Appointment Analytics",
        "🩺 Doctor & Department Analytics",
        "🏥 Admission & Discharge Analytics",
        "🔬 Diagnosis & Treatment Trends",
        "🛡️ Healthcare Data Quality",
        "💡 Clinical Recommendations",
        "📋 Dataset Preview"
    ])

    # TAB B: PATIENT ANALYTICS
    with tab_pat:
        st.markdown("#### Patient Demographics & Cohort Analysis")
        pat_data = get_patient_analytics(health_df)

        col_p1, col_p2 = st.columns([1, 1])
        with col_p1:
            if not pat_data["valid_ages_df"].empty:
                fig_age = px.histogram(
                    pat_data["valid_ages_df"],
                    x="_clean_age",
                    nbins=20,
                    title="Patient Age Distribution",
                    color_discrete_sequence=["#38BDF8"]
                )
                fig_age.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    xaxis_title="Patient Age (Years)",
                    yaxis_title="Patient Count"
                )
                st.plotly_chart(fig_age, use_container_width=True)
            else:
                st.info("Age data not detected or all values invalid.")

            if not pat_data["age_groups"].empty:
                fig_age_grp = px.bar(
                    pat_data["age_groups"],
                    x="Age Group",
                    y="Patient Count",
                    title="Patient Cohorts by Life Stage",
                    color="Patient Count",
                    color_continuous_scale="Blues"
                )
                fig_age_grp.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    xaxis_title=""
                )
                st.plotly_chart(fig_age_grp, use_container_width=True)

        with col_p2:
            if not pat_data["gender_distribution"].empty:
                fig_gen = px.pie(
                    pat_data["gender_distribution"],
                    names="Gender",
                    values="Count",
                    hole=0.45,
                    title="Patient Gender Breakdown",
                    color_discrete_sequence=["#38BDF8", "#F472B6", "#A78BFA"]
                )
                fig_gen.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0"
                )
                st.plotly_chart(fig_gen, use_container_width=True)

            if not pat_data["patient_type"].empty:
                st.markdown("##### New vs Repeat Patients")
                kpi_p1, kpi_p2 = st.columns(2)
                kpi_p1.metric("Repeat Patient Rate", f"{pat_data['repeat_rate']}%")
                kpi_p2.metric("Total Unique Patients", f"{health_summary['unique_patients']:,}")

                fig_type = px.bar(
                    pat_data["patient_type"],
                    x="Patient Type",
                    y="Count",
                    color="Patient Type",
                    title="Patient Visit Frequency Pattern",
                    color_discrete_map={"New Patients (1 Visit)": "#38BDF8", "Repeat Patients (2+ Visits)": "#34D399"}
                )
                fig_type.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    showlegend=False
                )
                st.plotly_chart(fig_type, use_container_width=True)

    # TAB C: APPOINTMENT ANALYTICS
    with tab_apt:
        st.markdown("#### Appointment Operations & Fulfillment")
        apt_data = get_appointment_analytics(health_df)

        akpi1, akpi2, akpi3, akpi4 = st.columns(4)
        akpi1.metric("Completed Appointments", f"{apt_data['completed']:,}", f"{apt_data['completion_rate']}% Completed")
        akpi2.metric("Cancelled Appointments", f"{apt_data['cancelled']:,}", f"{apt_data['cancellation_rate']}% Cancelled", delta_color="inverse")
        akpi3.metric("Pending / Scheduled", f"{apt_data['pending']:,}")
        akpi4.metric("No-Shows", f"{apt_data['no_show']:,}")

        # Trend Frequency
        if not apt_data["trend_df"].empty:
            apt_view = st.radio("Appointment Trend View:", ["Daily Volume", "Monthly Volume"], horizontal=True, key="apt_freq_radio")
            freq_key = "monthly" if "Monthly" in apt_view else "daily"
            apt_trend_recalc = get_appointment_analytics(health_df, freq=freq_key)["trend_df"]

            fig_apt_trend = px.line(
                apt_trend_recalc,
                x="Period",
                y="Total_Appointments",
                markers=True,
                title=f"Appointment Volume Over Time ({freq_key.capitalize()})",
                color_discrete_sequence=["#38BDF8"]
            )
            if "Completed" in apt_trend_recalc.columns:
                fig_apt_trend.add_scatter(
                    x=apt_trend_recalc["Period"],
                    y=apt_trend_recalc["Completed"],
                    mode="lines+markers",
                    name="Completed",
                    line=dict(color="#34D399", dash="dash")
                )
            if "Cancelled" in apt_trend_recalc.columns and apt_trend_recalc["Cancelled"].sum() > 0:
                fig_apt_trend.add_scatter(
                    x=apt_trend_recalc["Period"],
                    y=apt_trend_recalc["Cancelled"],
                    mode="lines+markers",
                    name="Cancelled",
                    line=dict(color="#F87171", dash="dot")
                )
            fig_apt_trend.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#16243A",
                font_color="#E2E8F0",
                hovermode="x unified",
                xaxis_title="Date",
                yaxis_title="Appointments"
            )
            st.plotly_chart(fig_apt_trend, use_container_width=True)

        col_apt1, col_apt2 = st.columns(2)
        with col_apt1:
            if not apt_data["doctor_appointments"].empty:
                fig_doc_apt = px.bar(
                    apt_data["doctor_appointments"].head(10),
                    x="Appointments",
                    y="Doctor",
                    orientation="h",
                    title="Top Doctors by Appointment Volume",
                    color="Appointments",
                    color_continuous_scale="Teal"
                )
                fig_doc_apt.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_doc_apt, use_container_width=True)

        with col_apt2:
            if not apt_data["department_appointments"].empty:
                fig_dept_apt = px.bar(
                    apt_data["department_appointments"],
                    x="Department",
                    y="Appointments",
                    title="Appointments by Department",
                    color="Appointments",
                    color_continuous_scale="Purples"
                )
                fig_dept_apt.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0"
                )
                st.plotly_chart(fig_dept_apt, use_container_width=True)

    # TAB D: DOCTOR & DEPARTMENT ANALYTICS
    with tab_doc:
        st.markdown("#### Clinical Staff Workload & Department Comparison")
        doc_dept = get_doctor_department_analytics(health_df)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            if not doc_dept["department_summary"].empty:
                fig_dept_pie = px.pie(
                    doc_dept["department_summary"],
                    names="Department",
                    values="Patients",
                    hole=0.45,
                    title="Patient Volume Share by Department",
                    color_discrete_sequence=px.colors.qualitative.Prism
                )
                fig_dept_pie.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0"
                )
                st.plotly_chart(fig_dept_pie, use_container_width=True)

        with col_d2:
            if not doc_dept["doctor_summary"].empty:
                fig_doc_enc = px.bar(
                    doc_dept["doctor_summary"].head(10),
                    x="Total_Encounters",
                    y="Doctor",
                    orientation="h",
                    title="Top Care Providers by Patient Encounters",
                    color="Department"
                )
                fig_doc_enc.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_doc_enc, use_container_width=True)

        if not doc_dept["department_summary"].empty:
            st.markdown("##### Comprehensive Department Performance Matrix")
            st.dataframe(doc_dept["department_summary"], use_container_width=True)

    # TAB E: ADMISSION & DISCHARGE ANALYTICS
    with tab_adm:
        st.markdown("#### Inpatient Flow, Admissions & Length of Stay (ALOS)")
        adm_data = get_admission_discharge_analytics(health_df)

        ad1, ad2, ad3, ad4 = st.columns(4)
        ad1.metric("Total Admissions", f"{health_summary['total_admissions']:,}")
        ad2.metric("Total Discharges", f"{health_summary['total_discharges']:,}")
        ad3.metric("Current Inpatients", f"{adm_data['inpatient_count']:,}", delta="Under Care" if adm_data['inpatient_count'] > 0 else "None")
        ad4.metric("Average Length of Stay", f"{adm_data['avg_los']} Days" if adm_data['avg_los'] is not None else "N/A", f"Median: {adm_data['median_los']} d" if adm_data['median_los'] is not None else None)

        if not adm_data["flow_trend"].empty:
            fig_flow = px.line(
                adm_data["flow_trend"],
                x="Period",
                y=["Admissions", "Discharges"],
                markers=True,
                title="Hospital Inflow vs Outflow Trends",
                color_discrete_map={"Admissions": "#38BDF8", "Discharges": "#34D399"}
            )
            fig_flow.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#16243A",
                font_color="#E2E8F0",
                hovermode="x unified"
            )
            st.plotly_chart(fig_flow, use_container_width=True)

        col_adm1, col_adm2 = st.columns(2)
        with col_adm1:
            if not adm_data["los_categories"].empty:
                fig_los = px.bar(
                    adm_data["los_categories"],
                    x="Stay Duration Category",
                    y="Patient Count",
                    color="Stay Duration Category",
                    title="Length of Stay (LOS) Categorization",
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                fig_los.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    showlegend=False
                )
                st.plotly_chart(fig_los, use_container_width=True)

        with col_adm2:
            if adm_data["inpatient_count"] > 0:
                st.markdown("##### 🛏️ Current Inpatients Under Active Hospital Care")
                st.dataframe(adm_data["inpatients_table"], use_container_width=True)
            else:
                st.info("No active inpatients currently admitted (all admitted patients have recorded discharges).")

    # TAB F: DIAGNOSIS & TREATMENT TRENDS
    with tab_diag:
        st.markdown("#### Clinical Diagnoses & Treatment Distribution")
        diag_data = get_diagnosis_treatment_trends(health_df)

        col_dg1, col_dg2 = st.columns(2)
        with col_dg1:
            if not diag_data["top_diagnoses"].empty:
                fig_top_diag = px.bar(
                    diag_data["top_diagnoses"].head(10),
                    x="Case Count",
                    y="Diagnosis",
                    orientation="h",
                    title="Top 10 Most Frequently Recorded Diagnoses",
                    color="Case Count",
                    color_continuous_scale="Viridis"
                )
                fig_top_diag.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_top_diag, use_container_width=True)

        with col_dg2:
            if not diag_data["top_treatments"].empty:
                fig_top_treat = px.bar(
                    diag_data["top_treatments"].head(10),
                    x="Frequency",
                    y="Treatment",
                    orientation="h",
                    title="Top Treatment & Care Protocols Administered",
                    color="Frequency",
                    color_continuous_scale="Blues"
                )
                fig_top_treat.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig_top_treat, use_container_width=True)

        if not diag_data["dept_diag_matrix"].empty:
            st.markdown("##### Department vs Diagnosis Cross-Tabulation Matrix")
            st.dataframe(diag_data["dept_diag_matrix"], use_container_width=True)

    # TAB G: HEALTHCARE DATA QUALITY
    with tab_qual:
        st.markdown("#### Clinical EHR Data Integrity & Quality Audit")
        hq1, hq2 = st.columns([1, 1])

        with hq1:
            gauge_hq = create_health_gauge(health_quality["health_score"])
            st.plotly_chart(gauge_hq, use_container_width=True)

        with hq2:
            st.markdown("##### Clinical Quality Assessment Summary")
            st.markdown(f"- **Identified Quality Anomalies**: `{health_quality['issue_count']}` distinct issue types")
            st.markdown(f"- **Duplicate Patient Rows**: `{health_quality['dup_count']}` records")
            
            # Count invalid ages and dates directly
            invalid_age_n = (health_df["_clean_age"].isna() | (health_df["_clean_age"] < 0) | (health_df["_clean_age"] > 120)).sum()
            inconsistent_date_n = health_df["_date_inconsistent"].sum() if "_date_inconsistent" in health_df.columns else 0
            
            st.markdown(f"- **Out-of-Range Patient Ages (<0 or >120)**: `{invalid_age_n}` records")
            st.markdown(f"- **Inconsistent Hospital Dates (Discharge < Admission)**: `{inconsistent_date_n}` records")

            if health_quality["health_score"] >= 80:
                st.success("High Clinical Data Health: Records comply with hospital auditing standards.")
            elif health_quality["health_score"] >= 50:
                st.warning("Moderate Clinical Data Health: Cleaning recommended before clinical reporting.")
            else:
                st.error("Poor Clinical Data Health: Critical data inconsistencies detected. Cleaning required.")

        st.markdown("##### Detailed Clinical Data Quality Findings")
        if not health_quality["issues_table"].empty:
            st.dataframe(health_quality["issues_table"], use_container_width=True)
        else:
            st.success("No critical clinical anomalies detected in the dataset.")

    # TAB G2: CLINICAL RECOMMENDATIONS
    with tab_recs:
        health_recs = generate_healthcare_recommendations(health_df, health_mapping, health_summary)
        render_recommendation_cards(health_recs, "Healthcare Clinical Operations")

    # TAB H: RAW DATASET PREVIEW
    with tab_prev:
        st.markdown("#### Patient Records Raw Preview")
        st.dataframe(df.head(15), use_container_width=True)

        with st.expander("View Healthcare Column Types & Health Summary"):
            col_info_h = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Unique Values": df.nunique().values,
                "Missing Values": df.isnull().sum().values,
                "Missing Rate (%)": (df.isnull().mean() * 100).round(2).values
            })
            st.dataframe(col_info_h, use_container_width=True)

    st.divider()






# ==========================================
# 11. EDUCATION ANALYTICS DASHBOARD
# ==========================================

elif not is_hr_detected and active_workspace == "Education" and (selected_page in ["Academic Analytics", "Overview"] or "Analytics" in selected_page):

    st.subheader("Academic & Institutional Performance Dashboard")
    st.caption(f"Currently analyzing: {dataset_name} ({len(df):,} student records)")

    # COLUMN MAPPING & READINESS STATUS
    missing_crit_edu = edu_mapping_info["missing_critical"]
    if missing_crit_edu:
        st.warning(
            f"⚠️ **Notice: Missing Critical Academic Column(s)**: {', '.join(missing_crit_edu)}. "
            "Some student metrics may be incomplete. Check column names or view Data Diagnosis."
        )

    with st.expander("🔍 View Detected Academic Columns & Data Readiness"):
        col_map_cols = st.columns(4)
        with col_map_cols[0]:
            st.markdown("**Student Identifiers**")
            st.markdown(f"- Student ID: `{edu_mapping.get('student_id', 'Not detected (Row-based)')}`")
            st.markdown(f"- Student Name: `{edu_mapping.get('student_name', 'Not detected (ID used)')}`")
            st.markdown(f"- Date / Term: `{edu_mapping.get('date', 'Not detected')}`")
        with col_map_cols[1]:
            st.markdown("**Academic Performance**")
            st.markdown(f"- Marks / Score: `{edu_mapping.get('marks', 'Not detected')}`")
            st.markdown(f"- Max Marks: `{edu_mapping.get('max_marks', 'Defaulted (100)')}`")
            st.markdown(f"- Pass Mark: `{edu_mapping.get('pass_mark', 'Standard (40%)')}`")
        with col_map_cols[2]:
            st.markdown("**Class & Curriculum**")
            st.markdown(f"- Subject: `{edu_mapping.get('subject', 'Not detected (General)')}`")
            st.markdown(f"- Class / Grade: `{edu_mapping.get('class', 'Not detected (General)')}`")
            st.markdown(f"- Department: `{edu_mapping.get('department', 'Not detected')}`")
        with col_map_cols[3]:
            st.markdown("**Faculty & Attendance**")
            st.markdown(f"- Attendance: `{edu_mapping.get('attendance', 'Not detected')}`")
            st.markdown(f"- Teacher / Instructor: `{edu_mapping.get('teacher', 'Not detected')}`")

    # A. EDUCATION OVERVIEW KPI CARDS
    st.markdown("### Executive Academic Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Enrolled Students</div>
                <div class="retail-kpi-value">{edu_summary['total_students']:,}</div>
                <div class="retail-kpi-sub">Total Records: {edu_summary['total_records']:,}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        teacher_label = f"{edu_summary['total_teachers']:,}" if edu_summary['total_teachers'] > 0 else "N/A"
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Faculty Members</div>
                <div class="retail-kpi-value">{teacher_label}</div>
                <div class="retail-kpi-sub">Departments: {edu_summary['total_departments']}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Active Classes / Cohorts</div>
                <div class="retail-kpi-value">{edu_summary['total_classes']}</div>
                <div class="retail-kpi-sub">Subjects Taught: {edu_summary['total_subjects']}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        avg_m_label = f"{edu_summary['avg_marks']}%" if edu_summary['avg_marks'] is not None else "N/A"
        pass_p_label = f"Pass Rate: {edu_summary['pass_pct']}%" if edu_summary['pass_pct'] is not None else "Pass Rate: N/A"
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Average Marks</div>
                <div class="retail-kpi-value">{avg_m_label}</div>
                <div class="retail-kpi-sub">{pass_p_label}</div>
            </div>
        """, unsafe_allow_html=True)

    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    with kpi5:
        att_label = f"{edu_summary['avg_attendance']}%" if edu_summary['avg_attendance'] is not None else "N/A"
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Average Attendance</div>
                <div class="retail-kpi-value">{att_label}</div>
                <div class="retail-kpi-sub">Benchmarked at 75% min</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi6:
        pass_status = f"{edu_summary['pass_pct']}%" if edu_summary['pass_pct'] is not None else "N/A"
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Pass Percentage</div>
                <div class="retail-kpi-value">{pass_status}</div>
                <div class="retail-kpi-sub">Standard passing cutoff</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi7:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Curriculum Breadth</div>
                <div class="retail-kpi-value">{edu_summary['total_subjects']} Subjects</div>
                <div class="retail-kpi-sub">Across {edu_summary['total_departments']} Academic Streams</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi8:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Academic Data Health</div>
                <div class="retail-kpi-value">{edu_quality['health_score']}/100</div>
                <div class="retail-kpi-sub">{edu_quality['issue_count']} Audit Findings</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TABS FOR EDUCATION SECTIONS
    tab_perf, tab_att, tab_cls, tab_trend, tab_qual, tab_recs, tab_prev = st.tabs([
        "🎯 Student Performance",
        "📅 Attendance Analytics",
        "🏫 Class & Department",
        "📈 Education Trends",
        "🛡️ Education Data Quality",
        "💡 Academic Recommendations",
        "📋 Raw Dataset Preview"
    ])

    perf_data = get_performance_analytics(edu_df)
    att_data = get_attendance_analytics(edu_df)
    cls_data = get_class_department_analytics(edu_df)
    trend_data = get_education_trends(edu_df, freq="monthly")

    # TAB 1: STUDENT PERFORMANCE
    with tab_perf:
        st.markdown("#### Student Performance & Grading Analytics")
        col_pf1, col_pf2 = st.columns(2)

        with col_pf1:
            if not perf_data["grade_distribution"].empty:
                fig_grade = px.pie(
                    perf_data["grade_distribution"],
                    names="Grade",
                    values="Count",
                    title="Student Grade Distribution",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                fig_grade.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")
                st.plotly_chart(fig_grade, use_container_width=True)
            else:
                st.info("Marks or grading data not available.")

        with col_pf2:
            if not perf_data["pass_fail_summary"].empty:
                fig_pf = px.pie(
                    perf_data["pass_fail_summary"],
                    names="Status",
                    values="Count",
                    title="Pass vs Fail Outcomes",
                    hole=0.45,
                    color="Status",
                    color_discrete_map={"Passed": "#34D399", "Failed": "#F87171"}
                )
                fig_pf.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")
                st.plotly_chart(fig_pf, use_container_width=True)
            else:
                st.info("Pass/fail metrics not available.")

        # Marks Distribution Histogram
        valid_scores = edu_df[edu_df["_clean_pct"].between(0, 100)]
        if not valid_scores.empty:
            fig_hist = px.histogram(
                valid_scores,
                x="_clean_pct",
                nbins=20,
                title="Score Distribution Histogram (%)",
                color_discrete_sequence=["#38BDF8"]
            )
            fig_hist.update_layout(
                paper_bgcolor="#0B1120",
                plot_bgcolor="#16243A",
                font_color="#E2E8F0",
                xaxis_title="Score Percentage (%)",
                yaxis_title="Student Count"
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        col_tp1, col_tp2 = st.columns(2)
        with col_tp1:
            st.markdown("##### 🏆 Top-Performing Students")
            if not perf_data["top_students"].empty:
                st.dataframe(perf_data["top_students"], use_container_width=True)
            else:
                st.info("Top performing students could not be ranked.")

        with col_tp2:
            st.markdown("##### 🤝 Students Recommended for Academic Tutoring & Support")
            st.caption("Identified with care to foster constructive academic interventions.")
            if not perf_data["support_needed_students"].empty:
                st.dataframe(perf_data["support_needed_students"], use_container_width=True)
            else:
                st.success("All students currently meet or exceed minimum passing benchmarks (>=50%).")

        st.markdown("##### Subject-Wise Performance Breakdown")
        if not perf_data["subject_performance"].empty:
            col_sb1, col_sb2 = st.columns([1, 1])
            with col_sb1:
                fig_subj = px.bar(
                    perf_data["subject_performance"],
                    x="Subject",
                    y="Avg_Marks",
                    title="Average Marks by Subject",
                    color="Avg_Marks",
                    color_continuous_scale="Teal"
                )
                fig_subj.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                st.plotly_chart(fig_subj, use_container_width=True)
            with col_sb2:
                st.dataframe(perf_data["subject_performance"], use_container_width=True)

        # Student Performance Comparison Tool
        if "_clean_sname" in edu_df.columns:
            st.markdown("##### Individual Student Performance Lookup")
            student_list = sorted(edu_df["_clean_sname"].dropna().unique().tolist())
            if student_list:
                selected_student = st.selectbox("Select a Student to Review Transcript:", student_list)
                student_df = edu_df[edu_df["_clean_sname"] == selected_student]
                cols_to_show = [c for c in [edu_mapping.get("student_id"), edu_mapping.get("student_name"), edu_mapping.get("class"), edu_mapping.get("subject"), edu_mapping.get("marks"), "_clean_pct", "_grade", edu_mapping.get("attendance")] if c in student_df.columns or c in df.columns]
                st.dataframe(student_df[cols_to_show] if cols_to_show else student_df, use_container_width=True)

    # TAB 2: ATTENDANCE ANALYTICS
    with tab_att:
        st.markdown("#### Student Attendance & Engagement Analytics")
        if not att_data["valid_attendance_df"].empty:
            att_kpi1, att_kpi2, att_kpi3 = st.columns(3)
            with att_kpi1:
                st.metric("Overall Attendance Average", f"{att_data['avg_attendance']}%")
            with att_kpi2:
                st.metric("Students with Low Attendance (<75%)", f"{att_data['low_attendance_count']} students")
            with att_kpi3:
                st.metric("Low Attendance Rate", f"{att_data['low_attendance_pct']}% of cohort")

            col_at1, col_at2 = st.columns(2)
            with col_at1:
                fig_att_hist = px.histogram(
                    att_data["valid_attendance_df"],
                    x="_clean_attendance",
                    nbins=20,
                    title="Attendance Percentage Distribution",
                    color_discrete_sequence=["#34D399"]
                )
                fig_att_hist.update_layout(
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0",
                    xaxis_title="Attendance (%)",
                    yaxis_title="Count"
                )
                st.plotly_chart(fig_att_hist, use_container_width=True)

            with col_at2:
                if not att_data["class_attendance"].empty:
                    fig_cls_att = px.bar(
                        att_data["class_attendance"],
                        x="Class",
                        y="Average Attendance (%)",
                        title="Average Attendance by Class",
                        color="Average Attendance (%)",
                        color_continuous_scale="Greens"
                    )
                    fig_cls_att.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                    st.plotly_chart(fig_cls_att, use_container_width=True)

            st.markdown("##### ⚠️ Low Attendance Roster (< 75% Attendance Alert)")
            if not att_data["low_attendance_table"].empty:
                st.dataframe(att_data["low_attendance_table"], use_container_width=True)
            else:
                st.success("All students currently maintain regular attendance (>=75%).")
        else:
            st.info("Attendance data was not mapped or detected in the current dataset.")

    # TAB 3: CLASS & DEPARTMENT ANALYTICS
    with tab_cls:
        st.markdown("#### Class Headcounts & Departmental Distribution")
        col_cd1, col_cd2 = st.columns(2)

        with col_cd1:
            if not cls_data["class_summary"].empty:
                fig_cls_cnt = px.bar(
                    cls_data["class_summary"],
                    x="Class",
                    y="Students",
                    title="Student Headcount by Class / Grade",
                    color="Students",
                    color_continuous_scale="Blues"
                )
                fig_cls_cnt.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                st.plotly_chart(fig_cls_cnt, use_container_width=True)

        with col_cd2:
            if not cls_data["department_summary"].empty:
                fig_dept_cnt = px.pie(
                    cls_data["department_summary"],
                    names="Department",
                    values="Students",
                    title="Enrollment by Academic Department",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Cyan
                )
                fig_dept_cnt.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")
                st.plotly_chart(fig_dept_cnt, use_container_width=True)

        st.markdown("##### Class-wise Academic & Attendance Matrix")
        if not cls_data["class_summary"].empty:
            st.dataframe(cls_data["class_summary"], use_container_width=True)

        st.markdown("##### Faculty Teaching Workloads & Course Performance")
        if not cls_data["teacher_summary"].empty:
            st.dataframe(cls_data["teacher_summary"], use_container_width=True)

    # TAB 4: EDUCATION TRENDS
    with tab_trend:
        st.markdown("#### Academic Enrollment & Performance Timelines")
        if not trend_data["monthly_performance"].empty or not trend_data["monthly_attendance"].empty:
            col_tr1, col_tr2 = st.columns(2)
            with col_tr1:
                if not trend_data["monthly_performance"].empty:
                    fig_perf_tr = px.line(
                        trend_data["monthly_performance"],
                        x="Period",
                        y="Average Score (%)",
                        title="Average Marks Trend Over Time",
                        markers=True,
                        line_shape="spline"
                    )
                    fig_perf_tr.update_traces(line_color="#38BDF8")
                    fig_perf_tr.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                    st.plotly_chart(fig_perf_tr, use_container_width=True)
            with col_tr2:
                if not trend_data["monthly_attendance"].empty:
                    fig_att_tr = px.line(
                        trend_data["monthly_attendance"],
                        x="Period",
                        y="Average Attendance (%)",
                        title="Monthly Attendance Trend",
                        markers=True,
                        line_shape="spline"
                    )
                    fig_att_tr.update_traces(line_color="#34D399")
                    fig_att_tr.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                    st.plotly_chart(fig_att_tr, use_container_width=True)

            if not trend_data["enrollment_trend"].empty:
                fig_enr_tr = px.bar(
                    trend_data["enrollment_trend"],
                    x="Period",
                    y="Evaluations / Enrollments",
                    title="Evaluation / Assessment Volume by Date",
                    color_discrete_sequence=["#6366F1"]
                )
                fig_enr_tr.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                st.plotly_chart(fig_enr_tr, use_container_width=True)
        else:
            st.info("Date information is not present to generate academic timelines.")

    # TAB 5: EDUCATION DATA QUALITY
    with tab_qual:
        st.markdown("#### Educational Data Quality & Integrity Audit")
        eq1, eq2 = st.columns([1, 1])

        with eq1:
            gauge_eq = create_health_gauge(edu_quality["health_score"])
            st.plotly_chart(gauge_eq, use_container_width=True)

        with eq2:
            st.markdown("##### Academic Quality Assessment Summary")
            st.markdown(f"- **Identified Quality Anomalies**: `{edu_quality['issue_count']}` distinct issue types")
            st.markdown(f"- **Duplicate Student Records**: `{edu_quality['dup_count']}` records")
            
            invalid_marks_n = int(edu_df["_invalid_marks"].sum()) if "_invalid_marks" in edu_df.columns else 0
            invalid_att_n = int(edu_df["_invalid_attendance"].sum()) if "_invalid_attendance" in edu_df.columns else 0
            st.markdown(f"- **Out-of-Range Marks (<0 or >Max)**: `{invalid_marks_n}` records")
            st.markdown(f"- **Out-of-Range Attendance (<0% or >100%)**: `{invalid_att_n}` records")

            if edu_quality["health_score"] >= 80:
                st.success("High Academic Data Health: Records comply with institutional auditing standards.")
            elif edu_quality["health_score"] >= 50:
                st.warning("Moderate Academic Data Health: Cleaning recommended before grade reporting.")
            else:
                st.error("Poor Academic Data Health: Data anomalies detected. Cleaning required.")

        st.markdown("##### Detailed Academic Data Quality Findings")
        if not edu_quality["issues_table"].empty:
            st.dataframe(edu_quality["issues_table"], use_container_width=True)
        else:
            st.success("No critical academic anomalies detected in the dataset.")

    # TAB 5B: ACADEMIC RECOMMENDATIONS
    with tab_recs:
        edu_recs = generate_education_recommendations(edu_df, edu_mapping, edu_summary)
        render_recommendation_cards(edu_recs, "Education Academic Performance")

    # TAB 6: RAW DATASET PREVIEW
    with tab_prev:
        st.markdown("#### Student Records Raw Preview")
        st.dataframe(df.head(15), use_container_width=True)

        with st.expander("View Education Column Types & Health Summary"):
            col_info_e = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Unique Values": df.nunique().values,
                "Missing Values": df.isnull().sum().values,
                "Missing Rate (%)": (df.isnull().mean() * 100).round(2).values
            })
            st.dataframe(col_info_e, use_container_width=True)

    st.divider()


# ==========================================
# 12. BUSINESS & FINANCE DASHBOARD
# ==========================================

elif not is_hr_detected and active_workspace == "Business & Finance" and (selected_page in ["Financial Analytics", "Overview"] or "Analytics" in selected_page):

    st.subheader("Corporate Finance & Business Performance Dashboard")
    st.caption(f"Currently analyzing: {dataset_name} ({len(df):,} transactions)")

    # COLUMN MAPPING & READINESS STATUS
    missing_crit_bf = bf_mapping_info["missing_critical"]
    if missing_crit_bf:
        st.warning(
            f"⚠️ **Notice: Missing Critical Financial Column(s)**: {', '.join(missing_crit_bf)}. "
            "Some financial metrics may be incomplete. Check column names or view Data Diagnosis."
        )

    with st.expander("🔍 View Detected Financial Columns & Data Readiness"):
        col_map_cols = st.columns(4)
        with col_map_cols[0]:
            st.markdown("**Financial Figures**")
            st.markdown(f"- Revenue: `{bf_mapping.get('revenue', 'Not detected (0.0)')}`")
            st.markdown(f"- Expense: `{bf_mapping.get('expense', 'Not detected (0.0)')}`")
            st.markdown(f"- Profit Mode: `Revenue - Expenses`")
        with col_map_cols[1]:
            st.markdown("**Cost Centers & Geography**")
            st.markdown(f"- Category: `{bf_mapping.get('category', 'Not detected (General)')}`")
            st.markdown(f"- Department: `{bf_mapping.get('department', 'Not detected')}`")
            st.markdown(f"- Region: `{bf_mapping.get('region', 'Not detected')}`")
        with col_map_cols[2]:
            st.markdown("**Commercial Details**")
            st.markdown(f"- Customer: `{bf_mapping.get('customer', 'Not detected')}`")
            st.markdown(f"- Product / Service: `{bf_mapping.get('product', 'Not detected')}`")
            st.markdown(f"- Budget vs Actual: `{'Active' if bf_meta['has_budget'] else 'Not detected'}`")
        with col_map_cols[3]:
            st.markdown("**Transactions & Operations**")
            st.markdown(f"- Transaction ID: `{bf_mapping.get('transaction_id', 'Auto-generated')}`")
            st.markdown(f"- Date: `{bf_mapping.get('date', 'Not detected')}`")
            st.markdown(f"- Status: `{bf_mapping.get('status', 'Completed')}`")
            st.markdown(f"- Payment Method: `{bf_mapping.get('payment_method', 'Direct')}`")

    # A. BUSINESS OVERVIEW KPI CARDS
    st.markdown("### Executive Business & Financial Overview")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    with kpi1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Revenue</div>
                <div class="retail-kpi-value">${bf_summary['total_revenue']:,.2f}</div>
                <div class="retail-kpi-sub">Total Inflow Generated</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Expenses</div>
                <div class="retail-kpi-value">${bf_summary['total_expenses']:,.2f}</div>
                <div class="retail-kpi-sub">Operational & Capital Outflows</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        profit_color = "#34D399" if bf_summary['net_profit'] >= 0 else "#F87171"
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Net Profit</div>
                <div class="retail-kpi-value" style="color:{profit_color};">${bf_summary['net_profit']:,.2f}</div>
                <div class="retail-kpi-sub">Margin: {bf_summary['profit_margin']}%</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Profit Margin</div>
                <div class="retail-kpi-value">{bf_summary['profit_margin']}%</div>
                <div class="retail-kpi-sub">Net Profit / Revenue</div>
            </div>
        """, unsafe_allow_html=True)

    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    with kpi5:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Transactions</div>
                <div class="retail-kpi-value">{bf_summary['total_transactions']:,}</div>
                <div class="retail-kpi-sub">Ledger Entries Recorded</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi6:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Avg Transaction Value</div>
                <div class="retail-kpi-value">${bf_summary['atv']:,.2f}</div>
                <div class="retail-kpi-sub">Per Revenue Transaction</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi7:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Active Customers / Accounts</div>
                <div class="retail-kpi-value">{bf_summary['total_customers']:,}</div>
                <div class="retail-kpi-sub">Unique Client Base</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi8:
        mom_label = f"{bf_summary['mom_growth']:+.1f}%" if bf_summary['mom_growth'] is not None else "N/A"
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">MoM Revenue Growth</div>
                <div class="retail-kpi-value">{mom_label}</div>
                <div class="retail-kpi-sub">Financial Health: {bf_quality['health_score']}/100</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # TABS FOR BUSINESS & FINANCE SECTIONS
    tab_rev, tab_exp, tab_pl, tab_txn, tab_budg, tab_cf, tab_fqual, tab_frecs, tab_fprev = st.tabs([
        "💰 Revenue Analytics",
        "📉 Expense Analytics",
        "⚖️ Profit & Loss",
        "💳 Transactions & Sales",
        "🎯 Budget vs Actual",
        "🌊 Cash Flow & Liquidity",
        "🛡️ Financial Data Quality",
        "💡 Financial Recommendations",
        "📋 Raw Dataset Preview"
    ])

    rev_data = get_revenue_analytics(bf_df)
    exp_data = get_expense_analytics(bf_df)
    pl_data = get_profit_loss_analytics(bf_df)
    txn_data = get_transaction_analytics(bf_df)
    budg_data = get_budget_analytics(bf_df)
    cf_data = get_cash_flow_analytics(bf_df)

    # TAB 1: REVENUE ANALYTICS
    with tab_rev:
        st.markdown("#### Revenue Streams & Growth Analytics")
        if not rev_data["revenue_trend"].empty:
            fig_rev_tr = px.line(
                rev_data["revenue_trend"],
                x="Period",
                y="Revenue",
                title="Revenue Performance Trend ($)",
                markers=True,
                line_shape="spline"
            )
            fig_rev_tr.update_traces(line_color="#38BDF8")
            fig_rev_tr.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
            st.plotly_chart(fig_rev_tr, use_container_width=True)

        col_rv1, col_rv2 = st.columns(2)
        with col_rv1:
            if not rev_data["product_revenue"].empty:
                fig_prod_rev = px.bar(
                    rev_data["product_revenue"].head(10),
                    x="Revenue",
                    y="Product / Service",
                    orientation="h",
                    title="Revenue by Product / Service",
                    color="Revenue",
                    color_continuous_scale="Teal"
                )
                fig_prod_rev.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0", yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig_prod_rev, use_container_width=True)
            else:
                st.info("Product/service level revenue attribution not available.")

        with col_rv2:
            if not rev_data["region_revenue"].empty:
                fig_reg_rev = px.pie(
                    rev_data["region_revenue"],
                    names="Region",
                    values="Revenue",
                    title="Geographic Revenue Share",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Teal
                )
                fig_reg_rev.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")
                st.plotly_chart(fig_reg_rev, use_container_width=True)
            else:
                st.info("Regional revenue attribution not available.")

        if not rev_data["dept_revenue"].empty:
            st.markdown("##### Departmental Revenue Contribution")
            st.dataframe(rev_data["dept_revenue"], use_container_width=True)

    # TAB 2: EXPENSE ANALYTICS
    with tab_exp:
        st.markdown("#### Operational & Capital Expense Breakdown")
        col_ex1, col_ex2 = st.columns(2)

        with col_ex1:
            if not exp_data["category_expense"].empty:
                fig_cat_exp = px.pie(
                    exp_data["category_expense"],
                    names="Expense Category",
                    values="Expense",
                    title="Expense by Category / Cost Center",
                    hole=0.45,
                    color_discrete_sequence=px.colors.sequential.Blues
                )
                fig_cat_exp.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")
                st.plotly_chart(fig_cat_exp, use_container_width=True)
            else:
                st.info("Expense category breakdown not available.")

        with col_ex2:
            if not exp_data["dept_expense"].empty:
                fig_dept_exp = px.bar(
                    exp_data["dept_expense"],
                    x="Department",
                    y="Expense",
                    title="Departmental Expenditure ($)",
                    color="Expense",
                    color_continuous_scale="Reds"
                )
                fig_dept_exp.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                st.plotly_chart(fig_dept_exp, use_container_width=True)
            else:
                st.info("Departmental expense breakdown not available.")

        if not exp_data["expense_trend"].empty:
            fig_exp_tr = px.bar(
                exp_data["expense_trend"],
                x="Period",
                y="Expense",
                title="Monthly Expense Trend ($)",
                color_discrete_sequence=["#F87171"]
            )
            fig_exp_tr.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
            st.plotly_chart(fig_exp_tr, use_container_width=True)

        if not exp_data["category_expense"].empty:
            st.markdown("##### Expense Category Table")
            st.dataframe(exp_data["category_expense"], use_container_width=True)

    # TAB 3: PROFIT & LOSS
    with tab_pl:
        st.markdown("#### Profit & Loss (P&L) Statement Analysis")
        if not pl_data["monthly_pl"].empty:
            fig_pl = go.Figure()
            fig_pl.add_trace(go.Bar(
                x=pl_data["monthly_pl"]["Period"],
                y=pl_data["monthly_pl"]["Revenue"],
                name="Revenue",
                marker_color="#38BDF8"
            ))
            fig_pl.add_trace(go.Bar(
                x=pl_data["monthly_pl"]["Period"],
                y=pl_data["monthly_pl"]["Expense"],
                name="Expense",
                marker_color="#F87171"
            ))
            fig_pl.add_trace(go.Scatter(
                x=pl_data["monthly_pl"]["Period"],
                y=pl_data["monthly_pl"]["Net_Profit"],
                name="Net Profit",
                mode="lines+markers",
                line=dict(color="#34D399", width=3)
            ))
            fig_pl.update_layout(
                barmode="group",
                title="Monthly P&L Comparison (Revenue vs Expense vs Net Profit)",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#16243A",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig_pl, use_container_width=True)

            st.markdown("##### Monthly P&L Ledger Breakdown")
            st.dataframe(pl_data["monthly_pl"], use_container_width=True)

        col_pl1, col_pl2 = st.columns(2)
        with col_pl1:
            st.markdown("##### Department Profitability Matrix")
            if not pl_data["dept_profitability"].empty:
                st.dataframe(pl_data["dept_profitability"], use_container_width=True)

        with col_pl2:
            st.markdown("##### Product / Service Profitability Matrix")
            if not pl_data["product_profitability"].empty:
                st.dataframe(pl_data["product_profitability"].head(10), use_container_width=True)

    # TAB 4: TRANSACTIONS & SALES
    with tab_txn:
        st.markdown("#### Transaction Integrity & Customer Spending")
        col_tx1, col_tx2 = st.columns(2)

        with col_tx1:
            if not txn_data["status_summary"].empty:
                fig_st = px.pie(
                    txn_data["status_summary"],
                    names="Status",
                    values="Count",
                    title="Transaction Settlement Status",
                    hole=0.45,
                    color="Status",
                    color_discrete_map={"Completed": "#34D399", "Pending": "#FBBF24", "Failed": "#F87171"}
                )
                fig_st.update_layout(paper_bgcolor="#0B1120", font_color="#E2E8F0")
                st.plotly_chart(fig_st, use_container_width=True)

        with col_tx2:
            if not txn_data["payment_methods"].empty:
                fig_pm = px.bar(
                    txn_data["payment_methods"],
                    x="Payment Method",
                    y="Transactions",
                    title="Payment Channel Distribution",
                    color="Transactions",
                    color_continuous_scale="Purples"
                )
                fig_pm.update_layout(paper_bgcolor="#0B1120", plot_bgcolor="#16243A", font_color="#E2E8F0")
                st.plotly_chart(fig_pm, use_container_width=True)

        st.markdown("##### Top 10 High-Value Client Accounts")
        if not txn_data["top_customers"].empty:
            st.dataframe(txn_data["top_customers"], use_container_width=True)

    # TAB 5: BUDGET VS ACTUAL
    with tab_budg:
        st.markdown("#### Budget Variance & Cost Center Accountability")
        if budg_data["has_budget_data"]:
            b_kpi1, b_kpi2, b_kpi3, b_kpi4 = st.columns(4)
            with b_kpi1:
                st.metric("Total Budget Allocated", f"${budg_data['total_budget']:,.2f}")
            with b_kpi2:
                st.metric("Actual Expenditure", f"${budg_data['total_actual']:,.2f}")
            with b_kpi3:
                var_val = budg_data['variance']
                st.metric("Budget Variance", f"${var_val:,.2f}", delta=f"{'Under Budget' if var_val >= 0 else 'Over Budget'}")
            with b_kpi4:
                st.metric("Budget Utilization", f"{budg_data['utilization_pct']}%")

            if not budg_data["dept_budget_table"].empty:
                fig_budg = go.Figure()
                fig_budg.add_trace(go.Bar(
                    x=budg_data["dept_budget_table"]["Department"],
                    y=budg_data["dept_budget_table"]["Budget"],
                    name="Allocated Budget",
                    marker_color="#38BDF8"
                ))
                fig_budg.add_trace(go.Bar(
                    x=budg_data["dept_budget_table"]["Department"],
                    y=budg_data["dept_budget_table"]["Actual"],
                    name="Actual Spend",
                    marker_color="#F87171"
                ))
                fig_budg.update_layout(
                    barmode="group",
                    title="Departmental Budget vs Actual Spending ($)",
                    paper_bgcolor="#0B1120",
                    plot_bgcolor="#16243A",
                    font_color="#E2E8F0"
                )
                st.plotly_chart(fig_budg, use_container_width=True)

                st.markdown("##### Department Budget Variance Analysis")
                st.dataframe(budg_data["dept_budget_table"], use_container_width=True)
        else:
            st.info("Budget information (Budget & Actual columns) was not detected in this dataset.")

    # TAB 6: CASH FLOW & LIQUIDITY
    with tab_cf:
        st.markdown("#### Cash Inflows, Outflows & Liquidity Velocity")
        cf_kpi1, cf_kpi2, cf_kpi3 = st.columns(3)
        with cf_kpi1:
            st.metric("Cash Inflow (Revenue)", f"${cf_data['cash_inflow']:,.2f}")
        with cf_kpi2:
            st.metric("Cash Outflow (Expenses)", f"${cf_data['cash_outflow']:,.2f}")
        with cf_kpi3:
            net_color = "normal" if cf_data['net_cash_flow'] >= 0 else "inverse"
            st.metric("Net Cash Flow", f"${cf_data['net_cash_flow']:,.2f}")

        if not cf_data["cash_flow_trend"].empty:
            fig_cf = go.Figure()
            fig_cf.add_trace(go.Bar(
                x=cf_data["cash_flow_trend"]["Period"],
                y=cf_data["cash_flow_trend"]["Inflow"],
                name="Cash Inflow",
                marker_color="#34D399"
            ))
            fig_cf.add_trace(go.Bar(
                x=cf_data["cash_flow_trend"]["Period"],
                y=cf_data["cash_flow_trend"]["Outflow"],
                name="Cash Outflow",
                marker_color="#F87171"
            ))
            fig_cf.add_trace(go.Scatter(
                x=cf_data["cash_flow_trend"]["Period"],
                y=cf_data["cash_flow_trend"]["Net_Cash_Flow"],
                name="Net Cash Flow",
                mode="lines+markers",
                line=dict(color="#38BDF8", width=3)
            ))
            fig_cf.update_layout(
                barmode="group",
                title="Monthly Cash Flow Dynamics (Inflow vs Outflow vs Net Flow)",
                paper_bgcolor="#0B1120",
                plot_bgcolor="#16243A",
                font_color="#E2E8F0"
            )
            st.plotly_chart(fig_cf, use_container_width=True)

    # TAB 7: FINANCIAL DATA QUALITY
    with tab_fqual:
        st.markdown("#### Financial Ledger Integrity & Quality Audit")
        fq1, fq2 = st.columns([1, 1])

        with fq1:
            gauge_fq = create_health_gauge(bf_quality["health_score"])
            st.plotly_chart(gauge_fq, use_container_width=True)

        with fq2:
            st.markdown("##### Financial Quality Assessment Summary")
            st.markdown(f"- **Identified Audit Issues**: `{len(bf_quality['issues_table'])}` distinct issue types")
            dup_txns = int(df.duplicated().sum())
            neg_rev_n = int(bf_df["_is_negative_rev"].sum()) if "_is_negative_rev" in bf_df.columns else 0
            neg_exp_n = int(bf_df["_is_negative_exp"].sum()) if "_is_negative_exp" in bf_df.columns else 0
            st.markdown(f"- **Duplicate Ledger Transactions**: `{dup_txns}` records")
            st.markdown(f"- **Negative Revenue/Expense Entries**: `{neg_rev_n + neg_exp_n}` records")

            if bf_quality["health_score"] >= 80:
                st.success("High Financial Data Health: Ledger entries meet reconciliation standards.")
            elif bf_quality["health_score"] >= 50:
                st.warning("Moderate Financial Data Health: Reconciliation recommended before filing.")
            else:
                st.error("Poor Financial Data Health: Critical ledger anomalies detected. Cleaning required.")

        st.markdown("##### Detailed Financial Audit Findings")
        if not bf_quality["issues_table"].empty:
            st.dataframe(bf_quality["issues_table"], use_container_width=True)
        else:
            st.success("No critical financial ledger anomalies detected in the dataset.")

    # TAB 7B: FINANCIAL RECOMMENDATIONS
    with tab_frecs:
        finance_recs = generate_finance_recommendations(bf_df, bf_mapping, bf_summary)
        render_recommendation_cards(finance_recs, "Business & Corporate Finance")

    # TAB 8: RAW DATASET PREVIEW
    with tab_fprev:
        st.markdown("#### Financial Records Raw Preview")
        st.dataframe(df.head(15), use_container_width=True)

        with st.expander("View Financial Column Types & Health Summary"):
            col_info_f = pd.DataFrame({
                "Column": df.columns,
                "Data Type": df.dtypes.astype(str).values,
                "Unique Values": df.nunique().values,
                "Missing Values": df.isnull().sum().values,
                "Missing Rate (%)": (df.isnull().mean() * 100).round(2).values
            })
            st.dataframe(col_info_f, use_container_width=True)

    st.divider()








# ==========================================
# 13. AI DATA DOCTOR PAGE (FEATURE 1)
# ==========================================

elif selected_page == "🩺 AI Data Doctor":
    st.subheader("🩺 AI Data Doctor & Intelligent Dataset Health")
    st.caption(f"Comprehensive diagnostic reasoning, root cause analysis & actionable remediation ({len(df):,} records)")

    effective_ws = "Employee & HR" if is_hr_detected else active_workspace
    diag_res = diagnose_dataset(df, effective_ws, current_mapping)
    issues = diag_res["issues"]

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Health Score</div>
                <div class="retail-kpi-value">{original_score}/100</div>
                <div class="retail-kpi-sub">Baseline Quality Index</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi2:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Identified Deficiencies</div>
                <div class="retail-kpi-value">{len(issues)} Issues</div>
                <div class="retail-kpi-sub">Quality Audit Findings</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi3:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Critical / High Severity</div>
                <div class="retail-kpi-value" style="color:#F87171;">{diag_res['high_severity_count']} High</div>
                <div class="retail-kpi-sub">Requires Immediate Action</div>
            </div>
        """, unsafe_allow_html=True)
    with kpi4:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Est. Health Deduction</div>
                <div class="retail-kpi-value" style="color:#FBBF24;">-{diag_res['health_impact']} pts</div>
                <div class="retail-kpi-sub">Score Degradation</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"**Diagnostic Overview**: {diag_res['summary']}")

    # Severity filter
    sev_filter = st.radio("Filter Issues by Severity:", ["All Severities", "High Severity Only", "Medium & Low"], horizontal=True)
    if sev_filter == "High Severity Only":
        filtered_issues = [i for i in issues if i["severity"] == "High"]
    elif sev_filter == "Medium & Low":
        filtered_issues = [i for i in issues if i["severity"] in ["Medium", "Low"]]
    else:
        filtered_issues = issues

    if not filtered_issues:
        st.success("No data quality issues match the selected criteria.")
    else:
        for idx, iss in enumerate(filtered_issues):
            sev_class = "diag-card-high" if iss["severity"] == "High" else ("diag-card-medium" if iss["severity"] == "Medium" else "diag-card-low")
            badge_class = "pill-danger" if iss["severity"] == "High" else ("pill-warning" if iss["severity"] == "Medium" else "pill-info")
            
            with st.expander(f"{iss['severity'].upper()}: {iss['problem_name']} ({iss['affected_records']:,} records affected)"):
                st.markdown(f"""
                    <div class="diag-card {sev_class}">
                        <span class="pill-badge {badge_class}" style="margin-bottom:8px;">{iss['severity']} Severity</span>
                        <h4 style="color:#F8FAFC; margin-top:6px; margin-bottom:8px;">{iss['problem_name']}</h4>
                        <p style="color:#CBD5E1; font-size:13px; margin-bottom:12px;"><b>Why This Matters:</b> {iss['why_it_matters']}</p>
                        <p style="color:#38BDF8; font-size:13px; margin-bottom:4px;"><b>Suggested Fix:</b> {iss['suggested_fix']}</p>
                        <p style="color:#94A3B8; font-size:12px; margin-bottom:0;"><b>Expected Outcome:</b> {iss['expected_impact']}</p>
                    </div>
                """, unsafe_allow_html=True)

                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    preview_key = f"preview_btn_{iss['id']}_{idx}"
                    show_prev = st.checkbox("👁️ Preview Proposed Fix (Before vs After Diff)", key=preview_key)
                    if show_prev:
                        prev = preview_fix(df, iss["fix_type"], iss["target_col"])
                        if prev:
                            st.info(f"**Transformation Logic**: {prev['explanation']}")
                            p_col1, p_col2 = st.columns(2)
                            with p_col1:
                                st.markdown("##### Before Transformation (Sample)")
                                st.dataframe(prev["before_sample"], use_container_width=True)
                            with p_col2:
                                st.markdown("##### After Proposed Fix (Sample)")
                                st.dataframe(prev["after_sample"], use_container_width=True)
                with col_btn2:
                    apply_key = f"apply_btn_{iss['id']}_{idx}"
                    if st.button(f"✅ Apply Fix: {iss['suggested_fix']}", key=apply_key, use_container_width=True):
                        new_df = apply_fix(df, iss["fix_type"], iss["target_col"])
                        st.session_state["cleaned_df"] = new_df
                        st.success(f"Fix '{iss['suggested_fix']}' applied successfully! Updated dataset stored in session.")
                        st.rerun()

    st.divider()
    with st.expander("📊 View Detailed Diagnostic Tables (Missing, Duplicates, Outliers)"):
        st.markdown("##### Missing Values Table")
        miss_df = pd.DataFrame({"Column": df.columns, "Missing Values": df.isnull().sum().values, "Percentage (%)": (df.isnull().mean().values * 100).round(2)})
        st.dataframe(miss_df, use_container_width=True)
        if original_duplicates > 0:
            st.markdown(f"##### Duplicate Rows ({original_duplicates:,} rows)")
            st.dataframe(df[df.duplicated(keep=False)].head(15), use_container_width=True)
        if not outlier_results.empty:
            st.markdown("##### Outlier Analysis Table")
            st.dataframe(outlier_results, use_container_width=True)


# ==========================================
# 14. ANOMALY DETECTIVE PAGE (FEATURE 2)
# ==========================================

elif selected_page == "🔍 Anomaly Detective":
    st.subheader("🔍 Smart Anomaly Detective")
    st.caption(f"Multivariate pattern recognition, statistical distribution outliers & operational guardrail auditing ({active_workspace})")

    anom_res = detect_smart_anomalies(df, active_workspace, current_mapping)
    anom_list = anom_res["anomalies"]
    total_anom = anom_res["total_anomalies"]
    high_anom = anom_res["high_severity_count"]

    akpi1, akpi2, akpi3 = st.columns(3)
    with akpi1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Total Detected Anomalies</div>
                <div class="retail-kpi-value">{total_anom} Events</div>
                <div class="retail-kpi-sub">Statistical & Domain Outliers</div>
            </div>
        """, unsafe_allow_html=True)
    with akpi2:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Critical / High Severity</div>
                <div class="retail-kpi-value" style="color:#F87171;">{high_anom} High</div>
                <div class="retail-kpi-sub">Severe Pattern Deviations</div>
            </div>
        """, unsafe_allow_html=True)
    with akpi3:
        st.markdown("""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Active Detection Stack</div>
                <div class="retail-kpi-value" style="font-size:20px; color:#34D399;">Multi-Layered</div>
                <div class="retail-kpi-sub">Multivariate + Z-Score + Rules</div>
            </div>
        """, unsafe_allow_html=True)

    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        st.markdown("### Interactive Anomaly Distribution Map")
        default_idx = 0
        if anom_res["featured_col"] and anom_res["featured_col"] in numeric_columns:
            default_idx = numeric_columns.index(anom_res["featured_col"])
        
        selected_anom_col = st.selectbox("Select Numeric Feature to Map Anomalies Against:", numeric_columns, index=default_idx)
        anom_fig = create_anomaly_visual_chart(df, selected_anom_col, anom_list)
        if anom_fig:
            st.plotly_chart(anom_fig, use_container_width=True)

    st.markdown("### Detected Anomaly Register")
    if not anom_res["anomaly_table"].empty:
        st.dataframe(anom_res["anomaly_table"], use_container_width=True)

        st.markdown("#### Individual Anomaly Drill-Down")
        for idx, a in enumerate(anom_list[:10]):
            sev_color = "#F87171" if a["severity"] == "High" else "#FBBF24"
            st.markdown(f"""
                <div class="diag-card" style="border-left-color: {sev_color};">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="color:#F8FAFC; margin:0;">{a['anomaly_type']} &bull; {a['column_or_category']}</h4>
                        <span class="pill-badge pill-danger" style="color:{sev_color};">{a['severity']}</span>
                    </div>
                    <p style="color:#94A3B8; font-size:13px; margin-top:6px; margin-bottom:4px;">
                        <b>Actual Value:</b> <code>{a['actual_value']}</code> &nbsp;|&nbsp; 
                        <b>Reference / Expected:</b> <code>{a['expected_value']}</code> &nbsp;|&nbsp; 
                        <b>Method:</b> <i>{a['detection_method']}</i>
                    </p>
                    <p style="color:#CBD5E1; font-size:13px; margin:0;">{a['explanation']}</p>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.success("No critical distribution or operational anomalies detected in the current dataset.")


# ==========================================
# 15. WHAT-IF SIMULATOR PAGE (FEATURE 3)
# ==========================================

elif selected_page == "🎛️ What-If Simulator":
    st.subheader("🎛️ What-If Scenario Simulator")
    st.caption(f"Dynamic sensitivity modeling and stress-testing without modifying baseline records ({active_workspace})")

    st.markdown("""
        <div style="background-color:#16243A; border-left:4px solid #38BDF8; padding:12px; border-radius:6px; margin-bottom:20px;">
            <p style="color:#CBD5E1; font-size:13px; margin:0;">
                ℹ️ <b>Simulation Disclaimer:</b> What-If projections model sensitivity outcomes based on mathematical parameter adjustments. 
                They reflect hypothetical scenarios for planning purposes, not guaranteed commercial, clinical, or academic predictions.
            </p>
        </div>
    """, unsafe_allow_html=True)

    if active_workspace == "Retail Store":
        st.markdown("#### Retail Commercial Scenario Controls")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            price_adj = st.slider("Product Price Adjustment (%)", -30, 50, 5, step=1)
        with col_s2:
            cost_adj = st.slider("COGS / Supplier Cost Adjustment (%)", -30, 30, -5, step=1)
        with col_s3:
            vol_adj = st.slider("Sales Volume / Footfall Adjustment (%)", -50, 100, 0, step=5)

        sim_out = simulate_retail_scenario(df, store_mapping, price_adj, cost_adj, vol_adj)

        wkpi1, wkpi2, wkpi3 = st.columns(3)
        with wkpi1:
            st.metric("Simulated Revenue", f"${sim_out['sim_revenue']:,.2f}", delta=f"{sim_out['rev_delta_pct']:+.1f}% vs Baseline")
        with wkpi2:
            st.metric("Simulated Gross Profit", f"${sim_out['sim_profit']:,.2f}", delta=f"{sim_out['profit_delta_pct']:+.1f}% vs Baseline")
        with wkpi3:
            st.metric("Simulated Profit Margin", f"{sim_out['sim_margin']}%", delta=f"{sim_out['margin_delta']:+.1f}% pts")

        comp_fig = create_simulation_comparison_chart(
            ["Total Revenue ($)", "Gross Profit ($)"],
            [sim_out["orig_revenue"], sim_out["orig_profit"]],
            [sim_out["sim_revenue"], sim_out["sim_profit"]],
            "Baseline vs Simulated Retail Performance"
        )
        st.plotly_chart(comp_fig, use_container_width=True)

    elif active_workspace == "Healthcare":
        st.markdown("#### Healthcare Capacity & Fulfillment Controls")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            cancel_red = st.slider("Cancellation Reduction (%)", 0, 80, 25, step=5)
        with col_s2:
            cap_exp = st.slider("Clinical Capacity Expansion (%)", -20, 50, 10, step=5)

        sim_out = simulate_healthcare_scenario(df, health_mapping, cancel_red, cap_exp)

        wkpi1, wkpi2, wkpi3 = st.columns(3)
        with wkpi1:
            st.metric("Simulated Completed Visits", f"{sim_out['sim_completed']:,}", delta=f"+{sim_out['delta_patients']:,} Additional Encounters")
        with wkpi2:
            st.metric("Remaining Cancellations", f"{sim_out['sim_cancelled']:,}", delta=f"-{sim_out['recovered_visits']:,} Recovered")
        with wkpi3:
            st.metric("Completion Rate", f"{sim_out['sim_completion_rate']}%", delta=f"{round(sim_out['sim_completion_rate'] - sim_out['orig_completion_rate'], 1):+.1f}% pts")

        comp_fig = create_simulation_comparison_chart(
            ["Completed Appointments", "Remaining Cancellations"],
            [sim_out["orig_completed"], sim_out["orig_cancelled"]],
            [sim_out["sim_completed"], sim_out["sim_cancelled"]],
            "Baseline vs Simulated Appointment Throughput"
        )
        st.plotly_chart(comp_fig, use_container_width=True)

    elif active_workspace == "Education":
        st.markdown("#### Academic Tutoring & Attendance Intervention Controls")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            marks_boost = st.slider("Academic Score Improvement (%)", 0, 25, 5, step=1)
        with col_s2:
            att_boost = st.slider("Attendance Rate Boost (%)", 0, 30, 5, step=1)
        with col_s3:
            pass_cut = st.slider("Passing Score Threshold (%)", 30, 60, 40, step=5)

        sim_out = simulate_education_scenario(df, edu_mapping, marks_boost, att_boost, pass_cut)

        wkpi1, wkpi2, wkpi3 = st.columns(3)
        with wkpi1:
            st.metric("Simulated Average Marks", f"{sim_out['sim_avg_marks']}%", delta=f"{round(sim_out['sim_avg_marks'] - sim_out['orig_avg_marks'], 1):+.1f}% pts")
        with wkpi2:
            st.metric("Simulated Pass Rate", f"{sim_out['sim_pass_pct']}%", delta=f"{round(sim_out['sim_pass_pct'] - sim_out['orig_pass_pct'], 1):+.1f}% pts")
        with wkpi3:
            st.metric("Students Needing Support (<50%)", f"{sim_out['sim_support_needed']:,}", delta=f"-{sim_out['students_lifted']:,} Lifted Above Benchmark")

        comp_fig = create_simulation_comparison_chart(
            ["Average Marks (%)", "Pass Rate (%)", "Average Attendance (%)"],
            [sim_out["orig_avg_marks"], sim_out["orig_pass_pct"], sim_out["orig_avg_att"]],
            [sim_out["sim_avg_marks"], sim_out["sim_pass_pct"], sim_out["sim_avg_att"]],
            "Baseline vs Simulated Academic Cohort Performance"
        )
        st.plotly_chart(comp_fig, use_container_width=True)

    else: # Business & Finance
        st.markdown("#### Corporate Financial Strategy Controls")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            rev_growth = st.slider("Top-Line Revenue Growth (%)", -30, 50, 8, step=1)
        with col_s2:
            exp_cut = st.slider("Operational Expense Reduction (%)", 0, 40, 10, step=1)

        sim_out = simulate_finance_scenario(df, bf_mapping, rev_growth, exp_cut)

        wkpi1, wkpi2, wkpi3 = st.columns(3)
        with wkpi1:
            st.metric("Simulated Revenue", f"${sim_out['sim_revenue']:,.2f}", delta=f"{rev_growth:+.1f}% Growth")
        with wkpi2:
            st.metric("Simulated Net Profit", f"${sim_out['sim_profit']:,.2f}", delta=f"{sim_out['profit_delta_pct']:+.1f}% vs Baseline")
        with wkpi3:
            st.metric("Simulated Profit Margin", f"{sim_out['sim_margin']}%", delta=f"{sim_out['margin_delta']:+.1f}% pts")

        comp_fig = create_simulation_comparison_chart(
            ["Total Revenue ($)", "Total Expenses ($)", "Net Profit ($)"],
            [sim_out["orig_revenue"], sim_out["orig_expenses"], sim_out["orig_profit"]],
            [sim_out["sim_revenue"], sim_out["sim_expenses"], sim_out["sim_profit"]],
            "Baseline vs Simulated Financial P&L"
        )
        st.plotly_chart(comp_fig, use_container_width=True)


# ==========================================
# 16. CHAT WITH DATASET PAGE (FEATURE 4)
# ==========================================

elif selected_page == "💬 Chat With Dataset":
    st.subheader("💬 Chat With Your Dataset")
    st.caption(f"Conversational analytical assistant & deterministic query processor ({active_workspace})")

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    st.markdown("**Suggested Quick Prompts:**")
    quick_cols = st.columns(4)
    if active_workspace == "Retail Store":
        chips = ["What is the total revenue?", "Which product generated highest revenue?", "Show missing values", "What is the dataset health score?"]
    elif active_workspace == "Healthcare":
        chips = ["How many appointments were completed?", "Which department has most patients?", "Show missing values", "What is the dataset health score?"]
    elif active_workspace == "Education":
        chips = ["What is the overall pass percentage?", "Which subject has highest average marks?", "How many students have low attendance?", "What is the dataset health score?"]
    else:
        chips = ["What are our total expenses and profit?", "What is our net profit margin?", "Show missing values", "What is the dataset health score?"]

    user_query = None
    for idx, chip in enumerate(chips):
        with quick_cols[idx]:
            if st.button(chip, key=f"chip_{idx}", use_container_width=True):
                user_query = chip

    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("table") is not None:
                st.dataframe(msg["table"], use_container_width=True)
            if msg.get("chart") is not None:
                st.plotly_chart(msg["chart"], use_container_width=True)

    input_text = st.chat_input("Ask any question about your dataset (e.g. revenue, missing values, top products, outliers)...")
    if input_text:
        user_query = input_text

    if user_query:
        st.session_state["chat_history"].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        response = ask_dataset(user_query, df, active_workspace, current_mapping)
        
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
            if response["calculation"]:
                with st.expander("🔍 View Mathematical Methodology"):
                    st.caption(response["calculation"])
            if response["table"] is not None:
                st.dataframe(response["table"], use_container_width=True)
            if response["chart"] is not None:
                st.plotly_chart(response["chart"], use_container_width=True)

        st.session_state["chat_history"].append({
            "role": "assistant",
            "content": response["answer"],
            "table": response["table"],
            "chart": response["chart"]
        })

    if st.session_state["chat_history"]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ Clear Conversation History"):
            st.session_state["chat_history"] = []
            st.rerun()


# ==========================================
# 17. EXECUTIVE PDF REPORT PAGE (FEATURE 5)
# ==========================================

elif selected_page in ["📑 Executive PDF Report", "Reports"]:
    st.subheader("📑 One-Click Executive PDF Report Generator")
    st.caption("Generate publication-ready C-level PDF audit reports with verified dataset metrics")

    effective_ws = "Employee & HR" if is_hr_detected else active_workspace
    anom_res = detect_smart_anomalies(df, effective_ws, current_mapping)
    diag_res = diagnose_dataset(df, effective_ws, current_mapping)

    rcol1, rcol2 = st.columns([1, 1])
    with rcol1:
        st.markdown(f"""
            <div class="section-box">
                <h4 style="color:#38BDF8; margin-top:0;">Audit Snapshot Overview</h4>
                <p><b>Target Workspace:</b> {effective_ws}</p>
                <p><b>Dataset Document:</b> <code>{dataset_name}</code></p>
                <p><b>Total Observations:</b> {len(df):,} records | <b>Columns:</b> {len(df.columns)}</p>
                <p><b>Current Health Score:</b> <span class="pill-badge pill-success">{original_score}/100</span></p>
                <p><b>Detected Anomalies:</b> {anom_res['total_anomalies']} events</p>
                <p><b>Cleaning Status:</b> {'Cleaned Dataset Active' if cleaned_df is not None else 'Raw Baseline'}</p>
            </div>
        """, unsafe_allow_html=True)

    with rcol2:
        st.markdown("""
            <div class="section-box">
                <h4 style="color:#38BDF8; margin-top:0;">Executive PDF Inclusions</h4>
                <p style="color:#94A3B8; font-size:13px;">
                    &bull; Executive Scorecard (Health Score, Missing %, Duplicates, Outliers)<br>
                    &bull; Domain Specific Key Performance Indicators (KPIs)<br>
                    &bull; Smart Anomaly Audit Findings Register<br>
                    &bull; Before-and-After Cleaning Reconciliation Table<br>
                    &bull; Enterprise Strategic Governance Recommendations
                </p>
            </div>
        """, unsafe_allow_html=True)

    pdf_bytes = generate_pdf_report(
        df=df,
        workspace=effective_ws,
        dataset_name=dataset_name,
        summary_dict=domain_summary,
        health_score=original_score,
        anomalies_list=anom_res["anomalies"],
        issues_list=diag_res["issues"],
        cleaned_df=cleaned_df
    )

    pdf_filename = f"DataDoctor_Audit_{active_workspace.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.pdf"

    st.download_button(
        label="📥 Download Executive PDF Audit Report",
        data=pdf_bytes,
        file_name=pdf_filename,
        mime="application/pdf",
        use_container_width=True
    )

    # Legacy text report download
    st.divider()
    with st.expander("📄 Download Plain-Text Health Report (.txt)"):
        report_txt = f"""DATADOCTOR AI HEALTH REPORT ({active_workspace})
Dataset: {dataset_name}
Rows: {len(df)}, Cols: {len(df.columns)}
Health Score: {original_score}/100
Missing Values: {original_missing}
Duplicate Rows: {original_duplicates}
Outliers: {original_outliers}
Date Consistency Issues: {original_consistency}
"""
        st.download_button(
            label="Download Plain-Text Report (.txt)",
            data=report_txt,
            file_name=f"DataDoctor_Report_{active_workspace}.txt",
            mime="text/plain",
            use_container_width=True
        )


# ==========================================
# 18. DATASET HEALTH HISTORY PAGE (FEATURE 6)
# ==========================================

elif selected_page == "📜 Health History":
    st.subheader("📜 Dataset Health Audit History & Progress Tracking")
    st.caption("Persistent SQLite repository tracking health score trajectories with zero sensitive PII retention")

    hcol1, hcol2 = st.columns([2, 1])
    with hcol1:
        st.markdown(f"Save snapshot of current **{active_workspace}** audit ({original_score}/100) to persistent database.")
    with hcol2:
        if st.button("💾 Save Current Audit Snapshot", use_container_width=True):
            anom_cnt = detect_smart_anomalies(df, active_workspace, current_mapping)["total_anomalies"]
            save_health_session(
                workspace=active_workspace,
                dataset_name=dataset_name,
                row_count=len(df),
                col_count=len(df.columns),
                health_score=original_score,
                missing_count=original_missing,
                duplicate_count=original_duplicates,
                outlier_count=original_outliers,
                anomaly_count=anom_cnt,
                cleaning_applied="Cleaned" if cleaned_df is not None else "Initial Baseline"
            )
            st.success("Audit snapshot saved to persistent health database!")
            st.rerun()

    hist_df = get_health_history()

    if not hist_df.empty:
        trend_fig = create_history_trend_chart(hist_df)
        if trend_fig:
            st.plotly_chart(trend_fig, use_container_width=True)

        st.markdown("### Historical Audit Sessions Log")
        display_hist = hist_df.rename(columns={
            "id": "Session ID",
            "timestamp": "Timestamp",
            "workspace": "Workspace",
            "dataset_name": "Dataset",
            "row_count": "Rows",
            "health_score": "Health Score",
            "missing_count": "Missing",
            "duplicate_count": "Duplicates",
            "anomaly_count": "Anomalies",
            "cleaning_applied": "Cleaning State"
        })
        st.dataframe(display_hist, use_container_width=True)

        if st.button("🗑️ Clear Historical Audit Records"):
            clear_health_history()
            st.warning("Audit history cleared from database.")
            st.rerun()
    else:
        st.info("No audit sessions saved yet. Click 'Save Current Audit Snapshot' above to create your first baseline record.")


# ==========================================
# 19. ML READINESS ADVISOR PAGE (FEATURE 7)
# ==========================================

elif selected_page in ["🤖 ML Readiness Advisor", "ML Readiness"]:
    st.subheader("🤖 Machine Learning Readiness Advisor")
    st.caption("Automated feature diagnostics, candidate target identification, class imbalance, and model family recommendations")

    target_candidate = st.selectbox("Select Target Column to Predict (Optional):", [None] + df.columns.tolist())
    task_choice = st.radio("Specify Machine Learning Task Type:", ["Auto-Detect", "Classification", "Regression", "Clustering"], horizontal=True)

    ml_eval = evaluate_ml_readiness(df, selected_target=target_candidate, task_type=task_choice)

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    with mcol1:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">ML Readiness Score</div>
                <div class="retail-kpi-value">{ml_eval['readiness_score']}/100</div>
                <div class="retail-kpi-sub">Hygiene & Completeness</div>
            </div>
        """, unsafe_allow_html=True)
    with mcol2:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Inferred Task</div>
                <div class="retail-kpi-value" style="color:#34D399; font-size:22px;">{ml_eval['task_type']}</div>
                <div class="retail-kpi-sub">Problem Formulation</div>
            </div>
        """, unsafe_allow_html=True)
    with mcol3:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Target Feature</div>
                <div class="retail-kpi-value" style="font-size:20px;">{str(ml_eval['target_col'])[:14]}</div>
                <div class="retail-kpi-sub">Prediction Objective</div>
            </div>
        """, unsafe_allow_html=True)
    with mcol4:
        st.markdown(f"""
            <div class="retail-kpi-card">
                <div class="retail-kpi-label">Sample Volume</div>
                <div class="retail-kpi-value">{len(df):,} Rows</div>
                <div class="retail-kpi-sub">{len(df.columns)} Features</div>
            </div>
        """, unsafe_allow_html=True)

    fb = ml_eval["feature_breakdown"]
    st.markdown(f"**Feature Composition:** `{fb.get('Numeric Features', 0)}` Numeric &bull; `{fb.get('Categorical Features', 0)}` Categorical &bull; `{fb.get('Date/Time Features', 0)}` Date/Time")

    st.markdown("### Dataset Preparation Checklist")
    for item in ml_eval["checklist"]:
        status_color = "#34D399" if item["status"] == "Pass" else ("#FBBF24" if item["status"] == "Warning" else "#F87171")
        badge_type = "pill-success" if item["status"] == "Pass" else ("pill-warning" if item["status"] == "Warning" else "pill-danger")
        st.markdown(f"""
            <div class="diag-card" style="border-left-color: {status_color};">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <h4 style="color:#F8FAFC; margin:0;">{item['dimension']}</h4>
                    <span class="pill-badge {badge_type}">{item['status']}</span>
                </div>
                <p style="color:#CBD5E1; font-size:13px; margin-top:6px; margin-bottom:4px;">{item['detail']}</p>
                <p style="color:#38BDF8; font-size:12px; margin:0;"><b>Recommended Preprocessing:</b> {item['action']}</p>
            </div>
        """, unsafe_allow_html=True)

    if ml_eval["leakage_warnings"]:
        st.markdown("### ⚠️ Potential Data Leakage Flags")
        for lw in ml_eval["leakage_warnings"]:
            st.warning(lw)

    st.markdown("### Suitable Model Family Recommendations")
    for rec in ml_eval["model_recommendations"]:
        with st.expander(f"⭐ {rec['model']} &mdash; {rec['suitability']}"):
            st.markdown(f"**Why this algorithm is recommended for your data:** {rec['why']}")

    st.info(f"ℹ️ **Evaluation Limitations**: {ml_eval['limitations']}")


# ==========================================
# 20. SMART CLEANING & EXPORT PAGE
# ==========================================

elif selected_page in ["🧹 Smart Cleaning & Data Export", "Smart Cleaning"]:
    st.subheader("🧹 Smart Dataset Cleaning & Data Export")
    st.caption(f"Domain-tailored smart data cleaning with before-and-after reconciliation ({active_workspace})")

    st.markdown("### Select Cleaning Operations")

    if active_workspace == "Healthcare":
        col_cl1, col_cl2 = st.columns(2)
        with col_cl1:
            fill_missing = st.checkbox("Handle missing clinical & demographic values", value=True, key="fill_missing")
            remove_duplicates = st.checkbox("Remove duplicate patient encounter rows", value=True, key="remove_duplicates")
        with col_cl2:
            fix_ages = st.checkbox("Correct invalid/out-of-range patient ages (<0 or >120)", value=True, key="fix_ages")
            fix_dates = st.checkbox("Reconcile inconsistent dates (discharge before admission)", value=True, key="fix_dates")

        with st.expander("📋 Review Proposed Clinical Cleaning Changes"):
            st.markdown("- **Missing Values**: Imputes numeric with median, categoricals with mode (never mutates raw data).")
            st.markdown("- **Duplicates**: Identifies exact repeat encounter records and keeps first occurrence.")
            st.markdown("- **Age Bounds**: Resets negative or >120 ages to the department median age.")
            st.markdown("- **Date Inconsistency**: Automatically transposes inverted admission/discharge dates.")

        if st.button("Execute Clinical Cleaning Pipeline", use_container_width=True):
            cleaned_df = clean_healthcare_dataset(
                df,
                health_mapping,
                fill_missing=fill_missing,
                remove_duplicates=remove_duplicates,
                fix_ages=fix_ages,
                fix_dates=fix_dates
            )
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Clinical dataset cleaning completed successfully!")

    elif active_workspace == "Education":
        col_cl1, col_cl2 = st.columns(2)
        with col_cl1:
            fill_missing = st.checkbox("Handle missing academic & student values", value=True, key="fill_missing_edu")
            remove_duplicates = st.checkbox("Remove duplicate student records", value=True, key="remove_dup_edu")
        with col_cl2:
            fix_marks = st.checkbox("Correct out-of-range student marks (<0 or >Max)", value=True, key="fix_marks_edu")
            fix_att = st.checkbox("Correct out-of-range attendance percentages", value=True, key="fix_att_edu")

        with st.expander("📋 Review Proposed Academic Cleaning Changes"):
            st.markdown("- **Missing Values**: Imputes numeric marks/attendance with cohort medians, text fields with defaults.")
            st.markdown("- **Duplicates**: Removes duplicate student assessment entries.")
            st.markdown("- **Marks Range**: Clamps out-of-range marks to valid [0, Max Marks] interval.")
            st.markdown("- **Attendance Range**: Bounds attendance percentages to valid [0%, 100%] scale.")

        if st.button("Execute Academic Cleaning Pipeline", use_container_width=True):
            cleaned_df = clean_education_dataset(
                df,
                edu_mapping,
                fill_missing=fill_missing,
                remove_duplicates=remove_duplicates,
                fix_marks=fix_marks,
                fix_attendance=fix_att
            )
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Academic dataset cleaning completed successfully!")

    elif active_workspace == "Business & Finance":
        col_cl1, col_cl2 = st.columns(2)
        with col_cl1:
            fill_missing = st.checkbox("Handle missing financial and ledger values", value=True, key="fill_missing_bf")
            remove_duplicates = st.checkbox("Remove duplicate financial transactions", value=True, key="remove_dup_bf")
        with col_cl2:
            fix_negatives = st.checkbox("Reconcile negative revenue/expense entries", value=True, key="fix_neg_bf")
            fix_unassigned = st.checkbox("Assign default general categories to unassigned entries", value=True, key="fix_unassigned_bf")

        with st.expander("📋 Review Proposed Financial Cleaning Changes"):
            st.markdown("- **Missing Values**: Replaces null monetary amounts with 0.00; categoricals with default cost centers.")
            st.markdown("- **Duplicates**: Eliminates redundant transaction ledger records.")
            st.markdown("- **Negative Figures**: Converts abnormal negative revenue/expense entries to absolute magnitudes.")
            st.markdown("- **Classification**: Standardizes unassigned departments and payment methods.")

        if st.button("Execute Financial Cleaning Pipeline", use_container_width=True):
            cleaned_df = clean_business_finance_dataset(
                df,
                bf_mapping,
                fill_missing=fill_missing,
                remove_duplicates=remove_duplicates,
                fix_negatives=fix_negatives,
                fix_unassigned=fix_unassigned
            )
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Financial dataset cleaning completed successfully!")

    else: # Retail Store
        fill_missing = st.checkbox("Handle missing values", key="fill_missing")
        remove_duplicates = st.checkbox("Remove duplicate rows", key="remove_duplicates")

        if st.button("Execute Retail Cleaning Pipeline", use_container_width=True):
            cleaned_df = clean_dataset(df, fill_missing, remove_duplicates)
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Dataset cleaning completed successfully!")

    cleaned_df = st.session_state.get("cleaned_df", None)

    if cleaned_df is not None:
        st.divider()
        st.subheader("Cleaned Dataset Reconciliation")
        st.dataframe(cleaned_df.head(10), use_container_width=True)

        cleaned_outliers = detect_outliers(cleaned_df)
        cleaned_consistency = check_date_consistency(cleaned_df)
        cleaned_health = calculate_health_score(cleaned_df, cleaned_outliers, cleaned_consistency)
        cleaned_score = cleaned_health["score"]
        cleaned_missing = int(cleaned_df.isnull().sum().sum())
        cleaned_duplicates = int(cleaned_df.duplicated().sum())

        st.markdown("### Before vs After Metric Comparison")
        comparison = pd.DataFrame({
            "Metric": ["Dataset Health Score", "Missing Values", "Duplicate Rows"],
            "Original Dataset": [f"{original_score}/100", f"{original_missing:,}", f"{original_duplicates:,}"],
            "Cleaned Dataset": [f"{cleaned_score}/100", f"{cleaned_missing:,}", f"{cleaned_duplicates:,}"],
            "Improvement": [f"+{round(cleaned_score - original_score, 1)} pts", f"-{original_missing - cleaned_missing:,} nulls", f"-{original_duplicates - cleaned_duplicates:,} dups"]
        })
        st.dataframe(comparison, use_container_width=True)

        cleaned_csv = cleaned_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Cleaned Dataset (CSV)",
            data=cleaned_csv,
            file_name=f"DataDoctor_Cleaned_{active_workspace.replace(' ', '_')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.info("Select your desired cleaning operations above and click Execute Pipeline to preview the cleaned dataset.")


# ==========================================
# 21. GLOBAL FOOTER
# ==========================================

st.divider()
st.markdown("""
<div style="text-align:center; padding:20px;">
    <p style="color:#38BDF8; font-weight:700; margin-bottom:4px; font-size:14px;">
        DataDoctor AI &bull; Enterprise Multi-Industry Intelligence Platform
    </p>
    <p style="color:#64748B; font-size:12px; margin:0;">
        Intelligent Dataset Health, Automated Anomaly Detection, Scenario Simulations & Executive Reporting
    </p>
</div>
""", unsafe_allow_html=True)
