
import streamlit as st

import pandas as pd

import plotly.express as px

import plotly.graph_objects as go

import os

from analyzer import (

    detect_outliers,

    check_date_consistency,

    check_ml_readiness

)

from cleaner import clean_dataset

from scoring import calculate_health_score

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





# ==========================================

# 1. PAGE CONFIGURATION

# ==========================================



st.set_page_config(

    page_title="DataDoctor AI",

    page_icon="D",

    layout="wide",

    initial_sidebar_state="expanded"

)





# ==========================================

# 2. CUSTOM CSS

# ==========================================



st.markdown("""

<style>



.stApp {

    background-color: #0B1120;

    color: #E2E8F0;

}



[data-testid="stSidebar"] {

    background-color: #111B2E;

    border-right: 1px solid #26354D;

}



.block-container {

    padding-top: 2rem;

    padding-bottom: 3rem;

    max-width: 1450px;

}



h1, h2, h3 {

    color: #F1F5F9 !important;

}



p, label {

    color: #CBD5E1;

}



[data-testid="stMetric"] {

    background-color: #16243A;

    border: 1px solid #263B59;

    padding: 20px;

    border-radius: 12px;

}



[data-testid="stMetricLabel"] {

    color: #94A3B8 !important;

}



[data-testid="stMetricValue"] {

    color: #38BDF8 !important;

}



.stButton > button {

    background-color: #0284C7;

    color: white;

    border: none;

    border-radius: 8px;

    padding: 0.55rem 1.2rem;

    font-weight: 600;

}



.stButton > button:hover {

    background-color: #0369A1;

    color: white;

    border: none;

}



.stDownloadButton > button {

    background-color: #164E63;

    color: white;

    border: 1px solid #0E7490;

    border-radius: 8px;

}



.stDownloadButton > button:hover {

    background-color: #0E7490;

    color: white;

}



[data-testid="stDataFrame"] {

    border: 1px solid #263B59;

    border-radius: 10px;

}



hr {

    border-color: #26354D;

}



.header-box {

    background: linear-gradient(120deg, #14243D, #102E46);

    border: 1px solid #1E4968;

    padding: 25px;

    border-radius: 15px;

    margin-bottom: 25px;

}



.header-title {

    color: #F8FAFC;

    font-size: 32px;

    font-weight: 750;

    margin-bottom: 5px;

}



.header-subtitle {

    color: #7DD3FC;

    font-size: 15px;

}



.section-box {

    background-color: #16243A;

    border: 1px solid #263B59;

    padding: 18px;

    border-radius: 12px;

    margin-bottom: 15px;

}



.small-text {

    color: #94A3B8;

    font-size: 13px;

}



.status-good {

    color: #34D399;

    font-weight: 600;

}



.status-warning {

    color: #FBBF24;

    font-weight: 600;

}



.status-bad {

    color: #F87171;

    font-weight: 600;

}

.retail-kpi-card {
    background: linear-gradient(135deg, #16243A 0%, #1A2D48 100%);
    border: 1px solid #263B59;
    border-radius: 12px;
    padding: 18px;
    margin-bottom: 12px;
    position: relative;
    border-top: 3px solid #38BDF8;
}

.retail-kpi-card.profit {
    border-top-color: #34D399;
}

.retail-kpi-card.warning {
    border-top-color: #FBBF24;
}

.retail-kpi-card.alert {
    border-top-color: #F87171;
}

.retail-kpi-label {
    color: #94A3B8;
    font-size: 13px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 6px;
}

.retail-kpi-value {
    color: #F8FAFC;
    font-size: 28px;
    font-weight: 750;
    line-height: 1.2;
}

.retail-kpi-sub {
    color: #64748B;
    font-size: 12px;
    margin-top: 5px;
}

.pill-badge {
    display: inline-block;
    padding: 3px 9px;
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

</style>

""", unsafe_allow_html=True)






# ==========================================
# 3. INDUSTRY WORKSPACE SELECTION
# ==========================================

# Store the selected workspace between Streamlit reruns.
if "workspace" not in st.session_state:
    st.session_state["workspace"] = None

# Show this screen until the user selects a workspace.
if st.session_state["workspace"] is None:

    st.markdown("""
        <div class="header-box">
            <div class="header-title">Welcome to DataDoctor AI</div>
            <div class="header-subtitle">
                Choose your industry to access specialized data analysis tools.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.subheader("Select Your Workspace")
    st.write(
        "Choose the workspace that best matches the type of data "
        "you want to analyze."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Retail Store</h3>
                <p>Analyze sales, revenue, profit, products, inventory and customers.</p>
                <p style="color:#34D399; font-weight:600;">Available</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Retail Store", key="enter_retail",
                     use_container_width=True):
            st.session_state["workspace"] = "Retail Store"
            st.rerun()

    with col2:
        st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Healthcare</h3>
                <p>Analyze patient admissions, appointments, clinical diagnoses, doctors, and bed flow.</p>
                <p style="color:#34D399; font-weight:600;">Available</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Healthcare", key="enter_healthcare",
                     use_container_width=True):
            st.session_state["workspace"] = "Healthcare"
            st.rerun()

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Education</h3>
                <p>Analyze student performance, attendance, grades, classes, departments, and academic trends.</p>
                <p style="color:#34D399; font-weight:600;">Available</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Education", key="enter_education",
                     use_container_width=True):
            st.session_state["workspace"] = "Education"
            st.rerun()

    with col4:
        st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Business & Finance</h3>
                <p>Analyze revenue, expenses, net profit, cash flow, budgets, transactions, and financial health.</p>
                <p style="color:#34D399; font-weight:600;">Available</p>
            </div>
        """, unsafe_allow_html=True)

        if st.button("Enter Business & Finance", key="enter_finance",
                     use_container_width=True):
            st.session_state["workspace"] = "Business & Finance"
            st.rerun()

    st.stop()



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

            number={

                "suffix": "/100",

                "font": {

                    "size": 38,

                    "color": "#F1F5F9"

                }

            },

            title={

                "text": "Dataset Health Score",

                "font": {

                    "size": 18,

                    "color": "#CBD5E1"

                }

            },

            gauge={

                "axis": {

                    "range": [0, 100],

                    "tickcolor": "#94A3B8"

                },

                "bar": {

                    "color": gauge_color

                },

                "bgcolor": "#16243A",

                "borderwidth": 0,

                "steps": [

                    {

                        "range": [0, 50],

                        "color": "#3B2633"

                    },

                    {

                        "range": [50, 80],

                        "color": "#3B3525"

                    },

                    {

                        "range": [80, 100],

                        "color": "#183B36"

                    }

                ]

            }

        )

    )



    fig.update_layout(

        paper_bgcolor="#0B1120",

        plot_bgcolor="#0B1120",

        height=300,

        margin=dict(l=25, r=25, t=65, b=20)

    )



    return fig





# ==========================================

# 4. SIDEBAR

# ==========================================



with st.sidebar:



    st.markdown(

        "<h2 style='color:#38BDF8;'>DataDoctor AI</h2>",

        unsafe_allow_html=True

    )



    active_workspace = st.session_state.get("workspace", "Retail Store")
    st.caption("Intelligent Dataset Health System")
    st.info(f"Current Workspace: {active_workspace}")

    if st.button("Change Workspace", use_container_width=True):
        st.session_state["workspace"] = None
        st.session_state.pop("cleaned_df", None)
        st.rerun()

    st.divider()


    st.subheader("Navigation")

    if active_workspace == "Healthcare":
        primary_nav = "Clinical Analytics"
    elif active_workspace == "Education":
        primary_nav = "Academic Analytics"
    elif active_workspace == "Business & Finance":
        primary_nav = "Financial Analytics"
    else:
        primary_nav = "Store Analytics"

    nav_options = [
        primary_nav,
        "Data Diagnosis",
        "Smart Cleaning",
        "ML Readiness",
        "Reports"
    ]

    selected_page = st.radio(
        "Go to",
        nav_options,
        index=0,
        label_visibility="collapsed"
    )

    st.divider()


    st.subheader("Dataset Source")

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
            st.session_state.pop("cleaned_df", None)
            st.rerun()

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"]
    )

    if uploaded_file is not None:
        st.session_state[sample_flag_key] = False

    st.divider()


    st.markdown(
        f"""
        <p class='small-text'>
        DataDoctor AI<br>
        {active_workspace} Intelligence Platform
        </p>
        """,
        unsafe_allow_html=True
    )





# ==========================================
# 5. MAIN HEADER
# ==========================================

if active_workspace == "Healthcare":
    st.markdown("""
    <div class="header-box">
        <div class="header-title">
            Healthcare Operations & Clinical Analytics
        </div>
        <div class="header-subtitle">
            Intelligent Patient Flow, Appointments, Admissions & Diagnoses Dashboard
        </div>
        <p style="color:#CBD5E1; margin-top:15px;">
            Analyze patient volume, appointment fulfillment, bed occupancy, doctor workloads, clinical diagnoses, 
            and diagnose EHR data health with DataDoctor AI.
        </p>
    </div>
    """, unsafe_allow_html=True)
elif active_workspace == "Education":
    st.markdown("""
    <div class="header-box">
        <div class="header-title">
            Academic & Institutional Performance Analytics
        </div>
        <div class="header-subtitle">
            Intelligent Student Performance, Attendance, Grade Distribution & Department Dashboard
        </div>
        <p style="color:#CBD5E1; margin-top:15px;">
            Analyze student marks, pass rates, attendance trends, class and department breakdowns, 
            and diagnose academic data health with DataDoctor AI.
        </p>
    </div>
    """, unsafe_allow_html=True)
elif active_workspace == "Business & Finance":
    st.markdown("""
    <div class="header-box">
        <div class="header-title">
            Corporate Finance & Business Performance Analytics
        </div>
        <div class="header-subtitle">
            Intelligent Revenue, Expenses, Net Profit, Budget & Cash Flow Dashboard
        </div>
        <p style="color:#CBD5E1; margin-top:15px;">
            Analyze company revenue streams, expense breakdowns, profit margins, cash flow liquidity, budget variances, 
            and diagnose financial data health with DataDoctor AI.
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <div class="header-box">
        <div class="header-title">
            Retail Store Analytics
        </div>
        <div class="header-subtitle">
            Intelligent Sales, Revenue, Profit & Inventory Optimization Dashboard
        </div>
        <p style="color:#CBD5E1; margin-top:15px;">
            Analyze store revenue, sales velocity, product profitability, customer buying patterns, 
            and diagnose data health with DataDoctor AI.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# 6. CHECK DATASET UPLOAD
# ==========================================

has_sample = (
    st.session_state.get(sample_flag_key, False)
    and os.path.exists(sample_file_path)
)

if uploaded_file is None and not has_sample:
    if active_workspace == "Healthcare":
        st.info("Upload your patient visits or hospital records CSV from the sidebar, or load the sample healthcare dataset to begin.")
        st.subheader("What the Healthcare Workspace provides")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Patient & Clinical Demographics</h3>
                <p>Analyze patient volume, age pyramid distribution, gender breakdown, and new vs repeat encounters.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Appointments & Bed Flow</h3>
                <p>Track completed vs cancelled visits, doctor workloads, admission trends, length of stay, and active inpatients.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Diagnoses & Data Integrity</h3>
                <p>Monitor top disease frequencies, treatments, and validate EHR data for inconsistent dates, invalid ages, and anomalies.</p>
            </div>
            """, unsafe_allow_html=True)

        if os.path.exists(sample_file_path):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Load Sample Healthcare Dataset to Explore", key="load_sample_hero", use_container_width=True):
                st.session_state["use_sample_healthcare"] = True
                st.rerun()

    elif active_workspace == "Education":
        st.info("Upload your student scores, attendance, or enrollment CSV from the sidebar, or load the sample education dataset to begin.")
        st.subheader("What the Education Workspace provides")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Performance & Grading</h3>
                <p>Analyze student marks, grade distribution, pass/fail rates, subject strengths, and identify students needing support.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Attendance & Engagement</h3>
                <p>Monitor attendance percentages, class comparisons, monthly attendance trends, and automated low-attendance alerts.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Classes & Faculty</h3>
                <p>Evaluate department and class headcounts, teacher workloads, academic enrollment trends, and dataset health.</p>
            </div>
            """, unsafe_allow_html=True)

        if os.path.exists(sample_file_path):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Load Sample Education Dataset to Explore", key="load_sample_hero", use_container_width=True):
                st.session_state["use_sample_education"] = True
                st.rerun()

    elif active_workspace == "Business & Finance":
        st.info("Upload your financial ledger, sales, or expense CSV from the sidebar, or load the sample business & finance dataset to begin.")
        st.subheader("What the Business & Finance Workspace provides")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Revenue & Profitability</h3>
                <p>Track total revenue, expenses, net profit, margins, product profitability, and month-over-month growth rates.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Budget & Expense Control</h3>
                <p>Monitor department expenditures, expense categories, budget utilization, and detect over-budget variances.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Cash Flow & Transactions</h3>
                <p>Analyze net cash inflows, payment method breakdowns, transaction success rates, and customer spending habits.</p>
            </div>
            """, unsafe_allow_html=True)

        if os.path.exists(sample_file_path):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Load Sample Financial Dataset to Explore", key="load_sample_hero", use_container_width=True):
                st.session_state["use_sample_finance"] = True
                st.rerun()

    else:
        st.info("Upload your store sales or inventory CSV from the sidebar, or load the sample retail dataset to begin.")
        st.subheader("What the Retail Store Workspace provides")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Sales & Profit Analysis</h3>
                <p>Track total revenue, sales volume, profit margins, and daily/monthly revenue trends.</p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Products & Inventory</h3>
                <p>Identify top-selling products, category revenue contribution, and low-stock inventory alerts.</p>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="section-box">
                <h3 style="color:#38BDF8;">Data Quality Check</h3>
                <p>Leverage DataDoctor AI to diagnose data anomalies, missing values, duplicates, and health score.</p>
            </div>
            """, unsafe_allow_html=True)

        if os.path.exists(sample_file_path):
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("🚀 Load Sample Retail Dataset to Explore", key="load_sample_hero", use_container_width=True):
                st.session_state["use_sample_retail"] = True
                st.rerun()

    st.stop()


# ==========================================
# 7. READ DATASET
# ==========================================

try:
    if uploaded_file is not None:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        dataset_name = uploaded_file.name
        file_id = (uploaded_file.name, uploaded_file.size)
    else:
        df = pd.read_csv(sample_file_path)
        if active_workspace == "Healthcare":
            dataset_name = "healthcare_sample.csv (Sample Dataset)"
        elif active_workspace == "Education":
            dataset_name = "education_sample.csv (Sample Dataset)"
        elif active_workspace == "Business & Finance":
            dataset_name = "business_finance_sample.csv (Sample Dataset)"
        else:
            dataset_name = "retail_store_sample.csv (Sample Dataset)"
        file_id = (f"sample_{active_workspace.lower().replace(' ', '_').replace('&', 'and')}", len(df))

except Exception as error:



    st.error(f"Unable to read CSV file: {error}")

    st.stop()





if df.empty or len(df.columns) == 0:



    st.error("The uploaded dataset is empty.")

    st.stop()





# Reset cleaning result if another file is uploaded

if st.session_state.get("file_id") != file_id:



    st.session_state["file_id"] = file_id

    st.session_state.pop("cleaned_df", None)





# ==========================================

# 8. DATASET ANALYSIS

# ==========================================



outlier_results = detect_outliers(df)



consistency_results = check_date_consistency(df)



ml_results = check_ml_readiness(df)



health_result = calculate_health_score(

    df,

    outlier_results,

    consistency_results

)



original_score = health_result["score"]



original_missing = int(

    df.isnull().sum().sum()

)



original_duplicates = int(

    df.duplicated().sum()

)



original_outliers = get_outlier_count(

    outlier_results

)



original_consistency = len(

    consistency_results

)

# Industry-specific processing
if active_workspace == "Healthcare":
    health_mapping_info = map_healthcare_columns(df)
    health_df, health_mapping, health_meta = prepare_healthcare_data(df, health_mapping_info["mapping"])
    health_summary = summarize_healthcare(health_df)
    health_quality = check_healthcare_quality(health_df, health_mapping)
elif active_workspace == "Education":
    edu_mapping_info = map_education_columns(df)
    edu_df, edu_mapping, edu_meta = prepare_education_data(df, edu_mapping_info["mapping"])
    edu_summary = summarize_education(edu_df)
    edu_quality = check_education_quality(edu_df, edu_mapping)
elif active_workspace == "Business & Finance":
    bf_mapping_info = map_business_finance_columns(df)
    bf_df, bf_mapping, bf_meta = prepare_business_finance_data(df, bf_mapping_info["mapping"])
    bf_summary = summarize_business_finance(bf_df)
    bf_quality = check_business_finance_quality(bf_df, bf_mapping)
else:
    mapping_info = map_store_columns(df)
    store_df, store_mapping, store_meta = prepare_sales_data(df, mapping_info["mapping"])
    store_summary = summarize_sales(store_df)


# ==========================================
# 9. GET CLEANED DATASET
# ==========================================

cleaned_df = st.session_state.get(
    "cleaned_df",
    None
)


# ==========================================
# 10. RETAIL STORE DASHBOARD
# ==========================================

if active_workspace == "Retail Store" and selected_page in ["Store Analytics", "Overview"]:

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
    tab_trends, tab_profit, tab_products, tab_categories, tab_inventory, tab_customers, tab_quality = st.tabs([
        "📈 Sales Trends",
        "💰 Profit Analysis",
        "🏆 Top-Selling Products",
        "🏷️ Category Performance",
        "📦 Inventory Health",
        "👥 Customer Insights",
        "🛡️ Data Quality Check"
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

elif active_workspace == "Healthcare" and selected_page in ["Clinical Analytics", "Overview"]:

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
    tab_pat, tab_apt, tab_doc, tab_adm, tab_diag, tab_qual, tab_prev = st.tabs([
        "👥 Patient Analytics",
        "📅 Appointment Analytics",
        "🩺 Doctor & Department Analytics",
        "🏥 Admission & Discharge Analytics",
        "🔬 Diagnosis & Treatment Trends",
        "🛡️ Healthcare Data Quality",
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

elif active_workspace == "Education" and selected_page in ["Academic Analytics", "Overview"]:

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
    tab_perf, tab_att, tab_cls, tab_trend, tab_qual, tab_prev = st.tabs([
        "🎯 Student Performance",
        "📅 Attendance Analytics",
        "🏫 Class & Department",
        "📈 Education Trends",
        "🛡️ Education Data Quality",
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

elif active_workspace == "Business & Finance" and selected_page in ["Financial Analytics", "Overview"]:

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
    tab_rev, tab_exp, tab_pl, tab_txn, tab_budg, tab_cf, tab_fqual, tab_fprev = st.tabs([
        "💰 Revenue Analytics",
        "📉 Expense Analytics",
        "⚖️ Profit & Loss",
        "💳 Transactions & Sales",
        "🎯 Budget vs Actual",
        "🌊 Cash Flow & Liquidity",
        "🛡️ Financial Data Quality",
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

# 11. DATA DIAGNOSIS PAGE

# ==========================================



elif selected_page == "Data Diagnosis":



    st.subheader("Smart Data Diagnosis")



    st.write(

        "Identify data quality issues detected in your dataset."

    )



    # MISSING VALUES

    st.markdown("### Missing Values Analysis")



    missing_data = pd.DataFrame({



        "Column": df.columns,



        "Missing Values": df.isnull().sum().values,



        "Missing Percentage": (

            df.isnull().mean().values * 100

        ).round(2)



    })



    st.dataframe(

        missing_data,

        use_container_width=True

    )



    st.divider()




    # DUPLICATES

    st.markdown("### Duplicate Records")



    if original_duplicates > 0:



        st.warning(

            f"{original_duplicates} duplicate row(s) detected."

        )



        st.dataframe(

            df[df.duplicated(keep=False)],

            use_container_width=True

        )



    else:



        st.success("No duplicate records detected.")



    st.divider()




    # OUTLIERS

    st.markdown("### Outlier Detection")



    if outlier_results.empty:



        st.info(

            "No suitable numeric columns available for outlier detection."

        )



    else:



        st.dataframe(

            outlier_results,

            use_container_width=True

        )



    st.divider()




    # CONSISTENCY GUARDIAN

    st.markdown("### Consistency Guardian")



    if consistency_results.empty:



        st.success(

            "No date consistency issues detected."

        )



    else:



        st.warning(

            f"{len(consistency_results)} date consistency issue(s) detected."

        )



        st.dataframe(

            consistency_results,

            use_container_width=True

        )



    st.divider()




    # SCORE BREAKDOWN

    st.markdown("### Health Score Breakdown")



    score_breakdown = pd.DataFrame({



        "Issue": [

            "Missing Values",

            "Duplicate Rows",

            "Outliers",

            "Date Inconsistency"

        ],



        "Score Deduction": [

            health_result["missing_penalty"],

            health_result["duplicate_penalty"],

            health_result["outlier_penalty"],

            health_result["consistency_penalty"]

        ]



    })



    st.dataframe(

        score_breakdown,

        use_container_width=True

    )





# ==========================================

# 12. SMART CLEANING PAGE

# ==========================================



elif selected_page == "Smart Cleaning":



    st.subheader("Smart Dataset Cleaning")



    st.write(

        "Choose the cleaning operations you want to apply."

    )



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

        if st.button("Clean Dataset", use_container_width=True):
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

        if st.button("Clean Dataset", use_container_width=True):
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

        if st.button("Clean Dataset", use_container_width=True):
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
    else:
        fill_missing = st.checkbox("Handle missing values", key="fill_missing")
        remove_duplicates = st.checkbox("Remove duplicate rows", key="remove_duplicates")

        if st.button("Clean Dataset", use_container_width=True):
            cleaned_df = clean_dataset(df, fill_missing, remove_duplicates)
            st.session_state["cleaned_df"] = cleaned_df
            st.success("Dataset cleaning completed successfully!")



    # GET LATEST CLEANED DATASET

    cleaned_df = st.session_state.get(

        "cleaned_df",

        None

    )



    if cleaned_df is not None:



        st.divider()




        st.subheader("Cleaned Dataset Preview")



        st.dataframe(

            cleaned_df.head(10),

            use_container_width=True

        )



        # CLEANED DATASET ANALYSIS

        cleaned_outliers = detect_outliers(

            cleaned_df

        )



        cleaned_consistency = check_date_consistency(

            cleaned_df

        )



        cleaned_health = calculate_health_score(

            cleaned_df,

            cleaned_outliers,

            cleaned_consistency

        )



        cleaned_score = cleaned_health["score"]



        cleaned_missing = int(

            cleaned_df.isnull().sum().sum()

        )



        cleaned_duplicates = int(

            cleaned_df.duplicated().sum()

        )



        cleaned_outlier_count = get_outlier_count(

            cleaned_outliers

        )



        cleaned_consistency_count = len(

            cleaned_consistency

        )



        # BEFORE VS AFTER

        st.divider()




        st.subheader("Before vs After Analysis")



        comparison = pd.DataFrame({



            "Metric": [

                "Missing Values",

                "Duplicate Rows",

                "Outliers",

                "Date Inconsistencies",

                "Health Score"

            ],



            "Before Cleaning": [

                original_missing,

                original_duplicates,

                original_outliers,

                original_consistency,

                original_score

            ],



            "After Cleaning": [

                cleaned_missing,

                cleaned_duplicates,

                cleaned_outlier_count,

                cleaned_consistency_count,

                cleaned_score

            ]



        })



        st.dataframe(

            comparison,

            use_container_width=True

        )



        # IMPROVEMENT

        improvement = round(

            cleaned_score - original_score,

            2

        )



        st.metric(

            "Health Score Improvement",

            f"{improvement} points"

        )



        if improvement > 0:



            st.success(

                "Dataset health improved after cleaning."

            )



        elif improvement == 0:



            st.info(

                "The dataset health score remains unchanged."

            )



        else:



            st.warning(

                "Dataset health score decreased. Review the cleaning results."

            )



        # VISUAL COMPARISON

        chart_data = pd.DataFrame({



            "Stage": [

                "Before Cleaning",

                "After Cleaning"

            ],



            "Health Score": [

                original_score,

                cleaned_score

            ]



        })



        fig = px.bar(

            chart_data,

            x="Stage",

            y="Health Score",

            color="Stage",

            title="Dataset Health Improvement",

            range_y=[0, 100]

        )



        fig.update_layout(

            paper_bgcolor="#0B1120",

            plot_bgcolor="#16243A",

            font_color="#E2E8F0",

            showlegend=False

        )



        st.plotly_chart(

            fig,

            use_container_width=True

        )



    else:



        st.info(

            "Select your cleaning operations and click Clean Dataset to see the results."

        )





# ==========================================

# 13. ML READINESS PAGE

# ==========================================



elif selected_page == "ML Readiness":



    st.subheader("Machine Learning Readiness Checker")



    st.write(

        "Identify potential preprocessing requirements "

        "before using your dataset in a machine learning workflow."

    )



    # ML SUMMARY

    col1, col2 = st.columns(2)



    with col1:



        st.metric(

            "Potential Issues",

            len(ml_results)

        )



    with col2:



        categorical_count = len(

            df.select_dtypes(

                include=["object", "category", "string"]

            ).columns

        )



        st.metric(

            "Categorical Columns",

            categorical_count

        )



    st.divider()




    if ml_results.empty:



        st.success(

            "No potential ML readiness issues detected by the current checks."

        )



    else:



        st.warning(

            "Review the following potential ML preprocessing issues."

        )



        st.dataframe(

            ml_results,

            use_container_width=True

        )



    st.divider()




    st.markdown("### Dataset Data Types")



    data_types = pd.DataFrame({



        "Column": df.columns,



        "Data Type": df.dtypes.astype(str).values,



        "Unique Values": df.nunique().values



    })



    st.dataframe(

        data_types,

        use_container_width=True

    )



    st.info(

        "This is a preliminary readiness check. "

        "It does not guarantee that a dataset is suitable for every ML model."

    )





# ==========================================

# 14. REPORTS PAGE

# ==========================================



elif selected_page == "Reports":



    st.subheader("Dataset Health Reports")



    st.write(

        "Download your dataset and its health analysis report."

    )



    cleaned_df = st.session_state.get(

        "cleaned_df",

        None

    )



    if cleaned_df is not None:



        cleaned_outliers = detect_outliers(

            cleaned_df

        )



        cleaned_consistency = check_date_consistency(

            cleaned_df

        )



        cleaned_health = calculate_health_score(

            cleaned_df,

            cleaned_outliers,

            cleaned_consistency

        )



        cleaned_score = cleaned_health["score"]



        cleaned_missing = int(

            cleaned_df.isnull().sum().sum()

        )



        cleaned_duplicates = int(

            cleaned_df.duplicated().sum()

        )



        cleaned_outlier_count = get_outlier_count(

            cleaned_outliers

        )



        cleaned_consistency_count = len(

            cleaned_consistency

        )



        improvement = round(

            cleaned_score - original_score,

            2

        )



        st.success(

            "A cleaned dataset is available for download."

        )



    else:



        # If cleaning has not been performed,

        # use the original dataset as the current dataset.

        cleaned_df = df.copy()



        cleaned_score = original_score



        cleaned_missing = original_missing



        cleaned_duplicates = original_duplicates



        cleaned_outlier_count = original_outliers



        cleaned_consistency_count = original_consistency



        improvement = 0



        st.info(

            "Cleaning has not been performed yet. "

            "The report will show the original dataset as the current dataset."

        )



    st.divider()




    # REPORT SUMMARY

    col1, col2, col3 = st.columns(3)



    with col1:



        st.metric(

            "Original Health Score",

            f"{original_score}/100"

        )



    with col2:



        st.metric(

            "Current Health Score",

            f"{cleaned_score}/100"

        )



    with col3:



        st.metric(

            "Health Improvement",

            f"{improvement} points"

        )



    st.divider()




    # DOWNLOAD DATASET

    st.markdown("### Download Dataset")



    cleaned_csv = cleaned_df.to_csv(

        index=False

    )



    st.download_button(

        label="Download Current Dataset",

        data=cleaned_csv,

        file_name="datadoctor_cleaned_dataset.csv",

        mime="text/csv",

        use_container_width=True

    )



    st.divider()




    # GENERATE REPORT

    st.markdown("### Dataset Health Report")



    if active_workspace == "Healthcare":
        domain_summary_text = f"""HEALTHCARE OPERATIONS & CLINICAL SUMMARY
Total Patients: {health_summary['unique_patients']:,}
Total Appointments: {health_summary['total_appointments']:,}
Total Admissions: {health_summary['total_admissions']:,}
Total Discharges: {health_summary['total_discharges']:,}
Active Inpatients: {health_summary['current_inpatients']:,}
Average Length of Stay: {health_summary['alos']} days
Total Billing: ${health_summary['total_billing']:,.2f}"""
    elif active_workspace == "Education":
        domain_summary_text = f"""ACADEMIC & INSTITUTIONAL SUMMARY
Total Students: {edu_summary['total_students']:,}
Total Teachers: {edu_summary['total_teachers']:,}
Total Classes: {edu_summary['total_classes']:,}
Total Subjects: {edu_summary['total_subjects']:,}
Average Marks: {edu_summary['avg_marks']}%
Pass Percentage: {edu_summary['pass_pct']}%
Average Attendance: {edu_summary['avg_attendance']}%"""
    elif active_workspace == "Business & Finance":
        domain_summary_text = f"""BUSINESS & FINANCIAL PERFORMANCE SUMMARY
Total Revenue: ${bf_summary['total_revenue']:,.2f}
Total Expenses: ${bf_summary['total_expenses']:,.2f}
Net Profit: ${bf_summary['net_profit']:,.2f}
Profit Margin: {bf_summary['profit_margin']}%
Total Transactions: {bf_summary['total_transactions']:,}
Average Transaction Value: ${bf_summary['atv']:,.2f}
Total Customers: {bf_summary['total_customers']:,}"""
    else:
        domain_summary_text = f"""RETAIL STORE PERFORMANCE SUMMARY
Total Revenue: ${store_summary['total_revenue']:,.2f}
Total Units Sold: {store_summary['total_units']:,}
Total Transactions: {store_summary['total_transactions']:,}
Unique Products: {store_summary['unique_products']:,}
Total Gross Profit: ${store_summary['total_profit']:,.2f}
Profit Margin: {store_summary['profit_margin']}%"""

    report = f"""
DATADOCTOR AI
INTELLIGENT DATASET HEALTH REPORT ({active_workspace.upper()} WORKSPACE)
==========================================

DATASET INFORMATION
Dataset Name: {dataset_name}
Original Rows: {len(df)}
Total Columns: {len(df.columns)}

------------------------------------------
{domain_summary_text}

------------------------------------------

ORIGINAL DATASET DIAGNOSIS

Missing Values: {original_missing}
Duplicate Rows: {original_duplicates}
Outliers: {original_outliers}
Date Consistency Issues: {original_consistency}

Original Health Score: {original_score}/100

------------------------------------------

CURRENT DATASET ANALYSIS

Missing Values: {cleaned_missing}
Duplicate Rows: {cleaned_duplicates}
Outliers: {cleaned_outlier_count}
Date Consistency Issues: {cleaned_consistency_count}

Current Health Score: {cleaned_score}/100

------------------------------------------

HEALTH IMPROVEMENT

Health Score Improvement: {improvement} points

------------------------------------------

ML READINESS

Potential ML Readiness Issues: {len(ml_results)}

------------------------------------------

SCORING INFORMATION

The health score is a project-defined heuristic
based on missing values, duplicate records,
outliers and date consistency issues.

It is not an industry-standard scoring formula.

------------------------------------------

Report generated by DataDoctor AI

"""



    st.download_button(

        label="Download Dataset Health Report",

        data=report,

        file_name="datadoctor_health_report.txt",

        mime="text/plain",

        use_container_width=True

    )





# ==========================================

# 15. FOOTER

# ==========================================



st.divider()



st.markdown("""

<div style="text-align:center; padding:15px;">



    <p style="color:#38BDF8; font-weight:600;">

        DataDoctor AI

    </p>



    <p style="color:#64748B; font-size:12px;">

        Intelligent Dataset Health & Optimization System

    </p>



</div>

""", unsafe_allow_html=True)
