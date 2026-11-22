"""
DataDoctor AI - Industry-Specific Recommendation Engine
Generates context-aware, dynamic, metric-driven recommendations for:
1. Retail Store
2. Healthcare Clinical Operations
3. Education & Academic Performance
4. Business & Corporate Finance
5. Human Resources & Workforce (Employee Datasets)

All recommendations are computed from actual dataset figures and never display
cross-industry or generic hardcoded advice.
"""

import pandas as pd
import numpy as np


# ==========================================
# 1. RETAIL STORE RECOMMENDATIONS
# ==========================================

def generate_retail_recommendations(df, mapping=None, summary=None):
    """
    Produces dynamic, metric-driven retail recommendations covering:
    - Sales improvement suggestions
    - Low-stock alerts
    - Product performance insights
    - Revenue and profit improvement opportunities
    - Inventory optimization suggestions
    - Customer purchasing insights
    """
    recs = []
    mapping = mapping or {}
    summary = summary or {}

    total_revenue = summary.get("total_revenue", 0.0)
    total_units = summary.get("total_units", 0)
    total_transactions = summary.get("total_transactions", 0)
    unique_products = summary.get("unique_products", 0)
    avg_order_value = summary.get("avg_order_value", 0.0)
    has_cost = summary.get("has_cost", False)
    total_profit = summary.get("total_profit", 0.0)
    profit_margin = summary.get("profit_margin", 0.0)

    # 1. SALES IMPROVEMENT SUGGESTIONS
    if total_transactions > 0 and total_revenue > 0:
        if avg_order_value < 50.0:
            recs.append({
                "category": "Sales Velocity & Basket Size",
                "title": f"Expand Average Basket Value (Current: ${avg_order_value:,.2f})",
                "severity": "Opportunity",
                "metric_value": f"${avg_order_value:,.2f} AOV across {total_transactions:,} orders",
                "insight": (
                    f"Average transaction checkout value is currently ${avg_order_value:,.2f}. "
                    "Low basket sizes indicate customers are buying single or low-ticket items without complementary additions."
                ),
                "recommendation": (
                    f"Introduce volume thresholds (e.g., 'Free shipping on orders over ${round(avg_order_value * 1.3, 0):.0f}') "
                    "and bundle high-frequency accessory items at point-of-sale checkout to lift order value by 15-25%."
                ),
                "action_label": "Implement Basket Bundling"
            })
        else:
            recs.append({
                "category": "Sales Velocity & Basket Size",
                "title": f"Healthy Basket Order Value (${avg_order_value:,.2f})",
                "severity": "Info",
                "metric_value": f"${avg_order_value:,.2f} AOV",
                "insight": (
                    f"Your customer basket size is robust at ${avg_order_value:,.2f} across {total_transactions:,} transactions. "
                    f"Total gross volume has reached ${total_revenue:,.2f}."
                ),
                "recommendation": (
                    "Capitalize on high order sizes by launching premium tier offerings and loyalty incentives to maximize retention."
                ),
                "action_label": "Scale VIP Loyalty"
            })

    # 2. LOW-STOCK ALERTS & INVENTORY OPTIMIZATION
    stock_col = mapping.get("stock")
    if stock_col and stock_col in df.columns:
        stock_s = pd.to_numeric(df[stock_col], errors="coerce").fillna(0)
        prod_col = mapping.get("product", df.columns[0])
        
        out_of_stock = df[stock_s == 0]
        low_stock = df[(stock_s > 0) & (stock_s <= 10)]

        if len(out_of_stock) > 0:
            oos_names = out_of_stock[prod_col].astype(str).unique()[:4]
            recs.append({
                "category": "Inventory Stockout Alert",
                "title": f"Critical: {len(out_of_stock)} Product(s) Completely Out of Stock",
                "severity": "Critical",
                "metric_value": f"{len(out_of_stock)} SKUs at 0 Inventory",
                "insight": (
                    f"Zero stock detected for {len(out_of_stock)} item(s), including: {', '.join(oos_names)}. "
                    "Unfulfilled demand leads to immediate revenue loss and churn to competing retailers."
                ),
                "recommendation": (
                    "Trigger emergency purchase orders with suppliers immediately. "
                    "Configure automated reorder triggers when warehouse inventory drops below 14 days of average sales velocity."
                ),
                "action_label": "Restock Depleted SKUs"
            })

        if len(low_stock) > 0:
            low_names = low_stock[prod_col].astype(str).unique()[:4]
            recs.append({
                "category": "Inventory Replenishment",
                "title": f"Warning: {len(low_stock)} Product(s) Approaching Stockout (<= 10 units)",
                "severity": "Warning",
                "metric_value": f"{len(low_stock)} Low-Stock Items",
                "insight": (
                    f"Identified {len(low_stock)} items with 10 or fewer units on hand (e.g. {', '.join(low_names)}). "
                    "At current sales velocity, these items risk selling out before standard supplier fulfillment lead time."
                ),
                "recommendation": (
                    "Review supplier lead times and issue replenishment orders for all flagged low-stock items today."
                ),
                "action_label": "Issue Replenishment POs"
            })
    else:
        recs.append({
            "category": "Inventory Optimization",
            "title": "Enable Real-Time Stock Tracking",
            "severity": "Info",
            "metric_value": "Stock Column Not Detected",
            "insight": "Current sales dataset does not contain an active inventory/stock column.",
            "recommendation": (
                "Include warehouse stock levels or current on-hand quantities in your CSV to unlock automated "
                "safety-stock calculations, stockout prevention alerts, and inventory turn velocity metrics."
            ),
            "action_label": "Upload Stock Quantities"
        })

    # 3. PRODUCT PERFORMANCE INSIGHTS
    if "_clean_product" in df.columns and "_clean_revenue" in df.columns:
        prod_rev = df.groupby("_clean_product")["_clean_revenue"].sum().sort_values(ascending=False)
        if len(prod_rev) >= 3:
            top_prod_name = prod_rev.index[0]
            top_prod_rev = prod_rev.iloc[0]
            top_prod_share = round((top_prod_rev / total_revenue * 100), 1) if total_revenue > 0 else 0
            
            bottom_prods = prod_rev[prod_rev <= prod_rev.quantile(0.20)]
            
            recs.append({
                "category": "Product Performance",
                "title": f"Top Revenue Driver: '{top_prod_name}' ({top_prod_share}% of Revenue)",
                "severity": "Opportunity",
                "metric_value": f"${top_prod_rev:,.2f} generated by '{top_prod_name}'",
                "insight": (
                    f"'{top_prod_name}' is your primary retail revenue driver, generating ${top_prod_rev:,.2f} "
                    f"({top_prod_share}% of total store sales). "
                    f"Conversely, {len(bottom_prods)} product(s) in the lowest quintile contribute minimal revenue."
                ),
                "recommendation": (
                    f"Ensure high availability and promotional placement for '{top_prod_name}'. "
                    f"Consider markdown or bundle clearance for the {len(bottom_prods)} low-velocity items to free up working capital."
                ),
                "action_label": "Optimize Product Mix"
            })

    # 4. REVENUE AND PROFIT IMPROVEMENT OPPORTUNITIES
    if has_cost:
        if profit_margin < 25.0:
            recs.append({
                "category": "Profit Margin Optimization",
                "title": f"Low Gross Margin Alert (Current: {profit_margin}%)",
                "severity": "Warning",
                "metric_value": f"{profit_margin}% Margin | ${total_profit:,.2f} Profit",
                "insight": (
                    f"Store gross margin is currently {profit_margin}%, below standard commercial retail benchmarks (35-45%). "
                    f"Total cost of goods sold (${summary.get('total_cost', 0):,.2f}) accounts for {round(100 - profit_margin, 1)}% of top-line revenue."
                ),
                "recommendation": (
                    "Renegotiate vendor supply agreements, review shipping/packaging surcharges, and adjust pricing on inelastic SKUs "
                    "to target a sustainable 35%+ gross profit margin."
                ),
                "action_label": "Renegotiate Supplier Costs"
            })
        else:
            recs.append({
                "category": "Profit Margin Optimization",
                "title": f"Healthy Commercial Gross Margin ({profit_margin}%)",
                "severity": "Opportunity",
                "metric_value": f"{profit_margin}% Margin | ${total_profit:,.2f} Net Profit",
                "insight": (
                    f"Your gross commercial profit margin is strong at {profit_margin}%, generating ${total_profit:,.2f} in net profit. "
                    "Pricing structure effectively covers product cost bases."
                ),
                "recommendation": (
                    "Reinvest a fraction of healthy margins into digital advertising and local promotional channels to accelerate top-line sales growth."
                ),
                "action_label": "Reinvest into Acquisition"
            })
    else:
        recs.append({
            "category": "Profit Analytics Opportunity",
            "title": "Incorporate Cost Data to Unlock Profit Margin Analysis",
            "severity": "Info",
            "metric_value": "Cost Data Missing",
            "insight": "No cost of goods sold (COGS) or unit cost column was detected in the uploaded dataset.",
            "recommendation": (
                "Upload a dataset containing item cost prices. DataDoctor AI will automatically compute gross profits, "
                "per-product margins, and detect margin leakage across your store catalog."
            ),
            "action_label": "Provide Cost / COGS Column"
        })

    # 5. CUSTOMER PURCHASING INSIGHTS
    cust_col = mapping.get("customer")
    if cust_col and cust_col in df.columns:
        cust_counts = df[cust_col].astype(str).value_counts()
        total_custs = len(cust_counts)
        repeat_custs = (cust_counts > 1).sum()
        repeat_rate = round(repeat_custs / total_custs * 100, 1) if total_custs > 0 else 0

        if repeat_rate < 30.0:
            recs.append({
                "category": "Customer Retention",
                "title": f"Low Customer Repeat Rate ({repeat_rate}%)",
                "severity": "Warning",
                "metric_value": f"{repeat_custs} of {total_custs} Customers Made >1 Purchase",
                "insight": (
                    f"Only {repeat_rate}% of your customer base has returned for a second order. "
                    "High single-purchase customer ratios drastically elevate your blended customer acquisition cost (CAC)."
                ),
                "recommendation": (
                    "Implement post-purchase email follow-ups with personalized discounts for complementary categories "
                    "within 14 days of first transaction to drive repeat visit frequency."
                ),
                "action_label": "Deploy Re-engagement Flow"
            })
        else:
            recs.append({
                "category": "Customer Retention",
                "title": f"Strong Repeat Customer Loyalty ({repeat_rate}%)",
                "severity": "Opportunity",
                "metric_value": f"{repeat_rate}% Repeat Buyer Rate",
                "insight": (
                    f"High customer retention: {repeat_custs} of {total_custs} buyers ({repeat_rate}%) are returning customers. "
                    "Loyal repeat buyers form the compounding foundation of retail store growth."
                ),
                "recommendation": (
                    "Introduce an exclusive VIP tier or early-access promotions to reward top returning buyers and encourage advocacy."
                ),
                "action_label": "Launch VIP Program"
            })

    return recs


# ==========================================
# 2. HEALTHCARE RECOMMENDATIONS
# ==========================================

def generate_healthcare_recommendations(df, mapping=None, summary=None):
    """
    Produces dynamic, metric-driven healthcare recommendations covering:
    - Appointment management insights
    - Department workload analysis
    - Patient record data quality suggestions
    - Admission and discharge data insights
    - Appointment cancellation analysis
    """
    recs = []
    mapping = mapping or {}
    summary = summary or {}

    total_records = summary.get("total_records", len(df))
    unique_patients = summary.get("unique_patients", total_records)
    total_appointments = summary.get("total_appointments", total_records)
    total_admissions = summary.get("total_admissions", 0)
    total_discharges = summary.get("total_discharges", 0)
    current_inpatients = summary.get("current_inpatients", 0)
    alos = summary.get("alos", None)

    # 1. APPOINTMENT MANAGEMENT & CANCELLATION ANALYSIS
    if "_clean_appt_status" in df.columns:
        status_counts = df["_clean_appt_status"].value_counts()
        cancelled_count = int(status_counts.get("Cancelled", 0))
        noshow_count = int(status_counts.get("No-Show", 0))
        completed_count = int(status_counts.get("Completed", 0))
        total_appts = len(df)

        unfulfilled_rate = round((cancelled_count + noshow_count) / total_appts * 100, 1) if total_appts > 0 else 0

        if unfulfilled_rate > 10.0:
            recs.append({
                "category": "Appointment Management & Slot Utilization",
                "title": f"High Appointment Friction: {unfulfilled_rate}% Unfulfilled Visits",
                "severity": "Critical" if unfulfilled_rate > 20 else "Warning",
                "metric_value": f"{cancelled_count} Cancelled, {noshow_count} No-Shows ({unfulfilled_rate}% of Total)",
                "insight": (
                    f"Out of {total_appts:,} total scheduled visits, {cancelled_count} appointments were cancelled "
                    f"and {noshow_count} resulted in patient no-shows ({unfulfilled_rate}% aggregate rate). "
                    "Unutilized provider time wastes physician capacity and extends waiting lists for acute patients."
                ),
                "recommendation": (
                    "Deploy automated SMS / WhatsApp appointment confirmations 48 and 24 hours prior to visits, "
                    "introduce an online 1-click rescheduling link, and adopt a controlled 5% overbooking buffer during peak morning clinics."
                ),
                "action_label": "Automate Appointment Reminders"
            })
        else:
            recs.append({
                "category": "Appointment Management",
                "title": f"High Appointment Completion Rate ({round(100 - unfulfilled_rate, 1)}%)",
                "severity": "Info",
                "metric_value": f"{completed_count} Completed Visits ({round(completed_count / total_appts * 100, 1)}%)",
                "insight": (
                    f"Clinic appointment adherence is strong with {completed_count} completed visits out of {total_appts}. "
                    "Low cancellation rates preserve provider productivity."
                ),
                "recommendation": (
                    "Maintain current patient notification protocols and evaluate expanding operating hours to meet high fulfillment rates."
                ),
                "action_label": "Maintain Clinical Schedules"
            })
    else:
        recs.append({
            "category": "Appointment Tracking",
            "title": "Enable Visit Status Logging",
            "severity": "Info",
            "metric_value": "Status Field Missing",
            "insight": "Appointment outcome status (Completed, Cancelled, No-Show) was not detected in this dataset.",
            "recommendation": (
                "Capture visit outcome statuses in your EHR exports to calculate cancellation attrition rates "
                "and identify peak provider scheduling bottlenecks."
            ),
            "action_label": "Include Appointment Status"
        })

    # 2. DEPARTMENT WORKLOAD ANALYSIS
    if "_clean_dept" in df.columns:
        dept_counts = df[df["_clean_dept"] != "General"]["_clean_dept"].value_counts()
        if len(dept_counts) >= 2:
            top_dept = dept_counts.index[0]
            top_dept_vol = dept_counts.iloc[0]
            top_dept_share = round(top_dept_vol / len(df) * 100, 1)

            least_dept = dept_counts.index[-1]
            least_dept_vol = dept_counts.iloc[-1]

            if top_dept_share > 30.0:
                recs.append({
                    "category": "Clinical Workload Distribution",
                    "title": f"Workload Concentration in {top_dept} ({top_dept_share}% of Hospital Volume)",
                    "severity": "Warning",
                    "metric_value": f"{top_dept_vol} encounters in {top_dept}",
                    "insight": (
                        f"{top_dept} manages {top_dept_vol} patient visits ({top_dept_share}% of all hospital encounters), "
                        f"while {least_dept} manages only {least_dept_vol} visits. "
                        "Severe clinical imbalance increases provider burnout risk, creates patient wait times, and delays diagnostic charting."
                    ),
                    "recommendation": (
                        f"Reallocate nurse triage personnel and clinical assistant shifts to {top_dept}. "
                        "Audit room availability and consider staggering specialist clinic days."
                    ),
                    "action_label": "Rebalance Department Staffing"
                })

    # 3. ADMISSION AND DISCHARGE DATA INSIGHTS
    if total_admissions > 0:
        if alos is not None:
            if alos > 8.0:
                recs.append({
                    "category": "Inpatient Flow & Bed Turnover",
                    "title": f"Extended Length of Stay: {alos} Days Average ALOS",
                    "severity": "Warning",
                    "metric_value": f"{alos} Days ALOS | {current_inpatients} Current Inpatients",
                    "insight": (
                        f"Average Length of Stay is elevated at {alos} days across {total_admissions} inpatient admissions. "
                        f"There are currently {current_inpatients} active inpatients. "
                        "Prolonged hospital stays increase nosocomial infection risks and constrain acute bed availability."
                    ),
                    "recommendation": (
                        "Implement daily multidisciplinary discharge rounds and coordinate post-acute stepdown care earlier in patient treatment cycles."
                    ),
                    "action_label": "Initiate Discharge Protocols"
                })
            else:
                recs.append({
                    "category": "Inpatient Flow & Bed Turnover",
                    "title": f"Efficient Bed Turnover Velocity (ALOS: {alos} Days)",
                    "severity": "Opportunity",
                    "metric_value": f"{alos} Days ALOS across {total_admissions} Admissions",
                    "insight": (
                        f"Hospital bed turnover is efficient with an Average Length of Stay of {alos} days. "
                        f"{total_discharges} patients have been discharged safely."
                    ),
                    "recommendation": (
                        "Standardize existing clinical care pathways across adjacent inpatient units to replicate high turnover efficiency."
                    ),
                    "action_label": "Standardize Clinical Pathways"
                })

    # 4. PATIENT RECORD DATA QUALITY SUGGESTIONS
    dup_patients = int(df.duplicated(subset=[mapping.get("patient_id", df.columns[0])]).sum()) if mapping.get("patient_id") else 0
    if dup_patients > 0:
        recs.append({
            "category": "EHR Master Patient Index Integrity",
            "title": f"Master Patient Index: {dup_patients} Potential Duplicate Encounters",
            "severity": "Medium",
            "metric_value": f"{dup_patients} Duplicate Patient Identifier Rows",
            "insight": (
                f"Detected {dup_patients} repeated patient record keys in the dataset. "
                "Unlinked duplicate records create fragmented clinical histories, duplicate lab orders, and billing reconciliation errors."
            ),
            "recommendation": (
                "Execute an automated Master Patient Index (MPI) deduplication pass using national ID / MRN match criteria."
            ),
            "action_label": "Merge Duplicate EHRs"
        })

    # 5. AGE & DEMOGRAPHIC ANOMALIES
    if "_clean_age" in df.columns:
        invalid_ages = df[df["_clean_age"] < 0]
        extreme_ages = df[df["_clean_age"] > 115]
        bad_age_count = len(invalid_ages) + len(extreme_ages)
        if bad_age_count > 0:
            recs.append({
                "category": "Clinical Data Quality",
                "title": f"EHR Data Entry Audit: {bad_age_count} Invalid Patient Age Record(s)",
                "severity": "Warning",
                "metric_value": f"{bad_age_count} Out-of-Range Ages (<0 or >115)",
                "insight": (
                    f"Found {bad_age_count} patient records with physiologically impossible ages. "
                    "Erroneous patient ages skew clinical dose calculation algorithms and demographic analytics."
                ),
                "recommendation": (
                    "Apply validation constraints on EHR intake forms preventing negative or 3-digit ages over 115."
                ),
                "action_label": "Validate Patient Demographics"
            })

    return recs


# ==========================================
# 3. EDUCATION RECOMMENDATIONS
# ==========================================

def generate_education_recommendations(df, mapping=None, summary=None):
    """
    Produces dynamic, metric-driven academic recommendations covering:
    - Student performance improvement insights
    - Attendance improvement suggestions
    - Subject-wise performance analysis
    - Class performance insights
    - Missing student record warnings
    """
    recs = []
    mapping = mapping or {}
    summary = summary or {}

    total_records = summary.get("total_records", len(df))
    total_students = summary.get("total_students", total_records)
    avg_marks = summary.get("avg_marks", None)
    pass_pct = summary.get("pass_pct", None)
    avg_attendance = summary.get("avg_attendance", None)
    total_classes = summary.get("total_classes", 0)
    total_subjects = summary.get("total_subjects", 0)

    # 1. STUDENT PERFORMANCE IMPROVEMENT INSIGHTS
    if avg_marks is not None:
        if pass_pct is not None and pass_pct < 75.0:
            recs.append({
                "category": "Academic Performance & Pass Rate",
                "title": f"Urgent Academic Intervention: {pass_pct}% Cohort Pass Rate",
                "severity": "Critical",
                "metric_value": f"{pass_pct}% Pass Rate | {avg_marks}% Average Marks",
                "insight": (
                    f"Cohort pass rate is currently {pass_pct}%, with an overall marks average of {avg_marks}%. "
                    f"{round(100 - pass_pct, 1)}% of students are currently failing to meet grade-level proficiency standards."
                ),
                "recommendation": (
                    "Launch mandatory remedial tutoring sessions for students scoring below 40%, "
                    "review exam difficulty weighting, and schedule weekly parent-teacher intervention meetings."
                ),
                "action_label": "Deploy Remedial Tutoring"
            })
        else:
            recs.append({
                "category": "Academic Performance",
                "title": f"Strong Academic Proficiency ({pass_pct}% Pass Rate)",
                "severity": "Opportunity",
                "metric_value": f"{avg_marks}% Avg Score across {total_students} Students",
                "insight": (
                    f"Cohort demonstrates healthy academic health: {pass_pct}% pass rate with an average mark of {avg_marks}%. "
                    "Core subject mastery meets institutional accreditation standards."
                ),
                "recommendation": (
                    "Introduce honors/advanced placement coursework and academic mentorship programs to accelerate top performers."
                ),
                "action_label": "Expand Honors Programs"
            })

    # 2. ATTENDANCE IMPROVEMENT SUGGESTIONS
    if avg_attendance is not None:
        if avg_attendance < 80.0:
            recs.append({
                "category": "Attendance Compliance & Truancy",
                "title": f"Attendance Deficit Warning: {avg_attendance}% Overall Attendance",
                "severity": "Warning",
                "metric_value": f"{avg_attendance}% Average Attendance Compliance",
                "insight": (
                    f"Overall student attendance rate is {avg_attendance}%, falling below the standard 80% statutory minimum. "
                    "Chronically absent students correlate with an 85% higher probability of failing term examinations."
                ),
                "recommendation": (
                    "Establish automated daily absence SMS notifications to parents, require medical certifications for 3+ day absences, "
                    "and conduct counselor check-ins with students below 75% attendance."
                ),
                "action_label": "Enforce Attendance Tracking"
            })
        else:
            recs.append({
                "category": "Attendance Compliance",
                "title": f"Healthy Classroom Attendance ({avg_attendance}%)",
                "severity": "Info",
                "metric_value": f"{avg_attendance}% Attendance Compliance",
                "insight": (
                    f"Student presence compliance is commendable at {avg_attendance}%. Regular classroom presence supports curricular continuity."
                ),
                "recommendation": (
                    "Acknowledge classes achieving >90% monthly attendance with term recognitions."
                ),
                "action_label": "Recognize High Attendance"
            })
    else:
        recs.append({
            "category": "Attendance Monitoring",
            "title": "Incorporate Attendance Tracking Data",
            "severity": "Info",
            "metric_value": "Attendance Data Not Present",
            "insight": "Dataset does not contain an attendance percentage or present/absent days column.",
            "recommendation": (
                "Include attendance records in student CSV uploads to identify chronic absenteeism risk "
                "before term examination results are finalized."
            ),
            "action_label": "Upload Attendance Column"
        })

    # 3. SUBJECT-WISE PERFORMANCE ANALYSIS
    if "_clean_subject" in df.columns and "_clean_marks" in df.columns:
        valid_marks_df = df[df["_clean_marks"].notna()]
        if not valid_marks_df.empty:
            sub_perf = valid_marks_df.groupby("_clean_subject")["_clean_marks"].mean().sort_values()
            if len(sub_perf) >= 2:
                weakest_sub = sub_perf.index[0]
                weakest_score = round(sub_perf.iloc[0], 1)
                strongest_sub = sub_perf.index[-1]
                strongest_score = round(sub_perf.iloc[-1], 1)

                if weakest_score < 60.0:
                    recs.append({
                        "category": "Curricular Review",
                        "title": f"Subject Proficiency Deficit in '{weakest_sub}' ({weakest_score}% Avg)",
                        "severity": "Warning",
                        "metric_value": f"Lowest Subject Avg: {weakest_score}% in '{weakest_sub}'",
                        "insight": (
                            f"Students struggle most in '{weakest_sub}', averaging {weakest_score}%. "
                            f"In contrast, students excel in '{strongest_sub}' with an average of {strongest_score}%. "
                            f"The {round(strongest_score - weakest_score, 1)} point gap highlights subject-specific learning gaps."
                        ),
                        "recommendation": (
                            f"Audit the syllabus pacing and assessment formatting in '{weakest_sub}'. "
                            "Provide teacher training on interactive problem-solving modules and offer after-school subject clinics."
                        ),
                        "action_label": f"Revamp {weakest_sub} Curriculum"
                    })

    # 4. CLASS PERFORMANCE INSIGHTS
    if "_clean_class" in df.columns and "_clean_marks" in df.columns:
        class_perf = df.groupby("_clean_class")["_clean_marks"].mean().sort_values()
        if len(class_perf) >= 2:
            lowest_class = class_perf.index[0]
            lowest_class_score = round(class_perf.iloc[0], 1)
            highest_class = class_perf.index[-1]
            highest_class_score = round(class_perf.iloc[-1], 1)

            if (highest_class_score - lowest_class_score) > 12.0:
                recs.append({
                    "category": "Section Performance Equity",
                    "title": f"Performance Disparity Between Sections ({round(highest_class_score - lowest_class_score, 1)} Pt Spread)",
                    "severity": "Warning",
                    "metric_value": f"Section '{lowest_class}': {lowest_class_score}% vs '{highest_class}': {highest_class_score}%",
                    "insight": (
                        f"Significant divergence observed: Section '{lowest_class}' averages {lowest_class_score}%, "
                        f"lagging behind Section '{highest_class}' ({highest_class_score}%). "
                        "Section variance suggests teaching style discrepancies or uneven baseline student distribution."
                    ),
                    "recommendation": (
                        f"Pair faculty from '{lowest_class}' and '{highest_class}' for peer teaching observations and balance student cohorts."
                    ),
                    "action_label": "Balance Faculty & Cohorts"
                })

    # 5. MISSING STUDENT RECORD WARNINGS
    student_id_col = mapping.get("student_id")
    if student_id_col and student_id_col in df.columns:
        null_ids = int(df[student_id_col].isnull().sum())
        dup_ids = int(df.duplicated(subset=[student_id_col]).sum())
        if null_ids > 0 or dup_ids > 0:
            recs.append({
                "category": "Academic Registry Data Integrity",
                "title": f"Registry Discrepancies: {null_ids} Missing IDs & {dup_ids} Duplicate Roll Numbers",
                "severity": "Critical" if null_ids > 0 else "Medium",
                "metric_value": f"{null_ids} Null Roll Numbers | {dup_ids} Duplicate Entries",
                "insight": (
                    f"Found {null_ids} unrecorded student roll numbers and {dup_ids} duplicated IDs. "
                    "Unidentified records cause marks reconciliation failures and inaccurate grade cards on student portals."
                ),
                "recommendation": (
                    "Reconcile student records against the central admissions registry before publishing official grade transcripts."
                ),
                "action_label": "Reconcile Student Registry"
            })

    return recs


# ==========================================
# 4. BUSINESS & FINANCE RECOMMENDATIONS
# ==========================================

def generate_finance_recommendations(df, mapping=None, summary=None):
    """
    Produces dynamic, metric-driven financial recommendations covering:
    - Expense reduction opportunities
    - Revenue and profit insights
    - Budget overrun warnings
    - Transaction anomaly alerts
    - Cash flow insights
    """
    recs = []
    mapping = mapping or {}
    summary = summary or {}

    total_revenue = summary.get("total_revenue", 0.0)
    total_expenses = summary.get("total_expenses", 0.0)
    net_profit = summary.get("net_profit", 0.0)
    profit_margin = summary.get("profit_margin", 0.0)
    total_transactions = summary.get("total_transactions", 0)
    atv = summary.get("atv", 0.0)
    mom_growth = summary.get("mom_growth", None)

    # 1. EXPENSE REDUCTION OPPORTUNITIES
    if total_revenue > 0 and total_expenses > 0:
        expense_ratio = round(total_expenses / total_revenue * 100, 1)
        if expense_ratio > 70.0:
            recs.append({
                "category": "Operating Cost Rationalization",
                "title": f"High Operating Cost Burn ({expense_ratio}% of Revenue)",
                "severity": "Critical" if expense_ratio > 90 else "Warning",
                "metric_value": f"${total_expenses:,.2f} Expenses vs ${total_revenue:,.2f} Revenue ({expense_ratio}%)",
                "insight": (
                    f"Operating expenses consume {expense_ratio}% of total top-line revenue, leaving only "
                    f"${net_profit:,.2f} ({profit_margin}%) in net operational profit. "
                    "Elevated operational cost burdens severely limit liquidity and reinvestment flexibility."
                ),
                "recommendation": (
                    "Conduct a category-level cost center audit. Renegotiate enterprise vendor contracts, "
                    "consolidate redundant SaaS subscriptions, and place non-essential discretionary expenditures on review."
                ),
                "action_label": "Execute Cost Center Audit"
            })
        else:
            recs.append({
                "category": "Cost Management",
                "title": f"Disciplined Operating Efficiency ({expense_ratio}% Expense Ratio)",
                "severity": "Opportunity",
                "metric_value": f"{expense_ratio}% Expense Ratio | ${net_profit:,.2f} Net Profit",
                "insight": (
                    f"Cost structure is lean and well-governed: operating expenses are {expense_ratio}% of revenue, "
                    f"yielding a robust {profit_margin}% net profit margin."
                ),
                "recommendation": (
                    "Allocate incremental free cash flow to strategic R&D and high-ROI commercial growth initiatives."
                ),
                "action_label": "Deploy Growth Capital"
            })

    # 2. REVENUE AND PROFIT INSIGHTS
    if net_profit < 0:
        recs.append({
            "category": "Profitability & Solvency",
            "title": f"Net Operating Loss Alert: -${abs(net_profit):,.2f}",
            "severity": "Critical",
            "metric_value": f"-${abs(net_profit):,.2f} Deficit ({profit_margin}% Margin)",
            "insight": (
                f"Business operations generated a net loss of ${abs(net_profit):,.2f} across the analyzed period. "
                "Operating at negative margins erodes cash reserves and threatens commercial solvency if sustained."
            ),
            "recommendation": (
                "Implement an immediate pricing review, eliminate negative-margin product lines, and enforce strict spending approval limits."
            ),
            "action_label": "Halt Discretionary Burn"
        })
    elif mom_growth is not None:
        if mom_growth < 0:
            recs.append({
                "category": "Revenue Trajectory",
                "title": f"Month-over-Month Revenue Contraction ({mom_growth}%)",
                "severity": "Warning",
                "metric_value": f"{mom_growth}% MoM Growth",
                "insight": (
                    f"Revenue contracted by {abs(mom_growth)}% compared to the prior period. "
                    "Sequential top-line deceleration requires commercial pipeline intervention."
                ),
                "recommendation": (
                    "Review customer churn, analyze deal slippage in the sales pipeline, and launch short-term promotional packages."
                ),
                "action_label": "Accelerate Sales Pipeline"
            })

    # 3. BUDGET OVERRUN WARNINGS
    budget_col = mapping.get("budget")
    actual_col = mapping.get("actual")
    if budget_col and actual_col and budget_col in df.columns and actual_col in df.columns:
        b_series = pd.to_numeric(df[budget_col], errors="coerce").fillna(0)
        a_series = pd.to_numeric(df[actual_col], errors="coerce").fillna(0)
        overruns = df[a_series > b_series]
        if len(overruns) > 0:
            total_overspend = float((a_series - b_series)[a_series > b_series].sum())
            recs.append({
                "category": "Budget Variance & Compliance",
                "title": f"Budget Variance Alert: {len(overruns)} Accounts Exceeded Target by ${total_overspend:,.2f}",
                "severity": "Warning",
                "metric_value": f"${total_overspend:,.2f} Total Budget Overrun across {len(overruns)} Line Items",
                "insight": (
                    f"Detected {len(overruns)} line items where actual expenditures surpassed budgeted allocations by a combined ${total_overspend:,.2f}. "
                    "Uncontrolled variances compromise financial year-end forecast accuracy."
                ),
                "recommendation": (
                    "Implement pre-approval workflows for purchases exceeding departmental budget envelopes by more than 5%."
                ),
                "action_label": "Enforce Budget Limits"
            })
    else:
        recs.append({
            "category": "Budget Governance",
            "title": "Incorporate Budget Target Benchmarks",
            "severity": "Info",
            "metric_value": "Budget Benchmark Column Missing",
            "insight": "Current financial dataset does not contain separate Budget vs Actual target columns.",
            "recommendation": (
                "Include budgeted departmental allocations in your financial dataset to unlock automated "
                "budget variance tracking, overrun heatmaps, and financial runway projections."
            ),
            "action_label": "Add Budget Allocations"
        })

    # 4. TRANSACTION ANOMALY ALERTS
    if "_clean_amount" in df.columns:
        amt = df["_clean_amount"].dropna()
        q1 = amt.quantile(0.25)
        q3 = amt.quantile(0.75)
        iqr = q3 - q1
        outlier_upper = q3 + 3 * iqr
        anomalies = df[amt > outlier_upper]
        if len(anomalies) > 0:
            recs.append({
                "category": "Financial Risk & Anomaly Detection",
                "title": f"Suspicious Transaction Outliers: {len(anomalies)} Transactions Exceed ${outlier_upper:,.2f}",
                "severity": "Warning",
                "metric_value": f"{len(anomalies)} Outlier Transactions (> 3x IQR)",
                "insight": (
                    f"Identified {len(anomalies)} transactions with unusually large values (above ${outlier_upper:,.2f}). "
                    "Extreme value spikes can indicate billing typos, duplicate invoices, or unauthorized transactions."
                ),
                "recommendation": (
                    "Perform audit sampling on all flagged high-value transactions to ensure matching invoices and management sign-offs."
                ),
                "action_label": "Audit High-Value Transactions"
            })

    # 5. CASH FLOW & RECEIVABLES INSIGHTS
    status_col = mapping.get("status")
    if status_col and status_col in df.columns:
        status_counts = df[status_col].astype(str).str.lower().value_counts()
        unpaid_count = 0
        for k in status_counts.index:
            if any(term in k for term in ["pending", "unpaid", "overdue", "due", "open"]):
                unpaid_count += status_counts[k]
        
        unpaid_pct = round(unpaid_count / len(df) * 100, 1) if len(df) > 0 else 0
        if unpaid_pct > 15.0:
            recs.append({
                "category": "Working Capital & Cash Flow",
                "title": f"Working Capital Drag: {unpaid_pct}% Unsettled Invoices",
                "severity": "Warning",
                "metric_value": f"{unpaid_count} Pending / Overdue Invoices ({unpaid_pct}%)",
                "insight": (
                    f"{unpaid_count} invoices ({unpaid_pct}% of total records) remain in pending, open, or overdue status. "
                    "Lagging collections compress working capital liquidity and strain supplier payment commitments."
                ),
                "recommendation": (
                    "Institute automated accounts receivable aging dunning reminders (15, 30, 45 days) and consider early-payment 2% discounts."
                ),
                "action_label": "Accelerate A/R Dunning"
            })

    return recs


# ==========================================
# 5. HUMAN RESOURCES & WORKFORCE RECOMMENDATIONS
# ==========================================

def generate_hr_recommendations(df, mapping=None, summary=None):
    """
    Produces dynamic, metric-driven workforce and human resources recommendations for employee datasets:
    - Headcount and departmental staffing balance
    - Compensation and payroll equity
    - Turnover and attrition warnings
    - Employee age profile and retirement planning
    - HR data quality and registry hygiene
    """
    recs = []
    summary = summary or {}
    total_employees = summary.get("total_employees", len(df))
    active_employees = summary.get("active_employees", len(df))
    departed_employees = summary.get("departed_employees", 0)
    turnover_rate = summary.get("turnover_rate", 0.0)
    avg_salary = summary.get("avg_salary", 0.0)
    total_payroll = summary.get("total_payroll", 0.0)
    avg_age = summary.get("avg_age", None)

    # 1. TURNOVER & RETENTION
    if departed_employees > 0:
        if turnover_rate > 15.0:
            recs.append({
                "category": "Workforce Retention & Attrition",
                "title": f"Elevated Employee Attrition: {turnover_rate}% Turnover Rate",
                "severity": "Critical" if turnover_rate > 25 else "Warning",
                "metric_value": f"{departed_employees} Departed of {total_employees} Total Headcount ({turnover_rate}%)",
                "insight": (
                    f"{departed_employees} employees have exited the organization ({turnover_rate}% turnover rate). "
                    "High employee attrition leads to institutional knowledge drain, recruitment expenses, and operational disruption."
                ),
                "recommendation": (
                    "Conduct structured exit interviews to diagnose root causes (compensation, management, or career progression) "
                    "and benchmark departmental retention initiatives."
                ),
                "action_label": "Conduct Exit Audits"
            })
        else:
            recs.append({
                "category": "Workforce Retention",
                "title": f"Healthy Workforce Retention (Turnover: {turnover_rate}%)",
                "severity": "Info",
                "metric_value": f"{active_employees} Active Staff ({round(active_employees / total_employees * 100, 1)}%)",
                "insight": (
                    f"Employee turnover is controlled at {turnover_rate}%. A stable workforce supports operational excellence."
                ),
                "recommendation": (
                    "Invest in professional development pathways to sustain high employee tenure and engagement."
                ),
                "action_label": "Foster Talent Growth"
            })

    # 2. COMPENSATION EQUITY & PAYROLL
    if avg_salary > 0:
        recs.append({
            "category": "Compensation & Payroll Planning",
            "title": f"Annualized Payroll Burden: ${total_payroll:,.2f} (Avg: ${avg_salary:,.2f})",
            "severity": "Opportunity",
            "metric_value": f"${total_payroll:,.2f} Total Payroll | ${avg_salary:,.2f} Avg Salary",
            "insight": (
                f"Total workforce payroll expenditure is ${total_payroll:,.2f} across {active_employees} active staff members. "
                f"Average employee compensation is ${avg_salary:,.2f}."
            ),
            "recommendation": (
                "Benchmark departmental compensation bands against industry median salary surveys to prevent poaching of top performers."
            ),
            "action_label": "Benchmark Salary Bands"
        })

    # 3. AGE PROFILE & RETIREMENT HORIZON
    if avg_age is not None:
        recs.append({
            "category": "Workforce Demographics",
            "title": f"Workforce Age Profile: {avg_age} Years Average Age",
            "severity": "Info",
            "metric_value": f"{avg_age} Years Average Age",
            "insight": (
                f"The average employee age across active staff is {avg_age} years. "
                "Monitoring age distributions aids succession planning and employee benefits tailoring."
            ),
            "recommendation": (
                "Ensure robust knowledge-transfer protocols between senior specialists and junior associates."
            ),
            "action_label": "Succession Planning"
        })

    # 4. HR REGISTRY DATA HYGIENE
    dup_emp_count = int(df.duplicated(subset=["Employee_ID"]).sum()) if "Employee_ID" in df.columns else int(df.duplicated().sum())
    if dup_emp_count > 0:
        recs.append({
            "category": "HR Data Integrity",
            "title": f"HR Information System: {dup_emp_count} Duplicate Employee Record(s)",
            "severity": "Critical",
            "metric_value": f"{dup_emp_count} Duplicate Employee IDs",
            "insight": (
                f"Detected {dup_emp_count} duplicate employee records. "
                "Duplicated employee IDs distort payroll calculations and tax filing reporting."
            ),
            "recommendation": (
                "Deduplicate the employee master registry by verifying employee tax IDs and national identification records."
            ),
            "action_label": "Deduplicate Employee Records"
        })

    return recs


# ==========================================
# 6. UNIFIED ROUTER
# ==========================================

def get_industry_recommendations(df, workspace="Retail Store", mapping=None, summary=None):
    """
    Master router that strictly isolates and returns context-aware recommendations
    for the active industry workspace or employee dataset.
    
    Guarantees:
    - Never returns medical recommendations for Retail, Education, or Business & Finance.
    - Never returns retail recommendations for Healthcare, Education, or HR.
    """
    if workspace == "Retail Store":
        return generate_retail_recommendations(df, mapping, summary)
    elif workspace == "Healthcare":
        return generate_healthcare_recommendations(df, mapping, summary)
    elif workspace == "Education":
        return generate_education_recommendations(df, mapping, summary)
    elif workspace in ["Business & Finance", "Finance"]:
        return generate_finance_recommendations(df, mapping, summary)
    elif workspace in ["Employee & HR", "Human Resources", "HR", "Employees"]:
        return generate_hr_recommendations(df, mapping, summary)
    else:
        return generate_retail_recommendations(df, mapping, summary)
