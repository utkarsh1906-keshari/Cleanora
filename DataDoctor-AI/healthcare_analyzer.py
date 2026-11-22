"""
DataDoctor AI - Healthcare Workspace Analyzer Module
Comprehensive clinical and operational analytics for patient records,
appointments, admissions, discharges, doctors, departments, diagnoses,
treatments, and healthcare data quality verification.
"""

import pandas as pd
import numpy as np


def map_healthcare_columns(df):
    """
    Intelligently maps raw DataFrame columns to standard healthcare keys
    using case-insensitive keyword and alias matching.
    """
    column_aliases = {
        "patient_id": [
            "patient_id", "patientid", "pid", "mrn", "patient_no",
            "patient_number", "pat_id", "subject_id", "member_id"
        ],
        "patient_name": [
            "patient_name", "patientname", "pat_name", "patient", "client_name"
        ],
        "age": [
            "age", "patient_age", "patientage", "years", "pat_age"
        ],
        "gender": [
            "gender", "sex", "patient_gender", "patient_sex"
        ],
        "department": [
            "department", "speciality", "specialty", "ward", "clinic",
            "medical_specialty", "dept", "service"
        ],
        "doctor": [
            "doctor_name", "doctor", "physician", "provider", "physician_name",
            "dr_name", "consultant", "practitioner", "care_provider"
        ],
        "appointment_date": [
            "appointment_date", "visit_date", "appt_date", "consultation_date",
            "encounter_date", "checkup_date", "booking_date"
        ],
        "appointment_status": [
            "appointment_status", "appt_status", "visit_status",
            "outcome_status", "booking_status"
        ],
        "diagnosis": [
            "diagnosis", "condition", "disease", "primary_diagnosis",
            "icd10", "illness", "health_condition", "clinical_diagnosis"
        ],
        "treatment": [
            "treatment", "procedure", "medication", "therapy",
            "intervention", "treatment_plan", "prescription", "care_plan"
        ],
        "admission_date": [
            "admission_date", "admitted_date", "hospital_admission", "inpatient_date", "admit_date"
        ],
        "discharge_date": [
            "discharge_date", "discharged_date", "hospital_discharge", "outpatient_date", "disch_date"
        ],
        "billing": [
            "billing_amount", "billing", "charges", "hospital_bill", "total_bill",
            "medical_charges", "patient_bill", "treatment_cost", "total_charges"
        ]
    }

    mapping = {}
    normalized_cols = {col.lower().strip().replace(" ", "_"): col for col in df.columns}

    # Explicit exclusions for non-clinical columns
    excluded_from_patient_id = ["employee_id", "empid", "emp_id", "staff_id", "student_id", "order_id", "trans_id", "transaction_id"]
    excluded_from_appts = ["joining_date", "leaving_date", "hire_date", "exit_date", "order_date", "invoice_date"]
    excluded_from_billing = ["salary", "wage", "payroll", "cogs", "inventory_cost"]

    for key, aliases in column_aliases.items():
        found_col = None
        # 1. Exact alias match
        for alias in aliases:
            if alias in normalized_cols:
                norm_cand = alias
                orig_cand = normalized_cols[alias]
                if key == "patient_id" and any(exc in norm_cand for exc in excluded_from_patient_id):
                    continue
                if key in ["appointment_date", "admission_date"] and any(exc in norm_cand for exc in excluded_from_appts):
                    continue
                if key == "billing" and any(exc in norm_cand for exc in excluded_from_billing):
                    continue
                found_col = orig_cand
                break

        # 2. Substring fallback match (only if alias is at least 3 characters and not a generic word)
        if not found_col:
            for norm_name, original_name in normalized_cols.items():
                if key == "patient_id" and any(exc in norm_name for exc in excluded_from_patient_id):
                    continue
                if key in ["appointment_date", "admission_date", "discharge_date"] and any(exc in norm_name for exc in excluded_from_appts):
                    continue
                if key == "billing" and any(exc in norm_name for exc in excluded_from_billing):
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
    if "patient_id" not in mapping and "patient_name" not in mapping:
        missing_critical.append("patient_id")
    
    # Check for genuine clinical context
    has_clinical_context = any(k in mapping for k in ["diagnosis", "treatment", "doctor", "admission_date", "discharge_date"]) or ("patient_id" in mapping and "appointment_date" in mapping)
    if not has_clinical_context:
        missing_critical.append("clinical context (e.g. diagnosis, treatment, doctor, or admission date)")

    return {
        "mapping": mapping,
        "detected": detected,
        "missing_critical": missing_critical,
        "optional_detected": [k for k in ["age", "gender", "department", "doctor", "appointment_date", "appointment_status", "diagnosis", "treatment", "admission_date", "discharge_date", "billing"] if k in mapping],
        "optional_missing": [k for k in ["age", "gender", "department", "doctor", "appointment_date", "appointment_status", "diagnosis", "treatment", "admission_date", "discharge_date", "billing"] if k not in mapping]
    }


def prepare_healthcare_data(df, mapping=None):
    """
    Cleans and standardizes healthcare records using detected column mapping.
    Safely computes internal fields for patient IDs, ages, genders, dates,
    admissions, length of stay, and billing without mutating original columns.
    """
    if mapping is None:
        mapping_info = map_healthcare_columns(df)
        mapping = mapping_info["mapping"]

    work_df = df.copy()

    # 1. Patient ID
    if "patient_id" in mapping:
        work_df["_clean_pid"] = work_df[mapping["patient_id"]].astype(str).str.strip().replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    elif "patient_name" in mapping:
        work_df["_clean_pid"] = work_df[mapping["patient_name"]].astype(str).str.strip()
    else:
        work_df["_clean_pid"] = [f"PAT-{i+1:04d}" for i in range(len(work_df))]

    # 2. Patient Name
    if "patient_name" in mapping:
        work_df["_clean_name"] = work_df[mapping["patient_name"]].astype(str).str.strip()
    else:
        work_df["_clean_name"] = work_df["_clean_pid"]

    # 3. Age & Age Brackets
    has_age = False
    if "age" in mapping:
        work_df["_clean_age"] = pd.to_numeric(work_df[mapping["age"]], errors="coerce")
        has_age = True
        
        def categorize_age(val):
            if pd.isna(val) or val < 0 or val > 120:
                return "Invalid/Unknown"
            elif val <= 12:
                return "Pediatric (0-12)"
            elif val <= 18:
                return "Adolescent (13-18)"
            elif val <= 40:
                return "Young Adult (19-40)"
            elif val <= 64:
                return "Middle Adult (41-64)"
            else:
                return "Senior (65+)"
                
        work_df["_age_group"] = work_df["_clean_age"].apply(categorize_age)
    else:
        work_df["_clean_age"] = np.nan
        work_df["_age_group"] = "Unspecified"

    # 4. Gender
    has_gender = False
    if "gender" in mapping:
        has_gender = True
        def normalize_gender(val):
            if pd.isna(val) or str(val).strip() in ["", "nan", "None"]:
                return "Unknown"
            s = str(val).strip().capitalize()
            if s.startswith("M"):
                return "Male"
            elif s.startswith("F"):
                return "Female"
            else:
                return "Other"
        work_df["_clean_gender"] = work_df[mapping["gender"]].apply(normalize_gender)
    else:
        work_df["_clean_gender"] = "Unknown"

    # 5. Department
    has_dept = False
    if "department" in mapping:
        has_dept = True
        work_df["_clean_dept"] = work_df[mapping["department"]].astype(str).str.strip().replace({"nan": "General", "None": "General", "": "General"})
    else:
        work_df["_clean_dept"] = "General"

    # 6. Doctor
    has_doctor = False
    if "doctor" in mapping:
        has_doctor = True
        work_df["_clean_doctor"] = work_df[mapping["doctor"]].astype(str).str.strip().replace({"nan": "Unassigned", "None": "Unassigned", "": "Unassigned"})
    else:
        work_df["_clean_doctor"] = "Unassigned"

    # 7. Appointment Date
    has_appt_date = False
    if "appointment_date" in mapping:
        work_df["_clean_appt_date"] = pd.to_datetime(work_df[mapping["appointment_date"]], errors="coerce")
        valid_appts = work_df["_clean_appt_date"].dropna()
        if len(valid_appts) > 0:
            has_appt_date = True
            work_df["_appt_year_month"] = work_df["_clean_appt_date"].dt.strftime("%Y-%m")
            work_df["_appt_month_label"] = work_df["_clean_appt_date"].dt.strftime("%b %Y")
            work_df["_appt_date_only"] = work_df["_clean_appt_date"].dt.date
            work_df["_appt_day_name"] = work_df["_clean_appt_date"].dt.day_name()

    # 8. Appointment Status
    has_appt_status = False
    if "appointment_status" in mapping:
        has_appt_status = True
        def normalize_status(val):
            if pd.isna(val) or str(val).strip() in ["", "nan", "None"]:
                return "Unknown"
            s = str(val).strip().capitalize()
            if "comp" in s.lower():
                return "Completed"
            elif "canc" in s.lower():
                return "Cancelled"
            elif "pend" in s.lower() or "sched" in s.lower():
                return "Pending"
            elif "no" in s.lower() and "show" in s.lower():
                return "No-Show"
            return s
        work_df["_clean_appt_status"] = work_df[mapping["appointment_status"]].apply(normalize_status)
    else:
        work_df["_clean_appt_status"] = "Completed"

    # 9. Diagnosis
    has_diagnosis = False
    if "diagnosis" in mapping:
        has_diagnosis = True
        work_df["_clean_diagnosis"] = work_df[mapping["diagnosis"]].astype(str).str.strip().replace({"nan": "Undiagnosed", "None": "Undiagnosed", "": "Undiagnosed"})
    else:
        work_df["_clean_diagnosis"] = "General Consultation"

    # 10. Treatment
    has_treatment = False
    if "treatment" in mapping:
        has_treatment = True
        work_df["_clean_treatment"] = work_df[mapping["treatment"]].astype(str).str.strip().replace({"nan": "Routine Care", "None": "Routine Care", "": "Routine Care"})
    else:
        work_df["_clean_treatment"] = "Standard Care"

    # 11. Admission and Discharge Dates & Length of Stay (LOS)
    has_admission = False
    has_discharge = False
    if "admission_date" in mapping:
        work_df["_clean_adm_date"] = pd.to_datetime(work_df[mapping["admission_date"]], errors="coerce")
        if work_df["_clean_adm_date"].notna().sum() > 0:
            has_admission = True
            work_df["_adm_year_month"] = work_df["_clean_adm_date"].dt.strftime("%Y-%m")
            work_df["_adm_date_only"] = work_df["_clean_adm_date"].dt.date
    else:
        work_df["_clean_adm_date"] = pd.NaT

    if "discharge_date" in mapping:
        work_df["_clean_dis_date"] = pd.to_datetime(work_df[mapping["discharge_date"]], errors="coerce")
        if work_df["_clean_dis_date"].notna().sum() > 0:
            has_discharge = True
            work_df["_dis_year_month"] = work_df["_clean_dis_date"].dt.strftime("%Y-%m")
            work_df["_dis_date_only"] = work_df["_clean_dis_date"].dt.date
    else:
        work_df["_clean_dis_date"] = pd.NaT

    # Computed admission indicators
    work_df["_is_admitted"] = work_df["_clean_adm_date"].notna()
    work_df["_is_discharged"] = work_df["_clean_dis_date"].notna()
    work_df["_is_inpatient"] = work_df["_is_admitted"] & (~work_df["_is_discharged"])

    # Length of Stay (LOS) calculation
    valid_los_mask = work_df["_clean_adm_date"].notna() & work_df["_clean_dis_date"].notna()
    work_df["_los_days"] = np.nan
    work_df["_date_inconsistent"] = False
    
    if valid_los_mask.sum() > 0:
        los_series = (work_df.loc[valid_los_mask, "_clean_dis_date"] - work_df.loc[valid_los_mask, "_clean_adm_date"]).dt.days
        work_df.loc[valid_los_mask, "_los_days"] = los_series
        # Inconsistent dates: discharge before admission
        work_df.loc[valid_los_mask & (los_series < 0), "_date_inconsistent"] = True

    # 12. Billing
    has_billing = False
    if "billing" in mapping:
        work_df["_clean_billing"] = pd.to_numeric(work_df[mapping["billing"]], errors="coerce").fillna(0.0)
        has_billing = True
    else:
        work_df["_clean_billing"] = 0.0

    metadata = {
        "has_patient_id": "patient_id" in mapping or "patient_name" in mapping,
        "has_age": has_age,
        "has_gender": has_gender,
        "has_dept": has_dept,
        "has_doctor": has_doctor,
        "has_appt_date": has_appt_date,
        "has_appt_status": has_appt_status,
        "has_diagnosis": has_diagnosis,
        "has_treatment": has_treatment,
        "has_admission": has_admission,
        "has_discharge": has_discharge,
        "has_billing": has_billing
    }

    return work_df, mapping, metadata


def summarize_healthcare(df):
    """
    Computes overarching executive healthcare KPIs.
    """
    total_records = len(df)
    unique_patients = df["_clean_pid"].nunique() if "_clean_pid" in df.columns else total_records
    
    # Total appointments (records with appointment date or all records)
    total_appointments = total_records
    if "_clean_appt_date" in df.columns and df["_clean_appt_date"].notna().sum() > 0:
        total_appointments = int(df["_clean_appt_date"].notna().sum())

    # Admissions & Discharges
    total_admissions = int(df["_is_admitted"].sum()) if "_is_admitted" in df.columns else 0
    total_discharges = int(df["_is_discharged"].sum()) if "_is_discharged" in df.columns else 0
    current_inpatients = int(df["_is_inpatient"].sum()) if "_is_inpatient" in df.columns else 0

    # Doctors & Departments
    valid_doctors = df[df["_clean_doctor"] != "Unassigned"]["_clean_doctor"] if "_clean_doctor" in df.columns else pd.Series([], dtype=str)
    total_doctors = int(valid_doctors.nunique()) if len(valid_doctors) > 0 else 0
    
    valid_depts = df[df["_clean_dept"] != "General"]["_clean_dept"] if "_clean_dept" in df.columns else pd.Series([], dtype=str)
    total_depts = int(valid_depts.nunique()) if len(valid_depts) > 0 else (1 if "_clean_dept" in df.columns else 0)

    # Average Patient Age (excluding invalid ages)
    valid_ages = df[df["_clean_age"].between(0, 120)]["_clean_age"] if "_clean_age" in df.columns else pd.Series([], dtype=float)
    avg_age = round(float(valid_ages.mean()), 1) if len(valid_ages) > 0 else None

    # Average Length of Stay (ALOS)
    valid_los = df[df["_los_days"] >= 0]["_los_days"] if "_los_days" in df.columns else pd.Series([], dtype=float)
    alos = round(float(valid_los.mean()), 1) if len(valid_los) > 0 else None

    # Billing
    total_billing = float(df["_clean_billing"].sum()) if "_clean_billing" in df.columns else 0.0
    avg_billing = round(total_billing / max(1, total_records), 2)

    return {
        "total_records": total_records,
        "unique_patients": unique_patients,
        "total_appointments": total_appointments,
        "total_admissions": total_admissions,
        "total_discharges": total_discharges,
        "current_inpatients": current_inpatients,
        "total_doctors": total_doctors,
        "total_departments": total_depts,
        "avg_age": avg_age,
        "alos": alos,
        "total_billing": total_billing,
        "avg_billing": avg_billing
    }


def get_patient_analytics(df):
    """
    Computes age distribution, gender breakdown, registration trends, and new vs repeat patients.
    """
    res = {}

    # 1. Age Distribution
    valid_ages = df[df["_clean_age"].between(0, 120)] if "_clean_age" in df.columns else pd.DataFrame()
    res["valid_ages_df"] = valid_ages
    
    if "_age_group" in df.columns:
        age_group_counts = df["_age_group"].value_counts().reset_index()
        age_group_counts.columns = ["Age Group", "Patient Count"]
        res["age_groups"] = age_group_counts
    else:
        res["age_groups"] = pd.DataFrame()

    # 2. Gender Breakdown
    if "_clean_gender" in df.columns:
        gender_counts = df["_clean_gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]
        gender_counts["Percentage"] = (gender_counts["Count"] / len(df) * 100).round(1)
        res["gender_distribution"] = gender_counts
    else:
        res["gender_distribution"] = pd.DataFrame()

    # 3. New vs Repeat Patients
    if "_clean_pid" in df.columns:
        pid_counts = df["_clean_pid"].value_counts()
        single_visit = int((pid_counts == 1).sum())
        repeat_visit = int((pid_counts > 1).sum())
        res["patient_type"] = pd.DataFrame({
            "Patient Type": ["New Patients (1 Visit)", "Repeat Patients (2+ Visits)"],
            "Count": [single_visit, repeat_visit],
            "Percentage": [round(single_visit / max(1, len(pid_counts)) * 100, 1),
                           round(repeat_visit / max(1, len(pid_counts)) * 100, 1)]
        })
        res["repeat_rate"] = round(repeat_visit / max(1, len(pid_counts)) * 100, 1)
    else:
        res["patient_type"] = pd.DataFrame()
        res["repeat_rate"] = 0.0

    return res


def get_appointment_analytics(df, freq="daily"):
    """
    Calculates completed/cancelled/pending appointments, status rates,
    temporal trends, and doctor/department distribution.
    """
    res = {}

    # 1. Status Breakdown
    if "_clean_appt_status" in df.columns:
        status_counts = df["_clean_appt_status"].value_counts().reset_index()
        status_counts.columns = ["Status", "Count"]
        status_counts["Percentage"] = (status_counts["Count"] / len(df) * 100).round(1)
        res["status_summary"] = status_counts
        
        # Specific KPI metrics
        status_dict = dict(zip(status_counts["Status"], status_counts["Count"]))
        res["completed"] = status_dict.get("Completed", 0)
        res["cancelled"] = status_dict.get("Cancelled", 0)
        res["pending"] = status_dict.get("Pending", 0)
        res["no_show"] = status_dict.get("No-Show", 0)
        res["completion_rate"] = round(res["completed"] / max(1, len(df)) * 100, 1)
        res["cancellation_rate"] = round(res["cancelled"] / max(1, len(df)) * 100, 1)
    else:
        res["status_summary"] = pd.DataFrame()
        res["completed"] = len(df)
        res["cancelled"] = 0
        res["pending"] = 0
        res["no_show"] = 0
        res["completion_rate"] = 100.0
        res["cancellation_rate"] = 0.0

    # 2. Appointment Trends over time
    if "_clean_appt_date" in df.columns and df["_clean_appt_date"].notna().sum() > 0:
        valid_df = df.dropna(subset=["_clean_appt_date"]).copy()
        if freq == "monthly" and "_appt_year_month" in valid_df.columns:
            trend = valid_df.groupby("_appt_year_month").agg(
                Total_Appointments=("_clean_pid", "count"),
                Completed=("_clean_appt_status", lambda s: (s == "Completed").sum()),
                Cancelled=("_clean_appt_status", lambda s: (s == "Cancelled").sum())
            ).reset_index().rename(columns={"_appt_year_month": "Period"})
        else:
            trend = valid_df.groupby("_appt_date_only").agg(
                Total_Appointments=("_clean_pid", "count"),
                Completed=("_clean_appt_status", lambda s: (s == "Completed").sum()),
                Cancelled=("_clean_appt_status", lambda s: (s == "Cancelled").sum())
            ).reset_index().rename(columns={"_appt_date_only": "Period"})
            trend["Period"] = pd.to_datetime(trend["Period"])
            trend = trend.sort_values("Period")
        res["trend_df"] = trend
    else:
        res["trend_df"] = pd.DataFrame()

    # 3. Doctor-wise appointments
    if "_clean_doctor" in df.columns:
        doc_df = df[df["_clean_doctor"] != "Unassigned"].groupby("_clean_doctor").agg(
            Appointments=("_clean_pid", "count"),
            Completed=("_clean_appt_status", lambda s: (s == "Completed").sum()),
            Cancelled=("_clean_appt_status", lambda s: (s == "Cancelled").sum())
        ).reset_index().rename(columns={"_clean_doctor": "Doctor"}).sort_values("Appointments", ascending=False)
        res["doctor_appointments"] = doc_df
    else:
        res["doctor_appointments"] = pd.DataFrame()

    # 4. Department-wise appointments
    if "_clean_dept" in df.columns:
        dept_df = df.groupby("_clean_dept").agg(
            Appointments=("_clean_pid", "count"),
            Completed=("_clean_appt_status", lambda s: (s == "Completed").sum()),
            Cancelled=("_clean_appt_status", lambda s: (s == "Cancelled").sum())
        ).reset_index().rename(columns={"_clean_dept": "Department"}).sort_values("Appointments", ascending=False)
        res["department_appointments"] = dept_df
    else:
        res["department_appointments"] = pd.DataFrame()

    return res


def get_doctor_department_analytics(df):
    """
    Computes doctor patient workloads and detailed department comparisons.
    """
    res = {}

    # Department Analytics
    if "_clean_dept" in df.columns:
        dept_summary = df.groupby("_clean_dept").agg(
            Patients=("_clean_pid", "nunique"),
            Encounters=("_clean_pid", "count"),
            Admissions=("_is_admitted", "sum"),
            Avg_Age=("_clean_age", lambda a: round(a.dropna().mean(), 1) if len(a.dropna()) > 0 else np.nan),
            Avg_LOS=("_los_days", lambda l: round(l[l >= 0].mean(), 1) if len(l[l >= 0]) > 0 else np.nan),
            Total_Billing=("_clean_billing", "sum")
        ).reset_index().rename(columns={"_clean_dept": "Department"})
        dept_summary["Avg_Billing_Per_Patient"] = (dept_summary["Total_Billing"] / dept_summary["Patients"].clip(lower=1)).round(2)
        dept_summary = dept_summary.sort_values("Patients", ascending=False)
        res["department_summary"] = dept_summary
    else:
        res["department_summary"] = pd.DataFrame()

    # Doctor Analytics
    if "_clean_doctor" in df.columns:
        doc_summary = df[df["_clean_doctor"] != "Unassigned"].groupby(["_clean_doctor", "_clean_dept"]).agg(
            Patients=("_clean_pid", "nunique"),
            Total_Encounters=("_clean_pid", "count"),
            Completed=("_clean_appt_status", lambda s: (s == "Completed").sum()),
            Total_Billing=("_clean_billing", "sum")
        ).reset_index().rename(columns={"_clean_doctor": "Doctor", "_clean_dept": "Department"})
        doc_summary = doc_summary.sort_values("Total_Encounters", ascending=False)
        res["doctor_summary"] = doc_summary
    else:
        res["doctor_summary"] = pd.DataFrame()

    return res


def get_admission_discharge_analytics(df, freq="monthly"):
    """
    Calculates admission and discharge temporal trends, current inpatients,
    and length of stay metrics.
    """
    res = {}

    # Current inpatients
    if "_is_inpatient" in df.columns:
        inpatients = df[df["_is_inpatient"]].copy()
        res["inpatient_count"] = len(inpatients)
        res["inpatients_table"] = inpatients[["_clean_pid", "_clean_name", "_clean_dept", "_clean_doctor", "_clean_adm_date", "_clean_diagnosis"]].rename(
            columns={
                "_clean_pid": "Patient ID",
                "_clean_name": "Patient Name",
                "_clean_dept": "Department",
                "_clean_doctor": "Doctor",
                "_clean_adm_date": "Admission Date",
                "_clean_diagnosis": "Diagnosis"
            }
        ) if len(inpatients) > 0 else pd.DataFrame()
    else:
        res["inpatient_count"] = 0
        res["inpatients_table"] = pd.DataFrame()

    # Admission & Discharge Trends
    adm_valid = df[df["_is_admitted"]].copy() if "_is_admitted" in df.columns else pd.DataFrame()
    dis_valid = df[df["_is_discharged"]].copy() if "_is_discharged" in df.columns else pd.DataFrame()

    if not adm_valid.empty and "_clean_adm_date" in adm_valid.columns:
        if freq == "monthly" and "_adm_year_month" in adm_valid.columns:
            adm_trend = adm_valid.groupby("_adm_year_month")["_clean_pid"].count().reset_index().rename(columns={"_adm_year_month": "Period", "_clean_pid": "Admissions"})
            dis_trend = dis_valid.groupby("_dis_year_month")["_clean_pid"].count().reset_index().rename(columns={"_dis_year_month": "Period", "_clean_pid": "Discharges"}) if not dis_valid.empty else pd.DataFrame(columns=["Period", "Discharges"])
        else:
            adm_trend = adm_valid.groupby("_adm_date_only")["_clean_pid"].count().reset_index().rename(columns={"_adm_date_only": "Period", "_clean_pid": "Admissions"})
            dis_trend = dis_valid.groupby("_dis_date_only")["_clean_pid"].count().reset_index().rename(columns={"_dis_date_only": "Period", "_clean_pid": "Discharges"}) if not dis_valid.empty else pd.DataFrame(columns=["Period", "Discharges"])

        combined_trend = pd.merge(adm_trend, dis_trend, on="Period", how="outer").fillna(0).sort_values("Period")
        res["flow_trend"] = combined_trend
    else:
        res["flow_trend"] = pd.DataFrame()

    # Length of Stay (LOS) Breakdown
    if "_los_days" in df.columns:
        valid_los = df[df["_los_days"] >= 0]["_los_days"]
        if len(valid_los) > 0:
            def categorize_los(days):
                if days <= 2:
                    return "Short Stay (1-2 Days)"
                elif days <= 5:
                    return "Moderate (3-5 Days)"
                elif days <= 10:
                    return "Extended (6-10 Days)"
                else:
                    return "Long Term (11+ Days)"
            
            los_cats = valid_los.apply(categorize_los).value_counts().reset_index()
            los_cats.columns = ["Stay Duration Category", "Patient Count"]
            res["los_categories"] = los_cats
            res["los_values"] = valid_los
            res["avg_los"] = round(float(valid_los.mean()), 1)
            res["median_los"] = float(valid_los.median())
        else:
            res["los_categories"] = pd.DataFrame()
            res["los_values"] = pd.Series([], dtype=float)
            res["avg_los"] = None
            res["median_los"] = None
    else:
        res["los_categories"] = pd.DataFrame()
        res["los_values"] = pd.Series([], dtype=float)
        res["avg_los"] = None
        res["median_los"] = None

    return res


def get_diagnosis_treatment_trends(df):
    """
    Identifies top diagnosed conditions, treatment frequencies,
    department diagnosis distribution, and diagnosis trends over time.
    """
    res = {}

    # 1. Top Diagnoses
    if "_clean_diagnosis" in df.columns:
        diag_counts = df[df["_clean_diagnosis"] != "Undiagnosed"]["_clean_diagnosis"].value_counts().reset_index()
        diag_counts.columns = ["Diagnosis", "Case Count"]
        diag_counts["Percentage"] = (diag_counts["Case Count"] / max(1, len(df)) * 100).round(1)
        res["top_diagnoses"] = diag_counts
    else:
        res["top_diagnoses"] = pd.DataFrame()

    # 2. Top Treatments / Procedures
    if "_clean_treatment" in df.columns:
        treat_counts = df[df["_clean_treatment"] != "Routine Care"]["_clean_treatment"].value_counts().reset_index()
        treat_counts.columns = ["Treatment", "Frequency"]
        treat_counts["Percentage"] = (treat_counts["Frequency"] / max(1, len(df)) * 100).round(1)
        res["top_treatments"] = treat_counts
    else:
        res["top_treatments"] = pd.DataFrame()

    # 3. Diagnosis by Department Heatmap / Cross-tabulation
    if "_clean_diagnosis" in df.columns and "_clean_dept" in df.columns:
        top_10_diag = res["top_diagnoses"]["Diagnosis"].head(10).tolist() if not res["top_diagnoses"].empty else []
        filtered_df = df[df["_clean_diagnosis"].isin(top_10_diag)]
        if not filtered_df.empty:
            dept_diag_matrix = pd.crosstab(filtered_df["_clean_dept"], filtered_df["_clean_diagnosis"])
            res["dept_diag_matrix"] = dept_diag_matrix
        else:
            res["dept_diag_matrix"] = pd.DataFrame()
    else:
        res["dept_diag_matrix"] = pd.DataFrame()

    # 4. Diagnosis Trends Over Time
    if "_clean_diagnosis" in df.columns and "_appt_year_month" in df.columns:
        top_5_diag = res["top_diagnoses"]["Diagnosis"].head(5).tolist() if not res["top_diagnoses"].empty else []
        trend_diag = df[df["_clean_diagnosis"].isin(top_5_diag)].groupby(["_appt_year_month", "_clean_diagnosis"])["_clean_pid"].count().reset_index()
        trend_diag.columns = ["Period", "Diagnosis", "Cases"]
        res["diagnosis_trends"] = trend_diag
    else:
        res["diagnosis_trends"] = pd.DataFrame()

    return res


def check_healthcare_quality(df, mapping=None):
    """
    Comprehensive healthcare-specific data quality guardian:
    Detects invalid ages, inconsistent admission/discharge dates,
    missing patient identification, missing diagnoses, duplicates, and outliers.
    """
    if mapping is None:
        mapping_info = map_healthcare_columns(df)
        mapping = mapping_info["mapping"]

    issues = []
    total_records = len(df)

    # 1. Duplicate records
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "Category": "Duplicate Records",
            "Severity": "High",
            "Affected Records": dup_count,
            "Impact": f"{round(dup_count / total_records * 100, 1)}% of rows are identical copies",
            "Recommendation": "Remove duplicate patient records to avoid double counting"
        })

    # 2. Missing Patient Identification
    if "_clean_pid" in df.columns:
        unknown_pids = (df["_clean_pid"] == "Unknown").sum()
        if unknown_pids > 0:
            issues.append({
                "Category": "Missing Patient ID",
                "Severity": "Critical",
                "Affected Records": int(unknown_pids),
                "Impact": "Unidentified patient visits cannot be linked to medical history",
                "Recommendation": "Impute or request patient MRN/ID from registration logs"
            })

    # 3. Invalid Ages (< 0 or > 120 or missing)
    if "_clean_age" in df.columns and "age" in mapping:
        invalid_ages = df[df["_clean_age"].isna() | (df["_clean_age"] < 0) | (df["_clean_age"] > 120)]
        invalid_age_count = len(invalid_ages)
        if invalid_age_count > 0:
            issues.append({
                "Category": "Invalid / Out-of-Range Age",
                "Severity": "Medium",
                "Affected Records": int(invalid_age_count),
                "Impact": "Distorts pediatric and geriatric risk cohort calculations",
                "Recommendation": "Clip or impute using median department age"
            })

    # 4. Inconsistent Dates (Discharge before Admission)
    if "_date_inconsistent" in df.columns:
        date_inconsistencies = int(df["_date_inconsistent"].sum())
        if date_inconsistencies > 0:
            issues.append({
                "Category": "Inconsistent Hospital Dates",
                "Severity": "Critical",
                "Affected Records": date_inconsistencies,
                "Impact": "Discharge date is earlier than admission date (negative stay)",
                "Recommendation": "Swap transposed dates or verify EHR discharge stamps"
            })

    # 5. Missing Diagnosis or Clinical Reason
    if "_clean_diagnosis" in df.columns and "diagnosis" in mapping:
        undiagnosed_count = int((df["_clean_diagnosis"] == "Undiagnosed").sum())
        if undiagnosed_count > 0:
            issues.append({
                "Category": "Missing Diagnosis",
                "Severity": "Medium",
                "Affected Records": undiagnosed_count,
                "Impact": "Encounter lacks recorded ICD/condition classification",
                "Recommendation": "Classify as 'Pending Clinical Review' or review chart notes"
            })

    # 6. Missing Doctor or Provider
    if "_clean_doctor" in df.columns and "doctor" in mapping:
        unassigned_docs = int((df["_clean_doctor"] == "Unassigned").sum())
        if unassigned_docs > 0:
            issues.append({
                "Category": "Unassigned Attending Doctor",
                "Severity": "Low",
                "Affected Records": unassigned_docs,
                "Impact": "Physician workload and attribution cannot be assigned",
                "Recommendation": "Assign to Department Lead or on-call hospitalist"
            })

    # 7. Overall Health Score for Healthcare Data
    # Base score 100 with weighted deductions
    penalty = 0.0
    if dup_count > 0:
        penalty += (dup_count / total_records) * 25.0
    if "_clean_pid" in df.columns:
        penalty += ((df["_clean_pid"] == "Unknown").sum() / total_records) * 30.0
    if "_clean_age" in df.columns and "age" in mapping:
        invalid_age_n = (df["_clean_age"].isna() | (df["_clean_age"] < 0) | (df["_clean_age"] > 120)).sum()
        penalty += (invalid_age_n / total_records) * 20.0
    if "_date_inconsistent" in df.columns:
        penalty += (df["_date_inconsistent"].sum() / total_records) * 25.0

    score = max(0.0, round(100.0 - penalty, 1))

    return {
        "health_score": score,
        "issues_table": pd.DataFrame(issues) if issues else pd.DataFrame(columns=["Category", "Severity", "Affected Records", "Impact", "Recommendation"]),
        "issue_count": len(issues),
        "dup_count": dup_count
    }


def clean_healthcare_dataset(df, mapping, fill_missing=True, remove_duplicates=True, fix_ages=True, fix_dates=True):
    """
    Cleans clinical dataset based on user-selected operations:
    - Removes duplicate records
    - Imputes missing demographics using median/mode
    - Corrects out-of-range ages to valid department bounds
    - Resolves transposed admission and discharge dates
    Returns the cleaned DataFrame.
    """
    cleaned = df.copy()

    # 1. Remove duplicate records
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    # 2. Fix inconsistent dates (swap if discharge < admission)
    if fix_dates and "admission_date" in mapping and "discharge_date" in mapping:
        adm_col = mapping["admission_date"]
        dis_col = mapping["discharge_date"]
        
        adm_dt = pd.to_datetime(cleaned[adm_col], errors="coerce")
        dis_dt = pd.to_datetime(cleaned[dis_col], errors="coerce")
        
        swap_mask = adm_dt.notna() & dis_dt.notna() & (dis_dt < adm_dt)
        if swap_mask.sum() > 0:
            cleaned.loc[swap_mask, adm_col], cleaned.loc[swap_mask, dis_col] = (
                cleaned.loc[swap_mask, dis_col],
                cleaned.loc[swap_mask, adm_col]
            )

    # 3. Fix invalid / out-of-range ages
    if fix_ages and "age" in mapping:
        age_col = mapping["age"]
        numeric_age = pd.to_numeric(cleaned[age_col], errors="coerce")
        median_age = numeric_age[(numeric_age >= 0) & (numeric_age <= 120)].median()
        if pd.isna(median_age):
            median_age = 40.0
            
        # Replace negative or > 120 or NaN with median valid age
        invalid_mask = numeric_age.isna() | (numeric_age < 0) | (numeric_age > 120)
        if invalid_mask.sum() > 0:
            cleaned.loc[invalid_mask, age_col] = int(round(median_age))

    # 4. Fill other missing values
    if fill_missing:
        for col in cleaned.columns:
            if col == mapping.get("age"):
                continue # Handled above
            if cleaned[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(cleaned[col]):
                    cleaned[col] = cleaned[col].fillna(cleaned[col].median())
                else:
                    mode_val = cleaned[col].mode()
                    if not mode_val.empty:
                        cleaned[col] = cleaned[col].fillna(mode_val[0])
                    else:
                        cleaned[col] = cleaned[col].fillna("Unknown")

    return cleaned
