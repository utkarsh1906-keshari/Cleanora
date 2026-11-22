"""
DataDoctor AI - Education Analytics Module
Comprehensive academic performance, attendance, class, department,
and data quality analytics for schools, colleges, and universities.
"""

import pandas as pd
import numpy as np


def map_education_columns(df):
    """
    Maps raw DataFrame columns to standard educational keys using
    case-insensitive keyword matching.
    """
    column_aliases = {
        "student_id": [
            "student_id", "studentid", "roll_no", "rollno", "id",
            "admission_no", "reg_no", "student_no", "enrollment_no", "stud_id"
        ],
        "student_name": [
            "student_name", "studentname", "name", "full_name",
            "pupil_name", "student", "candidate_name"
        ],
        "marks": [
            "marks", "score", "total_marks", "grade_point", "percentage",
            "final_score", "result", "exam_score", "points", "grade_pct"
        ],
        "subject": [
            "subject", "course", "subject_name", "topic", "module",
            "discipline", "paper", "course_name"
        ],
        "class": [
            "class", "grade", "section", "standard", "cohort",
            "year_group", "batch", "class_name", "grade_level"
        ],
        "department": [
            "department", "faculty", "stream", "dept", "branch",
            "academic_unit", "division", "program"
        ],
        "attendance": [
            "attendance", "attendance_percentage", "attendance_pct",
            "present_percentage", "att_rate", "presence", "att_pct"
        ],
        "teacher": [
            "teacher_name", "teacher", "faculty", "instructor",
            "professor", "educator", "faculty_name", "staff_name"
        ],
        "date": [
            "date", "exam_date", "term_date", "enrollment_date",
            "admission_date", "session_date", "test_date"
        ],
        "max_marks": [
            "max_marks", "total_possible", "out_of", "max_score", "maximum_marks"
        ],
        "pass_mark": [
            "pass_mark", "passing_marks", "passing_score", "min_pass", "cutoff"
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
            for norm_name, original_name in normalized_cols.items():
                for alias in aliases:
                    if alias in norm_name and original_name not in mapping.values():
                        found_col = original_name
                        break
                if found_col:
                    break
        if found_col:
            mapping[key] = found_col

    detected = list(mapping.keys())
    missing_critical = []
    if "student_id" not in mapping and "student_name" not in mapping:
        missing_critical.append("student identifier (e.g. Student_ID or Name)")
    if "marks" not in mapping and "attendance" not in mapping:
        missing_critical.append("academic metrics (e.g. Marks or Attendance)")

    return {
        "mapping": mapping,
        "detected": detected,
        "missing_critical": missing_critical,
        "optional_detected": [k for k in ["subject", "class", "department", "attendance", "teacher", "date"] if k in mapping],
        "optional_missing": [k for k in ["subject", "class", "department", "attendance", "teacher", "date"] if k not in mapping]
    }


def prepare_education_data(df, mapping=None):
    """
    Standardizes education dataset into internal clean columns.
    Computes percentages, grades, pass/fail status, and attendance indicators safely.
    """
    if mapping is None:
        mapping_info = map_education_columns(df)
        mapping = mapping_info["mapping"]

    work_df = df.copy()

    # 1. Student ID
    if "student_id" in mapping:
        work_df["_clean_sid"] = work_df[mapping["student_id"]].astype(str).str.strip().replace({"nan": "Unknown", "None": "Unknown", "": "Unknown"})
    elif "student_name" in mapping:
        work_df["_clean_sid"] = work_df[mapping["student_name"]].astype(str).str.strip()
    else:
        work_df["_clean_sid"] = [f"STU-{i+1:04d}" for i in range(len(work_df))]

    # 2. Student Name
    if "student_name" in mapping:
        work_df["_clean_sname"] = work_df[mapping["student_name"]].astype(str).str.strip()
    else:
        work_df["_clean_sname"] = work_df["_clean_sid"]

    # 3. Marks & Max Marks
    has_marks = False
    if "marks" in mapping:
        work_df["_clean_marks"] = pd.to_numeric(work_df[mapping["marks"]], errors="coerce")
        has_marks = True
        
        if "max_marks" in mapping:
            work_df["_clean_max_marks"] = pd.to_numeric(work_df[mapping["max_marks"]], errors="coerce").fillna(100.0)
        else:
            work_df["_clean_max_marks"] = 100.0
            
        work_df["_clean_pct"] = (work_df["_clean_marks"] / work_df["_clean_max_marks"] * 100.0).round(1)
        
        # Pass mark threshold
        if "pass_mark" in mapping:
            pass_threshold = float(pd.to_numeric(work_df[mapping["pass_mark"]], errors="coerce").fillna(40.0).iloc[0])
        else:
            pass_threshold = 40.0
            
        work_df["_is_passed"] = work_df["_clean_pct"] >= pass_threshold
        
        def assign_grade(pct):
            if pd.isna(pct) or pct < 0 or pct > 100:
                return "Ungraded/Invalid"
            elif pct >= 90:
                return "A+ (90-100%)"
            elif pct >= 80:
                return "A (80-89%)"
            elif pct >= 70:
                return "B (70-79%)"
            elif pct >= 60:
                return "C (60-69%)"
            elif pct >= pass_threshold:
                return "D (Pass)"
            else:
                return "F (Fail)"
                
        work_df["_grade"] = work_df["_clean_pct"].apply(assign_grade)
        work_df["_invalid_marks"] = work_df["_clean_marks"].isna() | (work_df["_clean_marks"] < 0) | (work_df["_clean_marks"] > work_df["_clean_max_marks"])
    else:
        work_df["_clean_marks"] = np.nan
        work_df["_clean_max_marks"] = 100.0
        work_df["_clean_pct"] = np.nan
        work_df["_is_passed"] = False
        work_df["_grade"] = "Ungraded"
        work_df["_invalid_marks"] = False

    # 4. Attendance
    has_attendance = False
    if "attendance" in mapping:
        work_df["_clean_attendance"] = pd.to_numeric(work_df[mapping["attendance"]], errors="coerce")
        has_attendance = True
        work_df["_invalid_attendance"] = work_df["_clean_attendance"].isna() | (work_df["_clean_attendance"] < 0) | (work_df["_clean_attendance"] > 100)
        work_df["_low_attendance"] = work_df["_clean_attendance"] < 75.0
    else:
        work_df["_clean_attendance"] = np.nan
        work_df["_invalid_attendance"] = False
        work_df["_low_attendance"] = False

    # 5. Subject
    has_subject = False
    if "subject" in mapping:
        has_subject = True
        work_df["_clean_subject"] = work_df[mapping["subject"]].astype(str).str.strip().replace({"nan": "General", "None": "General", "": "General"})
    else:
        work_df["_clean_subject"] = "General Curriculum"

    # 6. Class
    has_class = False
    if "class" in mapping:
        has_class = True
        work_df["_clean_class"] = work_df[mapping["class"]].astype(str).str.strip().replace({"nan": "General Class", "None": "General Class", "": "General Class"})
    else:
        work_df["_clean_class"] = "All Classes"

    # 7. Department / Stream
    has_dept = False
    if "department" in mapping:
        has_dept = True
        work_df["_clean_dept"] = work_df[mapping["department"]].astype(str).str.strip().replace({"nan": "Main Academic", "None": "Main Academic", "": "Main Academic"})
    else:
        work_df["_clean_dept"] = "Academic Division"

    # 8. Teacher
    has_teacher = False
    if "teacher" in mapping:
        has_teacher = True
        work_df["_clean_teacher"] = work_df[mapping["teacher"]].astype(str).str.strip().replace({"nan": "Unassigned", "None": "Unassigned", "": "Unassigned"})
    else:
        work_df["_clean_teacher"] = "Faculty Member"

    # 9. Dates
    has_date = False
    if "date" in mapping:
        work_df["_clean_date"] = pd.to_datetime(work_df[mapping["date"]], errors="coerce")
        if work_df["_clean_date"].notna().sum() > 0:
            has_date = True
            work_df["_year_month"] = work_df["_clean_date"].dt.strftime("%Y-%m")
            work_df["_month_label"] = work_df["_clean_date"].dt.strftime("%b %Y")
            work_df["_date_only"] = work_df["_clean_date"].dt.date

    metadata = {
        "has_marks": has_marks,
        "has_attendance": has_attendance,
        "has_subject": has_subject,
        "has_class": has_class,
        "has_dept": has_dept,
        "has_teacher": has_teacher,
        "has_date": has_date
    }

    return work_df, mapping, metadata


def summarize_education(df):
    """
    Computes high-level executive KPIs for educational institutions.
    """
    total_records = len(df)
    total_students = df["_clean_sid"].nunique() if "_clean_sid" in df.columns else total_records
    
    valid_teachers = df[df["_clean_teacher"] != "Unassigned"]["_clean_teacher"] if "_clean_teacher" in df.columns else pd.Series([], dtype=str)
    total_teachers = int(valid_teachers.nunique()) if len(valid_teachers) > 0 else 0
    
    total_classes = int(df["_clean_class"].nunique()) if "_clean_class" in df.columns else 0
    total_subjects = int(df["_clean_subject"].nunique()) if "_clean_subject" in df.columns else 0
    total_depts = int(df["_clean_dept"].nunique()) if "_clean_dept" in df.columns else 0

    # Average Marks
    valid_marks = df[df["_clean_pct"].between(0, 100)]["_clean_pct"] if "_clean_pct" in df.columns else pd.Series([], dtype=float)
    avg_marks = round(float(valid_marks.mean()), 1) if len(valid_marks) > 0 else None

    # Pass Percentage
    if "_is_passed" in df.columns and len(valid_marks) > 0:
        pass_pct = round(float(df[df["_clean_pct"].between(0, 100)]["_is_passed"].mean() * 100.0), 1)
    else:
        pass_pct = None

    # Overall Attendance
    valid_att = df[df["_clean_attendance"].between(0, 100)]["_clean_attendance"] if "_clean_attendance" in df.columns else pd.Series([], dtype=float)
    avg_attendance = round(float(valid_att.mean()), 1) if len(valid_att) > 0 else None

    return {
        "total_records": total_records,
        "total_students": total_students,
        "total_teachers": total_teachers,
        "total_classes": total_classes,
        "total_subjects": total_subjects,
        "total_departments": total_depts,
        "avg_marks": avg_marks,
        "pass_pct": pass_pct,
        "avg_attendance": avg_attendance
    }


def get_performance_analytics(df):
    """
    Computes grade distribution, subject-wise scores, class rankings,
    top-performing students, and sensitive academic support lists.
    """
    res = {}

    # 1. Grade Distribution
    if "_grade" in df.columns:
        valid_grades = df[df["_grade"] != "Ungraded/Invalid"]
        grade_counts = valid_grades["_grade"].value_counts().reset_index()
        grade_counts.columns = ["Grade", "Count"]
        grade_counts["Percentage"] = (grade_counts["Count"] / max(1, len(valid_grades)) * 100.0).round(1)
        res["grade_distribution"] = grade_counts
    else:
        res["grade_distribution"] = pd.DataFrame()

    # 2. Pass vs Fail
    if "_is_passed" in df.columns:
        valid_passed = df[df["_clean_pct"].between(0, 100)]["_is_passed"]
        passed_n = int(valid_passed.sum())
        failed_n = int((~valid_passed).sum())
        res["pass_fail_summary"] = pd.DataFrame({
            "Status": ["Passed", "Failed"],
            "Count": [passed_n, failed_n],
            "Percentage": [round(passed_n / max(1, len(valid_passed)) * 100, 1),
                           round(failed_n / max(1, len(valid_passed)) * 100, 1)]
        })
    else:
        res["pass_fail_summary"] = pd.DataFrame()

    # 3. Subject-wise performance
    if "_clean_subject" in df.columns and "_clean_pct" in df.columns:
        valid_subj = df[df["_clean_pct"].between(0, 100)].groupby("_clean_subject").agg(
            Students=("_clean_sid", "nunique"),
            Assessments=("_clean_sid", "count"),
            Avg_Marks=("_clean_pct", lambda m: round(m.mean(), 1)),
            Pass_Rate=("_is_passed", lambda p: round(p.mean() * 100, 1))
        ).reset_index().rename(columns={"_clean_subject": "Subject"}).sort_values("Avg_Marks", ascending=False)
        res["subject_performance"] = valid_subj
    else:
        res["subject_performance"] = pd.DataFrame()

    # 4. Class-wise performance
    if "_clean_class" in df.columns and "_clean_pct" in df.columns:
        valid_cls = df[df["_clean_pct"].between(0, 100)].groupby("_clean_class").agg(
            Students=("_clean_sid", "nunique"),
            Avg_Marks=("_clean_pct", lambda m: round(m.mean(), 1)),
            Pass_Rate=("_is_passed", lambda p: round(p.mean() * 100, 1))
        ).reset_index().rename(columns={"_clean_class": "Class"}).sort_values("Avg_Marks", ascending=False)
        res["class_performance"] = valid_cls
    else:
        res["class_performance"] = pd.DataFrame()

    # 5. Top Performing Students
    if "_clean_sid" in df.columns and "_clean_pct" in df.columns:
        valid_stu = df[df["_clean_pct"].between(0, 100)].groupby(["_clean_sid", "_clean_sname", "_clean_class"]).agg(
            Subjects=("_clean_subject", "nunique"),
            Avg_Score=("_clean_pct", lambda m: round(m.mean(), 1))
        ).reset_index().rename(columns={
            "_clean_sid": "Student ID",
            "_clean_sname": "Student Name",
            "_clean_class": "Class",
            "Avg_Score": "Average Score (%)"
        })
        res["top_students"] = valid_stu.sort_values("Average Score (%)", ascending=False).head(10)
        res["support_needed_students"] = valid_stu[valid_stu["Average Score (%)"] < 50.0].sort_values("Average Score (%)", ascending=True).head(10)
    else:
        res["top_students"] = pd.DataFrame()
        res["support_needed_students"] = pd.DataFrame()

    return res


def get_attendance_analytics(df):
    """
    Computes overall attendance rate, low-attendance alerts, and class comparisons.
    """
    res = {}
    valid_att = df[df["_clean_attendance"].between(0, 100)].copy() if "_clean_attendance" in df.columns else pd.DataFrame()

    if not valid_att.empty:
        res["valid_attendance_df"] = valid_att
        res["avg_attendance"] = round(float(valid_att["_clean_attendance"].mean()), 1)
        
        # Low Attendance (< 75%)
        low_att = valid_att[valid_att["_clean_attendance"] < 75.0]
        res["low_attendance_count"] = len(low_att)
        res["low_attendance_pct"] = round(len(low_att) / max(1, len(valid_att)) * 100.0, 1)
        res["low_attendance_table"] = low_att[["_clean_sid", "_clean_sname", "_clean_class", "_clean_attendance"]].rename(
            columns={
                "_clean_sid": "Student ID",
                "_clean_sname": "Student Name",
                "_clean_class": "Class",
                "_clean_attendance": "Attendance (%)"
            }
        ).sort_values("Attendance (%)", ascending=True)

        # Class-wise Attendance
        if "_clean_class" in valid_att.columns:
            cls_att = valid_att.groupby("_clean_class").agg(
                Students=("_clean_sid", "nunique"),
                Avg_Attendance=("_clean_attendance", lambda a: round(a.mean(), 1))
            ).reset_index().rename(columns={"_clean_class": "Class", "Avg_Attendance": "Average Attendance (%)"}).sort_values("Average Attendance (%)", ascending=False)
            res["class_attendance"] = cls_att
        else:
            res["class_attendance"] = pd.DataFrame()
    else:
        res["valid_attendance_df"] = pd.DataFrame()
        res["avg_attendance"] = None
        res["low_attendance_count"] = 0
        res["low_attendance_pct"] = 0.0
        res["low_attendance_table"] = pd.DataFrame()
        res["class_attendance"] = pd.DataFrame()

    return res


def get_class_department_analytics(df):
    """
    Calculates class student headcounts, department shares, and teacher workloads.
    """
    res = {}

    # 1. Class Comparison Matrix
    if "_clean_class" in df.columns:
        cls_df = df.groupby("_clean_class").agg(
            Students=("_clean_sid", "nunique"),
            Records=("_clean_sid", "count"),
            Avg_Marks=("_clean_pct", lambda m: round(m[m.between(0, 100)].mean(), 1) if len(m[m.between(0, 100)]) > 0 else np.nan),
            Pass_Rate=("_is_passed", lambda p: round(p.mean() * 100, 1) if "_is_passed" in df.columns else np.nan),
            Avg_Attendance=("_clean_attendance", lambda a: round(a[a.between(0, 100)].mean(), 1) if len(a[a.between(0, 100)]) > 0 else np.nan)
        ).reset_index().rename(columns={"_clean_class": "Class"}).sort_values("Students", ascending=False)
        res["class_summary"] = cls_df
    else:
        res["class_summary"] = pd.DataFrame()

    # 2. Department Breakdown
    if "_clean_dept" in df.columns:
        dept_df = df.groupby("_clean_dept").agg(
            Students=("_clean_sid", "nunique"),
            Classes=("_clean_class", "nunique"),
            Avg_Marks=("_clean_pct", lambda m: round(m[m.between(0, 100)].mean(), 1) if len(m[m.between(0, 100)]) > 0 else np.nan)
        ).reset_index().rename(columns={"_clean_dept": "Department"}).sort_values("Students", ascending=False)
        res["department_summary"] = dept_df
    else:
        res["department_summary"] = pd.DataFrame()

    # 3. Teacher Distribution
    if "_clean_teacher" in df.columns:
        t_df = df[df["_clean_teacher"] != "Unassigned"].groupby(["_clean_teacher", "_clean_subject"]).agg(
            Classes=("_clean_class", "nunique"),
            Students=("_clean_sid", "nunique"),
            Avg_Marks=("_clean_pct", lambda m: round(m[m.between(0, 100)].mean(), 1) if len(m[m.between(0, 100)]) > 0 else np.nan)
        ).reset_index().rename(columns={
            "_clean_teacher": "Teacher",
            "_clean_subject": "Subject",
            "Avg_Marks": "Avg Student Score (%)"
        }).sort_values("Students", ascending=False)
        res["teacher_summary"] = t_df
    else:
        res["teacher_summary"] = pd.DataFrame()

    return res


def get_education_trends(df, freq="monthly"):
    """
    Computes performance and attendance trends across calendar dates.
    """
    res = {}
    if "_clean_date" in df.columns and df["_clean_date"].notna().sum() > 0:
        valid_df = df.dropna(subset=["_clean_date"]).copy()
        date_col = "_year_month" if (freq == "monthly" and "_year_month" in valid_df.columns) else "_date_only"
        
        trend = valid_df.groupby(date_col).agg(
            Assessments=("_clean_sid", "count"),
            Avg_Score=("_clean_pct", lambda m: round(m[m.between(0, 100)].mean(), 1) if len(m[m.between(0, 100)]) > 0 else np.nan),
            Avg_Attendance=("_clean_attendance", lambda a: round(a[a.between(0, 100)].mean(), 1) if len(a[a.between(0, 100)]) > 0 else np.nan)
        ).reset_index().rename(columns={date_col: "Period"}).sort_values("Period")
        res["trend_df"] = trend
    else:
        res["trend_df"] = pd.DataFrame()

    return res


def check_education_quality(df, mapping=None):
    """
    Comprehensive academic data quality auditor:
    Detects missing student IDs, out-of-range marks, invalid attendance values,
    duplicate exam records, and computes a dedicated education health score.
    """
    if mapping is None:
        mapping_info = map_education_columns(df)
        mapping = mapping_info["mapping"]

    issues = []
    total_records = len(df)

    # 1. Duplicates
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        issues.append({
            "Category": "Duplicate Student Records",
            "Severity": "High",
            "Affected Records": dup_count,
            "Impact": f"{round(dup_count / total_records * 100, 1)}% of rows are identical duplicate submissions",
            "Recommendation": "Deduplicate student score records"
        })

    # 2. Missing Student ID
    if "_clean_sid" in df.columns:
        unknown_sid = int((df["_clean_sid"] == "Unknown").sum())
        if unknown_sid > 0:
            issues.append({
                "Category": "Missing Student ID",
                "Severity": "Critical",
                "Affected Records": unknown_sid,
                "Impact": "Unidentified academic records cannot be attributed to student transcript",
                "Recommendation": "Reconcile student roll number from enrollment registry"
            })

    # 3. Invalid Marks (< 0 or > 100)
    if "_invalid_marks" in df.columns and "marks" in mapping:
        invalid_m_count = int(df["_invalid_marks"].sum())
        if invalid_m_count > 0:
            issues.append({
                "Category": "Invalid / Out-of-Range Marks",
                "Severity": "High",
                "Affected Records": invalid_m_count,
                "Impact": "Scores negative or exceeding maximum possible marks distort class GPA",
                "Recommendation": "Clip or verify scoring scale against exam paper max score"
            })

    # 4. Invalid Attendance (< 0 or > 100%)
    if "_invalid_attendance" in df.columns and "attendance" in mapping:
        invalid_att_count = int(df["_invalid_attendance"].sum())
        if invalid_att_count > 0:
            issues.append({
                "Category": "Invalid Attendance Percentage",
                "Severity": "Medium",
                "Affected Records": invalid_att_count,
                "Impact": "Attendance values outside 0-100% boundary invalidate eligibility checks",
                "Recommendation": "Normalize attendance scale to 0-100%"
            })

    # 5. Missing Subject or Class
    if "_clean_subject" in df.columns and "subject" in mapping:
        missing_subj = int((df["_clean_subject"] == "General Curriculum").sum())
        if missing_subj > 0:
            issues.append({
                "Category": "Missing Subject Specification",
                "Severity": "Medium",
                "Affected Records": missing_subj,
                "Impact": "Assessment records lack specific course or subject attribution",
                "Recommendation": "Map assessment to course code"
            })

    # Quality Health Score
    penalty = 0.0
    if dup_count > 0:
        penalty += (dup_count / total_records) * 25.0
    if "_clean_sid" in df.columns:
        penalty += ((df["_clean_sid"] == "Unknown").sum() / total_records) * 30.0
    if "_invalid_marks" in df.columns and "marks" in mapping:
        penalty += (df["_invalid_marks"].sum() / total_records) * 25.0
    if "_invalid_attendance" in df.columns and "attendance" in mapping:
        penalty += (df["_invalid_attendance"].sum() / total_records) * 20.0

    score = max(0.0, round(100.0 - penalty, 1))

    return {
        "health_score": score,
        "issues_table": pd.DataFrame(issues) if issues else pd.DataFrame(columns=["Category", "Severity", "Affected Records", "Impact", "Recommendation"]),
        "issue_count": len(issues),
        "dup_count": dup_count
    }


def clean_education_dataset(df, mapping, fill_missing=True, remove_duplicates=True, fix_marks=True, fix_attendance=True):
    """
    Cleans educational dataset:
    - Removes duplicates
    - Clips out-of-range marks to [0, max_marks]
    - Clips out-of-range attendance to [0, 100]
    - Imputes missing scores using class median
    """
    cleaned = df.copy()

    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()

    # Fix marks
    if fix_marks and "marks" in mapping:
        m_col = mapping["marks"]
        numeric_m = pd.to_numeric(cleaned[m_col], errors="coerce")
        median_m = numeric_m[(numeric_m >= 0) & (numeric_m <= 100)].median()
        if pd.isna(median_m):
            median_m = 50.0
            
        # Clip or replace
        invalid_mask = numeric_m.isna() | (numeric_m < 0) | (numeric_m > 100)
        cleaned.loc[invalid_mask, m_col] = round(median_m, 1)

    # Fix attendance
    if fix_attendance and "attendance" in mapping:
        att_col = mapping["attendance"]
        numeric_att = pd.to_numeric(cleaned[att_col], errors="coerce")
        median_att = numeric_att[(numeric_att >= 0) & (numeric_att <= 100)].median()
        if pd.isna(median_att):
            median_att = 80.0
            
        invalid_att_mask = numeric_att.isna() | (numeric_att < 0) | (numeric_att > 100)
        cleaned.loc[invalid_att_mask, att_col] = round(median_att, 1)

    # Fill other missing values
    if fill_missing:
        for col in cleaned.columns:
            if col in [mapping.get("marks"), mapping.get("attendance")]:
                continue
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
