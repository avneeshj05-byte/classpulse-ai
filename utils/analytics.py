import pandas as pd
import numpy as np

def calculate_attendance_percentage(present_count: int, total_students: int) -> float:
    """Calculate attendance percentage: (Present Students / Total Students) * 100"""
    if total_students <= 0:
        return 0.0
    return round((present_count / total_students) * 100, 1)

def get_overall_attendance_stats(students_df: pd.DataFrame, attendance_df: pd.DataFrame) -> dict:
    """Computes high-level KPI metrics across all sessions and students."""
    total_registered = len(students_df) if not students_df.empty else 0
    
    if attendance_df.empty or total_registered == 0:
        return {
            "total_students": total_registered,
            "sessions_count": 0,
            "present_count": 0,
            "absent_count": 0,
            "late_count": 0,
            "overall_rate": 0.0,
            "late_rate": 0.0,
            "unique_dates": 0,
        }
        
    unique_sessions = attendance_df['session_name'].nunique()
    unique_dates = attendance_df['date'].nunique()
    
    status_counts = attendance_df['status'].value_counts()
    present_count = int(status_counts.get("Present", 0))
    absent_count = int(status_counts.get("Absent", 0))
    late_count = int(status_counts.get("Late", 0))
    total_records = len(attendance_df)
    
    # Standard formula: (Present / total recorded marks) * 100
    overall_rate = round((present_count / total_records * 100), 1) if total_records > 0 else 0.0
    late_rate = round((late_count / total_records * 100), 1) if total_records > 0 else 0.0
    
    return {
        "total_students": total_registered,
        "sessions_count": unique_sessions,
        "present_count": present_count,
        "absent_count": absent_count,
        "late_count": late_count,
        "total_records": total_records,
        "overall_rate": overall_rate,
        "late_rate": late_rate,
        "unique_dates": unique_dates,
    }

def get_latest_session_stats(students_df: pd.DataFrame, attendance_df: pd.DataFrame) -> dict:
    """Computes stats for the most recently logged date/session."""
    total_registered = len(students_df) if not students_df.empty else 0
    if attendance_df.empty:
        return {
            "session_name": "No sessions recorded",
            "date": "N/A",
            "present": 0,
            "absent": 0,
            "late": 0,
            "rate": 0.0,
            "total_marked": 0
        }
    
    # Sort by date / timestamp descending
    sorted_df = attendance_df.sort_values(by=["date", "timestamp"], ascending=[False, False])
    latest_date = sorted_df['date'].iloc[0]
    latest_session = sorted_df[sorted_df['date'] == latest_date]['session_name'].iloc[0]
    
    session_rows = sorted_df[(sorted_df['date'] == latest_date) & (sorted_df['session_name'] == latest_session)]
    present = int((session_rows['status'] == 'Present').sum())
    absent = int((session_rows['status'] == 'Absent').sum())
    late = int((session_rows['status'] == 'Late').sum())
    total_marked = len(session_rows)
    
    effective_total = total_registered if total_registered > 0 else total_marked
    rate = calculate_attendance_percentage(present, effective_total)
    
    return {
        "session_name": latest_session,
        "date": latest_date,
        "present": present,
        "absent": absent,
        "late": late,
        "total_marked": total_marked,
        "rate": rate
    }

def get_student_wise_summary(students_df: pd.DataFrame, attendance_df: pd.DataFrame) -> pd.DataFrame:
    """Generates per-student breakdown table with attendance percentage and status."""
    if students_df.empty:
        return pd.DataFrame(columns=["student_id", "name", "class_name", "sessions_recorded", "present", "absent", "late", "attendance_rate", "status_band"])
        
    records = []
    for _, stu in students_df.iterrows():
        s_id = str(stu['student_id'])
        stu_name = stu.get('name', 'Unknown')
        c_name = stu.get('class_name', 'General')
        is_demo = stu.get('is_demo', False)
        
        if attendance_df.empty:
            sessions = 0
            pres = 0
            ab = 0
            lat = 0
            rate = 0.0
        else:
            stu_att = attendance_df[attendance_df['student_id'].astype(str) == s_id]
            sessions = len(stu_att)
            pres = int((stu_att['status'] == 'Present').sum())
            ab = int((stu_att['status'] == 'Absent').sum())
            lat = int((stu_att['status'] == 'Late').sum())
            rate = calculate_attendance_percentage(pres, sessions) if sessions > 0 else 0.0
            
        if rate >= 80.0:
            band = "Good"
        elif rate >= 65.0:
            band = "Moderate"
        else:
            band = "Needs Attention" if sessions > 0 else "No Data"
            
        records.append({
            "student_id": s_id,
            "name": stu_name,
            "class_name": c_name,
            "sessions_recorded": sessions,
            "present": pres,
            "absent": ab,
            "late": lat,
            "attendance_rate": rate,
            "status_band": band,
            "is_demo": is_demo
        })
        
    return pd.DataFrame(records)

def get_session_wise_summary(attendance_df: pd.DataFrame) -> pd.DataFrame:
    """Generates session-by-session attendance table."""
    if attendance_df.empty:
        return pd.DataFrame(columns=["date", "session_name", "total_marked", "present", "absent", "late", "attendance_rate"])
        
    grouped = attendance_df.groupby(['date', 'session_name'])
    rows = []
    for (d, s), group in grouped:
        total = len(group)
        pres = int((group['status'] == 'Present').sum())
        ab = int((group['status'] == 'Absent').sum())
        lat = int((group['status'] == 'Late').sum())
        rate = round((pres / total * 100), 1) if total > 0 else 0.0
        rows.append({
            "date": d,
            "session_name": s,
            "total_marked": total,
            "present": pres,
            "absent": ab,
            "late": lat,
            "attendance_rate": rate
        })
    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values(by="date", ascending=True)
    return df_res

def calculate_composite_engagement(
    optical_presence_rate: float,
    acoustic_activity_rate: float,
    interaction_rate: float,
    punctuality_rate: float,
    weights: tuple[float, float, float, float] = (0.35, 0.25, 0.25, 0.15)
) -> dict:
    """
    Calculates transparent engagement indicator composite score (0-100).
    
    Formula:
    Score = w1 * OpticalPresence + w2 * AcousticActivity + w3 * Interaction + w4 * Punctuality
    
    Note: Engagement proxy scores do NOT measure individual comprehension, emotional state,
    or cognitive understanding. They provide an observational summary of classroom dynamics.
    """
    w1, w2, w3, w4 = weights
    # Normalize weights if sum != 1.0
    total_w = w1 + w2 + w3 + w4
    if total_w > 0:
        w1, w2, w3, w4 = w1/total_w, w2/total_w, w3/total_w, w4/total_w
        
    raw_score = (
        float(optical_presence_rate) * w1 +
        float(acoustic_activity_rate) * w2 +
        float(interaction_rate) * w3 +
        float(punctuality_rate) * w4
    )
    score = min(100.0, max(0.0, round(raw_score, 1)))
    
    if score >= 75.0:
        classification = "High"
        color = "#10B981" # Green
        badge = "High Engagement"
    elif score >= 50.0:
        classification = "Moderate"
        color = "#F59E0B" # Amber
        badge = "Moderate Engagement"
    else:
        classification = "Low"
        color = "#EF4444" # Red
        badge = "Low Engagement"
        
    return {
        "score": score,
        "classification": classification,
        "badge": badge,
        "color": color,
        "components": {
            "Optical Presence Factor (Visual)": optical_presence_rate,
            "Acoustic Activity Factor (Audio)": acoustic_activity_rate,
            "Interactive Participation (Q&A/Polls)": interaction_rate,
            "Punctuality / On-Time Factor": punctuality_rate,
        },
        "weights": {
            "Visual Presence Weight": round(w1, 2),
            "Acoustic Activity Weight": round(w2, 2),
            "Interaction Weight": round(w3, 2),
            "Punctuality Weight": round(w4, 2),
        },
        "explanation": "Calculated via weighted sum of observable visual presence, acoustic dynamics, interaction rates, and on-time arrival metrics."
    }
