import os
import pandas as pd
from datetime import datetime, date

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
STUDENTS_FILE = os.path.join(DATA_DIR, "students.csv")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.csv")
ENGAGEMENT_FILE = os.path.join(DATA_DIR, "engagement.csv")

DEFAULT_STUDENTS = [
    {"student_id": "STU-101", "name": "Aarav Sharma", "class_name": "CS-A 3rd Year", "email": "aarav.s@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
    {"student_id": "STU-102", "name": "Priya Patel", "class_name": "CS-A 3rd Year", "email": "priya.p@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
    {"student_id": "STU-103", "name": "Rahul Verma", "class_name": "CS-A 3rd Year", "email": "rahul.v@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
    {"student_id": "STU-104", "name": "Sneha Kulkarni", "class_name": "CS-A 3rd Year", "email": "sneha.k@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
    {"student_id": "STU-105", "name": "Ananya Iyer", "class_name": "CS-A 3rd Year", "email": "ananya.i@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
    {"student_id": "STU-106", "name": "Rohan Mehta", "class_name": "CS-A 3rd Year", "email": "rohan.m@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
    {"student_id": "STU-107", "name": "Deepa Nair", "class_name": "CS-A 3rd Year", "email": "deepa.n@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
    {"student_id": "STU-108", "name": "Vikram Singh", "class_name": "CS-A 3rd Year", "email": "vikram.s@campus.edu", "is_demo": True, "created_at": "2026-10-01 09:00:00"},
]

DEFAULT_ATTENDANCE = [
    # Day 1: Lecture 1
    {"record_id": "ATT-001", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-101", "status": "Present", "timestamp": "2026-10-23 09:05:00", "is_demo": True},
    {"record_id": "ATT-002", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-102", "status": "Present", "timestamp": "2026-10-23 09:04:12", "is_demo": True},
    {"record_id": "ATT-003", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-103", "status": "Present", "timestamp": "2026-10-23 09:06:05", "is_demo": True},
    {"record_id": "ATT-004", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-104", "status": "Late", "timestamp": "2026-10-23 09:16:30", "is_demo": True},
    {"record_id": "ATT-005", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-105", "status": "Present", "timestamp": "2026-10-23 09:02:18", "is_demo": True},
    {"record_id": "ATT-006", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-106", "status": "Absent", "timestamp": "2026-10-23 09:30:00", "is_demo": True},
    {"record_id": "ATT-007", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-107", "status": "Present", "timestamp": "2026-10-23 09:01:45", "is_demo": True},
    {"record_id": "ATT-008", "date": "2026-10-23", "session_name": "Machine Learning Lec 14", "student_id": "STU-108", "status": "Present", "timestamp": "2026-10-23 09:08:20", "is_demo": True},
    
    # Day 2: Lecture 2
    {"record_id": "ATT-009", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-101", "status": "Present", "timestamp": "2026-10-25 10:02:00", "is_demo": True},
    {"record_id": "ATT-010", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-102", "status": "Present", "timestamp": "2026-10-25 10:01:10", "is_demo": True},
    {"record_id": "ATT-011", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-103", "status": "Absent", "timestamp": "2026-10-25 10:30:00", "is_demo": True},
    {"record_id": "ATT-012", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-104", "status": "Present", "timestamp": "2026-10-25 10:03:40", "is_demo": True},
    {"record_id": "ATT-013", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-105", "status": "Present", "timestamp": "2026-10-25 10:05:10", "is_demo": True},
    {"record_id": "ATT-014", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-106", "status": "Late", "timestamp": "2026-10-25 10:18:22", "is_demo": True},
    {"record_id": "ATT-015", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-107", "status": "Present", "timestamp": "2026-10-25 10:00:55", "is_demo": True},
    {"record_id": "ATT-016", "date": "2026-10-25", "session_name": "Computer Vision Lec 15", "student_id": "STU-108", "status": "Absent", "timestamp": "2026-10-25 10:30:00", "is_demo": True},

    # Day 3: Today
    {"record_id": "ATT-017", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-101", "status": "Present", "timestamp": "2026-10-27 11:00:15", "is_demo": True},
    {"record_id": "ATT-018", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-102", "status": "Present", "timestamp": "2026-10-27 11:01:45", "is_demo": True},
    {"record_id": "ATT-019", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-103", "status": "Present", "timestamp": "2026-10-27 11:04:10", "is_demo": True},
    {"record_id": "ATT-020", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-104", "status": "Present", "timestamp": "2026-10-27 11:02:50", "is_demo": True},
    {"record_id": "ATT-021", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-105", "status": "Late", "timestamp": "2026-10-27 11:17:20", "is_demo": True},
    {"record_id": "ATT-022", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-106", "status": "Present", "timestamp": "2026-10-27 11:03:00", "is_demo": True},
    {"record_id": "ATT-023", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-107", "status": "Present", "timestamp": "2026-10-27 11:01:10", "is_demo": True},
    {"record_id": "ATT-024", "date": "2026-10-27", "session_name": "AI Lab Session 08", "student_id": "STU-108", "status": "Absent", "timestamp": "2026-10-27 11:30:00", "is_demo": True},
]

DEFAULT_ENGAGEMENT = [
    {
        "session_id": "ENG-001",
        "date": "2026-10-23",
        "session_name": "Machine Learning Lec 14",
        "duration_minutes": 60,
        "avg_engagement_score": 82.5,
        "activity_level": "High Activity",
        "classification": "High",
        "notes": "Interactive session with hands-on gradient descent demo. High student participation.",
        "is_demo": True,
        "created_at": "2026-10-23 10:05:00"
    },
    {
        "session_id": "ENG-002",
        "date": "2026-10-25",
        "session_name": "Computer Vision Lec 15",
        "duration_minutes": 55,
        "avg_engagement_score": 64.0,
        "activity_level": "Moderate Activity",
        "classification": "Moderate",
        "notes": "Theoretical coverage of convolutional filters. Pacing was slightly fast.",
        "is_demo": True,
        "created_at": "2026-10-25 11:00:00"
    },
    {
        "session_id": "ENG-003",
        "date": "2026-10-27",
        "session_name": "AI Lab Session 08",
        "duration_minutes": 90,
        "avg_engagement_score": 88.0,
        "activity_level": "High Activity",
        "classification": "High",
        "notes": "Group coding lab on multimodal agents. Active peer discussions and live troubleshooting.",
        "is_demo": True,
        "created_at": "2026-10-27 12:35:00"
    }
]

def init_storage():
    """Ensure data directory and base CSV files exist with default demo records."""
    os.makedirs(DATA_DIR, exist_ok=True)
    
    if not os.path.exists(STUDENTS_FILE) or os.path.getsize(STUDENTS_FILE) == 0:
        pd.DataFrame(DEFAULT_STUDENTS).to_csv(STUDENTS_FILE, index=False)
        
    if not os.path.exists(ATTENDANCE_FILE) or os.path.getsize(ATTENDANCE_FILE) == 0:
        pd.DataFrame(DEFAULT_ATTENDANCE).to_csv(ATTENDANCE_FILE, index=False)
        
    if not os.path.exists(ENGAGEMENT_FILE) or os.path.getsize(ENGAGEMENT_FILE) == 0:
        pd.DataFrame(DEFAULT_ENGAGEMENT).to_csv(ENGAGEMENT_FILE, index=False)

def load_students() -> pd.DataFrame:
    init_storage()
    try:
        df = pd.read_csv(STUDENTS_FILE)
        # Ensure correct column types
        if 'student_id' in df.columns:
            df['student_id'] = df['student_id'].astype(str)
        if 'is_demo' in df.columns:
            df['is_demo'] = df['is_demo'].astype(bool)
        return df
    except Exception:
        return pd.DataFrame(columns=["student_id", "name", "class_name", "email", "is_demo", "created_at"])

def save_students(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(STUDENTS_FILE, index=False)

def add_student(student_id: str, name: str, class_name: str, email: str = "", is_demo: bool = False) -> tuple[bool, str]:
    df = load_students()
    student_id = student_id.strip()
    name = name.strip()
    
    if not student_id:
        return False, "Student ID cannot be empty."
    if not name:
        return False, "Student Name cannot be empty."
    if not df.empty and student_id.lower() in df['student_id'].astype(str).str.lower().values:
        return False, f"Student ID '{student_id}' already exists. Duplicate IDs are not allowed."
    
    new_record = {
        "student_id": student_id,
        "name": name,
        "class_name": class_name.strip() or "General",
        "email": email.strip(),
        "is_demo": bool(is_demo),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    save_students(df)
    return True, f"Student {name} ({student_id}) successfully registered."

def update_student(student_id: str, name: str, class_name: str, email: str) -> tuple[bool, str]:
    df = load_students()
    if df.empty:
        return False, "No student records found."
    
    mask = df['student_id'].astype(str).str.lower() == student_id.strip().lower()
    if not mask.any():
        return False, f"Student ID '{student_id}' not found."
    
    df.loc[mask, 'name'] = name.strip()
    df.loc[mask, 'class_name'] = class_name.strip()
    df.loc[mask, 'email'] = email.strip()
    # Note: user edited so we keep or mark as non-demo if desired
    save_students(df)
    return True, f"Student details for {student_id} updated successfully."

def delete_student(student_id: str) -> tuple[bool, str]:
    df = load_students()
    if df.empty:
        return False, "No student records found."
    
    mask = df['student_id'].astype(str).str.lower() == student_id.strip().lower()
    if not mask.any():
        return False, f"Student ID '{student_id}' not found."
    
    df = df[~mask]
    save_students(df)
    
    # Also clean up attendance records for this student
    att_df = load_attendance()
    if not att_df.empty:
        att_df = att_df[att_df['student_id'].astype(str).str.lower() != student_id.strip().lower()]
        save_attendance(att_df)
        
    return True, f"Student {student_id} and associated attendance history deleted."

def load_attendance() -> pd.DataFrame:
    init_storage()
    try:
        df = pd.read_csv(ATTENDANCE_FILE)
        if 'student_id' in df.columns:
            df['student_id'] = df['student_id'].astype(str)
        if 'date' in df.columns:
            df['date'] = df['date'].astype(str)
        if 'is_demo' in df.columns:
            df['is_demo'] = df['is_demo'].astype(bool)
        return df
    except Exception:
        return pd.DataFrame(columns=["record_id", "date", "session_name", "student_id", "status", "timestamp", "is_demo"])

def save_attendance(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(ATTENDANCE_FILE, index=False)

def record_attendance_batch(date_str: str, session_name: str, attendance_dict: dict, is_demo: bool = False) -> tuple[bool, str]:
    """
    attendance_dict: {student_id: status_str}
    Overwrites or updates entries for that date and session, preserving other sessions.
    """
    df = load_attendance()
    date_str = str(date_str)
    session_name = session_name.strip() or "Standard Session"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    new_rows = []
    
    # Remove existing entries for this specific date + session to avoid duplicate duplicates on update
    if not df.empty:
        mask = (df['date'] == date_str) & (df['session_name'].str.lower() == session_name.lower())
        df = df[~mask].copy()
    
    idx_counter = len(df) + 1
    for s_id, status in attendance_dict.items():
        rec_id = f"ATT-{idx_counter:04d}"
        idx_counter += 1
        new_rows.append({
            "record_id": rec_id,
            "date": date_str,
            "session_name": session_name,
            "student_id": str(s_id),
            "status": status,
            "timestamp": timestamp_str,
            "is_demo": bool(is_demo)
        })
        
    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    save_attendance(df)
    return True, f"Attendance saved for {len(new_rows)} students for session '{session_name}' on {date_str}."

def load_engagement() -> pd.DataFrame:
    init_storage()
    try:
        df = pd.read_csv(ENGAGEMENT_FILE)
        if 'is_demo' in df.columns:
            df['is_demo'] = df['is_demo'].astype(bool)
        return df
    except Exception:
        return pd.DataFrame(columns=["session_id", "date", "session_name", "duration_minutes", "avg_engagement_score", "activity_level", "classification", "notes", "is_demo", "created_at"])

def save_engagement(df: pd.DataFrame):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(ENGAGEMENT_FILE, index=False)

def add_engagement_record(date_str: str, session_name: str, duration_minutes: int, avg_score: float, activity_level: str, classification: str, notes: str = "", is_demo: bool = False) -> tuple[bool, str]:
    df = load_engagement()
    session_id = f"ENG-{(len(df) + 1):03d}"
    new_record = {
        "session_id": session_id,
        "date": str(date_str),
        "session_name": session_name.strip() or "Classroom Session",
        "duration_minutes": int(duration_minutes),
        "avg_engagement_score": round(float(avg_score), 1),
        "activity_level": activity_level,
        "classification": classification,
        "notes": notes.strip(),
        "is_demo": bool(is_demo),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
    save_engagement(df)
    return True, f"Engagement record {session_id} created successfully."

def reset_all_to_demo():
    """Reset datasets back to default demo state."""
    os.makedirs(DATA_DIR, exist_ok=True)
    pd.DataFrame(DEFAULT_STUDENTS).to_csv(STUDENTS_FILE, index=False)
    pd.DataFrame(DEFAULT_ATTENDANCE).to_csv(ATTENDANCE_FILE, index=False)
    pd.DataFrame(DEFAULT_ENGAGEMENT).to_csv(ENGAGEMENT_FILE, index=False)
    return True, "All data successfully reset to initial demo dataset."
