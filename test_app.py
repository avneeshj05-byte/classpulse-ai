import os
import sys
import unittest
import pandas as pd
import numpy as np

# Ensure project directory is in python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.storage import (
    init_storage,
    load_students,
    save_students,
    add_student,
    update_student,
    delete_student,
    load_attendance,
    save_attendance,
    record_attendance_batch,
    load_engagement,
    add_engagement_record,
    reset_all_to_demo
)
from utils.analytics import (
    calculate_attendance_percentage,
    get_overall_attendance_stats,
    get_latest_session_stats,
    get_student_wise_summary,
    get_session_wise_summary,
    calculate_composite_engagement
)
from utils.video_analysis import (
    get_cv_status,
    generate_synthetic_demo_video_analysis
)
from utils.audio_analysis import (
    get_audio_backend_status,
    generate_synthetic_demo_audio,
    compute_rms_curve
)
from utils.recommendations import generate_teaching_recommendations

class TestClassPulseAI(unittest.TestCase):
    
    def setUp(self):
        # Reset to known clean state
        reset_all_to_demo()

    def test_01_storage_initialization(self):
        init_storage()
        students = load_students()
        attendance = load_attendance()
        engagement = load_engagement()
        
        self.assertFalse(students.empty, "Students table should not be empty")
        self.assertFalse(attendance.empty, "Attendance table should not be empty")
        self.assertFalse(engagement.empty, "Engagement table should not be empty")
        self.assertIn("STU-101", students['student_id'].values)

    def test_02_student_management(self):
        # 1. Add student
        test_id = "TEST-999"
        ok, msg = add_student(test_id, "Test Student", "CS-Test", "test@test.edu")
        self.assertTrue(ok, msg)
        
        # 2. Duplicate prevention
        ok_dup, msg_dup = add_student(test_id, "Duplicate Student", "CS-Test", "")
        self.assertFalse(ok_dup, "Duplicate ID must be rejected")
        self.assertIn("already exists", msg_dup)
        
        # 3. Update student
        ok_up, msg_up = update_student(test_id, "Updated Student", "CS-Test-B", "new@test.edu")
        self.assertTrue(ok_up, msg_up)
        
        # Verify update
        students = load_students()
        updated_row = students[students['student_id'] == test_id].iloc[0]
        self.assertEqual(updated_row['name'], "Updated Student")
        self.assertEqual(updated_row['class_name'], "CS-Test-B")
        
        # 4. Delete student
        ok_del, msg_del = delete_student(test_id)
        self.assertTrue(ok_del, msg_del)
        students_post = load_students()
        self.assertNotIn(test_id, students_post['student_id'].values)

    def test_03_attendance_calculations(self):
        # Test percentage calculation
        self.assertEqual(calculate_attendance_percentage(8, 10), 80.0)
        self.assertEqual(calculate_attendance_percentage(0, 0), 0.0)
        self.assertEqual(calculate_attendance_percentage(1, 3), 33.3)
        
        # Batch record attendance
        batch = {
            "STU-101": "Present",
            "STU-102": "Absent",
            "STU-103": "Late"
        }
        ok, msg = record_attendance_batch("2026-10-30", "Unit Test Session", batch)
        self.assertTrue(ok)
        
        att = load_attendance()
        sess_rows = att[(att['date'] == "2026-10-30") & (att['session_name'] == "Unit Test Session")]
        self.assertEqual(len(sess_rows), 3)

    def test_04_empty_datasets_safe_handling(self):
        empty_students = pd.DataFrame(columns=["student_id", "name", "class_name", "email", "is_demo"])
        empty_attendance = pd.DataFrame(columns=["record_id", "date", "session_name", "student_id", "status", "timestamp"])
        
        # Must not raise exceptions
        stats = get_overall_attendance_stats(empty_students, empty_attendance)
        self.assertEqual(stats["total_students"], 0)
        self.assertEqual(stats["overall_rate"], 0.0)
        
        latest = get_latest_session_stats(empty_students, empty_attendance)
        self.assertEqual(latest["rate"], 0.0)
        
        stu_summary = get_student_wise_summary(empty_students, empty_attendance)
        self.assertTrue(stu_summary.empty)
        
        sess_summary = get_session_wise_summary(empty_attendance)
        self.assertTrue(sess_summary.empty)

    def test_05_engagement_composite_scoring(self):
        res = calculate_composite_engagement(
            optical_presence_rate=90.0,
            acoustic_activity_rate=80.0,
            interaction_rate=85.0,
            punctuality_rate=95.0
        )
        self.assertIn("score", res)
        self.assertIn("classification", res)
        self.assertTrue(0 <= res["score"] <= 100)
        self.assertEqual(res["classification"], "High")

    def test_06_video_analysis(self):
        status = get_cv_status()
        self.assertIn("active_backend", status)
        
        demo_vid = generate_synthetic_demo_video_analysis()
        self.assertTrue(demo_vid["success"])
        self.assertGreater(len(demo_vid["frames"]), 0)
        self.assertGreater(demo_vid["stats"]["total_faces_detected"], 0)

    def test_07_audio_analysis(self):
        status = get_audio_backend_status()
        self.assertIn("active_backend", status)
        
        demo_audio = generate_synthetic_demo_audio()
        self.assertTrue(demo_audio["success"])
        self.assertGreater(demo_audio["metadata"]["duration_seconds"], 0)
        self.assertIn("overall_activity", demo_audio["stats"])
        
        # Test pure numpy RMS calculation
        dummy_signal = np.array([0.1, 0.5, -0.3, 0.8, -0.2, 0.4] * 500, dtype=np.float32)
        times, rms = compute_rms_curve(dummy_signal, frame_length=100, hop_length=50)
        self.assertGreater(len(rms), 0)

    def test_08_recommendation_engine(self):
        # Scenario A: Low Attendance
        att_low = {"overall_rate": 55.0, "sessions_count": 4, "late_rate": 5.0, "absent_count": 20}
        eng_normal = {"avg_engagement": 75.0, "trend": "stable"}
        recs = generate_teaching_recommendations(att_low, eng_normal)
        self.assertTrue(any("Attendance Intervention" in r["category"] for r in recs))
        
        # Scenario B: High Late Rate
        att_late = {"overall_rate": 88.0, "sessions_count": 4, "late_rate": 25.0, "absent_count": 2}
        recs_late = generate_teaching_recommendations(att_late, eng_normal)
        self.assertTrue(any("Punctuality" in r["category"] for r in recs_late))
        
        # Scenario C: Low Engagement
        att_good = {"overall_rate": 90.0, "sessions_count": 4, "late_rate": 5.0, "absent_count": 2}
        eng_low = {"avg_engagement": 45.0, "trend": "declining"}
        recs_eng = generate_teaching_recommendations(att_good, eng_low)
        self.assertTrue(any("Engagement Boost" in r["category"] for r in recs_eng))

    def tearDown(self):
        # Reset back to clean demo data
        reset_all_to_demo()

if __name__ == "__main__":
    unittest.main()
