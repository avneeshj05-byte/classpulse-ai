import pandas as pd
from typing import List, Dict

def generate_teaching_recommendations(
    attendance_stats: dict,
    engagement_stats: dict,
    student_summary_df: pd.DataFrame = None,
    audio_stats: dict = None,
    video_stats: dict = None
) -> List[Dict[str, str]]:
    """
    Local rule-based pedagogical recommendation engine.
    Analyzes actual attendance, engagement, and acoustic/visual signals to provide
    actionable, evidence-backed classroom interventions.
    """
    recommendations = []
    
    overall_rate = attendance_stats.get("overall_rate", 0.0)
    sessions_count = attendance_stats.get("sessions_count", 0)
    late_rate = attendance_stats.get("late_rate", 0.0)
    absent_count = attendance_stats.get("absent_count", 0)
    
    avg_eng = engagement_stats.get("avg_engagement", 75.0)
    eng_trend = engagement_stats.get("trend", "stable")
    
    # --- Rule 1: Attendance Health ---
    if sessions_count > 0:
        if overall_rate < 70.0:
            recommendations.append({
                "category": "Attendance Intervention",
                "priority": "High",
                "badge": "Critical Attendance Action",
                "observed_data": f"Overall classroom attendance is currently {overall_rate}% (below the 70% institutional target). A total of {absent_count} absences have been recorded across {sessions_count} sessions.",
                "recommendation": "Conduct targeted student outreach and review attendance patterns for recurring schedule conflicts.",
                "reason": "Early attendance interventions within the first few weeks have been demonstrated to significantly prevent course dropouts and reduce academic jeopardy.",
                "next_action": "1. Run an attendance export to identify students with <65% attendance.\n2. Send a supportive check-in email or arrange a brief 5-minute meeting during office hours to identify underlying blockers (transportation, health, prerequisites)."
            })
        elif overall_rate >= 85.0:
            recommendations.append({
                "category": "Attendance Health",
                "priority": "Low",
                "badge": "Strong Attendance Retention",
                "observed_data": f"Class attendance stands at a robust {overall_rate}% across {sessions_count} logged sessions.",
                "recommendation": "Maintain the current course structure and provide positive reinforcement.",
                "reason": "High attendance reflects consistent student routine, clear scheduling expectations, and perceived value in attending live class sessions.",
                "next_action": "Acknowledge the cohort's dedication with brief verbal praise at the start of the next lecture."
            })
            
    # --- Rule 2: Punctuality & Late Arrivals ---
    if late_rate >= 15.0:
        recommendations.append({
            "category": "Classroom Punctuality",
            "priority": "Medium",
            "badge": "High Late Arrival Rate",
            "observed_data": f"Late arrivals account for {late_rate}% of all recorded attendances.",
            "recommendation": "Incentivize on-time arrival with a 3-minute opening bell-ringer activity or low-stakes warmup quiz.",
            "reason": "Students arriving late disrupt classroom momentum and frequently miss key introductory context and agenda framing.",
            "next_action": "Start the class with an ungraded 2-question interactive warm-up quiz or recap question within the first 5 minutes."
        })
        
    # --- Rule 3: Engagement Level & Dynamics ---
    if avg_eng < 60.0:
        recommendations.append({
            "category": "Engagement Boost",
            "priority": "High",
            "badge": "Low Engagement Indicators",
            "observed_data": f"Average engagement score across evaluated sessions is {avg_eng}/100 ('Low' classification).",
            "recommendation": "Transition from continuous monologic lecture to active learning cycles every 15-20 minutes.",
            "reason": "Adult and college cognitive attention spans naturally experience micro-fatigue during passive listening beyond 18-20 minutes.",
            "next_action": "Implement the 'Think-Pair-Share' technique: pose a conceptual problem, give students 1 minute to think, 2 minutes to discuss with their neighbor, and call on 2 random pairs to share."
        })
    elif avg_eng >= 80.0:
        recommendations.append({
            "category": "Pedagogical Momentum",
            "priority": "Low",
            "badge": "High Classroom Engagement",
            "observed_data": f"Average engagement score is {avg_eng}/100, indicating highly vibrant classroom participation.",
            "recommendation": "Document and continue the specific instructional methods used during these sessions.",
            "reason": "Consistent high engagement scores correlate with deep cognitive processing and active student agency in learning.",
            "next_action": "Note the session format (e.g., live coding, case studies, hands-on labs) to replicate across future syllabus modules."
        })
        
    # --- Rule 4: Engagement Trend ---
    if eng_trend == "declining":
        recommendations.append({
            "category": "Pacing Adjustment",
            "priority": "Medium",
            "badge": "Downward Engagement Trend",
            "observed_data": "Engagement metrics indicate a downward trend over the last consecutive sessions.",
            "recommendation": "Check for cognitive overload or syllabus fatigue; introduce a mid-session review or formative checkpoint.",
            "reason": "Mid-semester cognitive fatigue often manifests as decreasing spontaneous classroom participation.",
            "next_action": "Insert a 5-minute anonymous 'Muddiest Point' checkpoint at the end of class asking: 'What was the single most confusing concept today?'"
        })
    elif eng_trend == "improving":
        recommendations.append({
            "category": "Positive Trend",
            "priority": "Low",
            "badge": "Upward Momentum",
            "observed_data": "Engagement has shown positive upward progression across recent classroom sessions.",
            "recommendation": "Reinforce the interactive activities introduced recently that contributed to this lift.",
            "reason": "Students are responding positively to recent pedagogical adjustments or increased collaborative formats.",
            "next_action": "Maintain the current ratio of lecture vs practical application."
        })

    # --- Rule 5: Audio Signal Insights (if provided) ---
    if audio_stats:
        quiet_pct = audio_stats.get("quiet_period_pct", 0)
        high_pct = audio_stats.get("high_activity_pct", 0)
        
        if quiet_pct > 65.0:
            recommendations.append({
                "category": "Acoustic Dynamics",
                "priority": "Medium",
                "badge": "High Silence / Monologue Proportion",
                "observed_data": f"Classroom audio analysis detected {quiet_pct}% quiet/low-energy periods during the analyzed recording.",
                "recommendation": "Introduce verbal question prompts or student response polls to stimulate auditory feedback and active discourse.",
                "reason": "Long continuous low-amplitude intervals often signify passive listening or reduced verbal interaction in the room.",
                "next_action": "Prepare 2 open-ended 'how' or 'why' prompt questions rather than yes/no questions for the upcoming class."
            })
        elif high_pct > 40.0:
            recommendations.append({
                "category": "Acoustic Dynamics",
                "priority": "Low",
                "badge": "High Discussion Energy",
                "observed_data": f"High acoustic activity accounted for {high_pct}% of the session time, indicating lively group work or discussion.",
                "recommendation": "Ensure discussion sessions include brief synthesis wrap-ups so key conclusions are formalized.",
                "reason": "Lively group work creates high enthusiasm; dedicating the final 5 minutes to synthesize ensures learning objectives are solidified.",
                "next_action": "Use a whiteboard to summarize 3 key takeaways volunteered by student groups at the end of the activity."
            })

    # --- Rule 6: Video Face Presence Insights (if provided) ---
    if video_stats:
        presence_rate = video_stats.get("optical_presence_rate", 100.0)
        avg_faces = video_stats.get("avg_faces_per_frame", 0.0)
        if presence_rate < 70.0:
            recommendations.append({
                "category": "Camera View Coverage",
                "priority": "Low",
                "badge": "Camera Field of View Check",
                "observed_data": f"Optical face detection was observed in {presence_rate}% of sampled video frames (average {avg_faces} faces per frame).",
                "recommendation": "Check classroom camera positioning and lighting conditions.",
                "reason": "Sub-optimal camera angle, backlighting from windows, or obstructed views reduce optical detection reliability.",
                "next_action": "Adjust webcam/camera to capture the primary seating rows with adequate front lighting."
            })

    # Default fallback recommendation if dataset is clean and empty
    if not recommendations:
        recommendations.append({
            "category": "General Best Practice",
            "priority": "Low",
            "badge": "Standard Teaching Guideline",
            "observed_data": "Classroom attendance and engagement metrics are balanced within normative parameters.",
            "recommendation": "Continue utilizing varied instructional formats (multimedia, problem-solving, brief lectures).",
            "reason": "Consistent multimodal pedagogical approaches maintain student curiosity and motivation over the semester.",
            "next_action": "Review upcoming lesson plans to ensure a healthy balance of concept introduction and practical exercises."
        })
        
    return recommendations
