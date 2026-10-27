import os
import tempfile
# Ensure matplotlib uses temporary writable directory on Windows
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "matplotlib_classpulse")

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Internal application modules
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
    save_engagement,
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
    process_video_file,
    generate_synthetic_demo_video_analysis
)
from utils.audio_analysis import (
    get_audio_backend_status,
    process_audio_file,
    generate_synthetic_demo_audio
)
from utils.recommendations import generate_teaching_recommendations

# Initialize storage upon script execution
init_storage()

# Streamlit Page Configuration
st.set_page_config(
    page_title="ClassPulse AI — Attendance & Engagement Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Blue, White, Light-Gray Palette)
CUSTOM_CSS = """
<style>
    /* Main container and typography */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Top Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1E40AF 0%, #2563EB 50%, #3B82F6 100%);
        color: white;
        padding: 1.6rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 14px rgba(37, 99, 235, 0.18);
    }
    .main-header h1 {
        color: #FFFFFF !important;
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .main-header p {
        color: #E0E7FF !important;
        font-size: 1.05rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
        font-weight: 400;
    }
    
    /* KPI Metric Cards */
    .kpi-container {
        display: flex;
        gap: 1rem;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #2563EB;
        border-radius: 10px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.08);
    }
    .kpi-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #64748B;
        margin-bottom: 0.25rem;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.2;
    }
    .kpi-subtext {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 0.35rem;
    }

    /* Content Cards */
    .pulse-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1.25rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04);
    }
    .pulse-card h3 {
        margin-top: 0;
        color: #1E293B;
        font-size: 1.2rem;
        font-weight: 600;
    }

    /* Recommendation card */
    .rec-card {
        background: #F8FAFC;
        border: 1px solid #CBD5E1;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
    }
    .rec-card.high-priority {
        border-left: 5px solid #EF4444;
        background: #FEF2F2;
    }
    .rec-card.medium-priority {
        border-left: 5px solid #F59E0B;
        background: #FFFBEB;
    }
    .rec-card.low-priority {
        border-left: 5px solid #10B981;
        background: #F0FDF4;
    }
    
    /* Demo badge */
    .demo-badge {
        display: inline-block;
        background: #EFF6FF;
        color: #1D4ED8;
        font-size: 0.75rem;
        font-weight: 600;
        padding: 0.2rem 0.55rem;
        border-radius: 4px;
        border: 1px solid #BFDBFE;
    }
    
    /* Disclaimer Note */
    .disclaimer-box {
        background: #F1F5F9;
        border-left: 4px solid #64748B;
        padding: 0.85rem 1.1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #475569;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- SIDEBAR NAVIGATION -----------------
with st.sidebar:
    st.title("ClassPulse AI")
    st.caption("Multimodal Classroom Intelligence")
    
    # Active navigation
    pages = [
        "Dashboard",
        "Student Management",
        "Attendance",
        "Engagement Analytics",
        "Video Analysis",
        "Audio Analysis",
        "AI Teaching Assistant",
        "Reports & Export"
    ]
    
    selected_page = st.radio("Navigation", pages, index=0)
    
    st.markdown("---")
    # Quick System Backend Status Indicator
    cv_info = get_cv_status()
    audio_info = get_audio_backend_status()
    
    st.caption("**System Capabilities Status**")
    st.markdown(f"• **Vision Engine:** `{cv_info['active_backend']}`")
    st.markdown(f"• **Audio Engine:** `{audio_info['active_backend']}`")
    st.markdown(f"• **Data Mode:** `Local CSV Storage`")
    
    st.markdown("---")
    st.caption("ClassPulse AI v1.0.0 (Local)")

# Common Header Banner
st.markdown("""
<div class="main-header">
    <h1>ClassPulse AI — Attendance & Engagement Intelligence</h1>
    <p>Smart classroom analytics powered by computer vision and AI-assisted insights.</p>
</div>
""", unsafe_allow_html=True)

# ==============================================================================
# PAGE 1: DASHBOARD
# ==============================================================================
if selected_page == "Dashboard":
    students_df = load_students()
    attendance_df = load_attendance()
    engagement_df = load_engagement()
    
    overall_stats = get_overall_attendance_stats(students_df, attendance_df)
    latest_stats = get_latest_session_stats(students_df, attendance_df)
    
    # Average engagement score
    if not engagement_df.empty:
        avg_eng_score = round(engagement_df['avg_engagement_score'].mean(), 1)
        eng_sessions_count = len(engagement_df)
    else:
        avg_eng_score = 0.0
        eng_sessions_count = 0

    # Display KPI Metric Cards
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    with k1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Total Students</div>
            <div class="kpi-value">{overall_stats['total_students']}</div>
            <div class="kpi-subtext">Active enrollment</div>
        </div>
        """, unsafe_allow_html=True)
    with k2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Present (Latest)</div>
            <div class="kpi-value" style="color: #10B981;">{latest_stats['present']}</div>
            <div class="kpi-subtext">{latest_stats['session_name'][:14]}...</div>
        </div>
        """, unsafe_allow_html=True)
    with k3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Absent (Latest)</div>
            <div class="kpi-value" style="color: #EF4444;">{latest_stats['absent']}</div>
            <div class="kpi-subtext">Requires follow-up</div>
        </div>
        """, unsafe_allow_html=True)
    with k4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Attendance Rate</div>
            <div class="kpi-value">{overall_stats['overall_rate']}%</div>
            <div class="kpi-subtext">Aggregate history</div>
        </div>
        """, unsafe_allow_html=True)
    with k5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Engagement Score</div>
            <div class="kpi-value" style="color: #2563EB;">{avg_eng_score}</div>
            <div class="kpi-subtext">Scale: 0 - 100</div>
        </div>
        """, unsafe_allow_html=True)
    with k6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Sessions Logged</div>
            <div class="kpi-value">{overall_stats['sessions_count']}</div>
            <div class="kpi-subtext">Total lectures</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # Notice for Demo Data
    has_demo = False
    if not students_df.empty and 'is_demo' in students_df.columns:
        if bool(students_df['is_demo'].any()):
            has_demo = True
    if has_demo:
        st.info("**DEMO DATA INCLUDED:** Realistic sample students, attendance, and engagement logs are currently loaded for demonstration. You can add your own real records anytime in Student Management or Attendance.")

    # Charts Section
    col_chart1, col_chart2 = st.columns([1, 1])
    
    with col_chart1:
        st.subheader("Attendance Distribution")
        if not attendance_df.empty:
            status_counts = attendance_df['status'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4.2))
            colors = {'Present': '#10B981', 'Absent': '#EF4444', 'Late': '#F59E0B'}
            labels = list(status_counts.keys())
            values = [status_counts[k] for k in labels]
            pie_colors = [colors.get(k, '#94A3B8') for k in labels]
            
            wedges, texts, autotexts = ax.pie(
                values, 
                labels=labels, 
                autopct='%1.1f%%', 
                colors=pie_colors, 
                startangle=140,
                textprops=dict(color="#1E293B", weight="bold")
            )
            for autotext in autotexts:
                autotext.set_color('white')
            ax.axis('equal')
            fig.patch.set_facecolor('#FFFFFF')
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No attendance records found to plot distribution.")

    with col_chart2:
        st.subheader("Engagement Trend Over Sessions")
        if not engagement_df.empty:
            fig2, ax2 = plt.subplots(figsize=(6, 4.2))
            eng_sorted = engagement_df.sort_values(by="date")
            x_labels = [f"{r['session_name'][:12]} ({r['date'][-5:]})" for _, r in eng_sorted.iterrows()]
            y_vals = eng_sorted['avg_engagement_score'].values
            
            ax2.plot(x_labels, y_vals, marker='o', linewidth=2.5, color='#2563EB', markersize=8)
            ax2.axhline(75, color='#10B981', linestyle='--', alpha=0.7, label='High (>=75)')
            ax2.axhline(50, color='#F59E0B', linestyle='--', alpha=0.7, label='Moderate (>=50)')
            ax2.set_ylim(0, 105)
            ax2.set_ylabel("Engagement Score (0-100)", color='#475569')
            ax2.set_facecolor('#F8FAFC')
            ax2.grid(True, linestyle=':', alpha=0.6)
            plt.xticks(rotation=20, ha='right', fontsize=9)
            ax2.legend(loc='lower left', fontsize=8)
            fig2.patch.set_facecolor('#FFFFFF')
            st.pyplot(fig2)
            plt.close(fig2)
        else:
            st.info("No engagement sessions logged yet.")

    # Recent Attendance Records Table
    st.subheader("Recent Attendance Records")
    if not attendance_df.empty:
        # Merge with student name if available
        merged_att = attendance_df.copy()
        if not students_df.empty:
            merged_att = merged_att.merge(
                students_df[['student_id', 'name', 'class_name']], 
                on='student_id', 
                how='left'
            )
        # Sort by timestamp desc
        recent_display = merged_att.sort_values(by="timestamp", ascending=False).head(10)
        
        # Format tags
        def style_status(val):
            if val == 'Present':
                return 'background-color: #D1FAE5; color: #065F46; font-weight: bold;'
            elif val == 'Absent':
                return 'background-color: #FEE2E2; color: #991B1B; font-weight: bold;'
            elif val == 'Late':
                return 'background-color: #FEF3C7; color: #92400E; font-weight: bold;'
            return ''
            
        display_cols = ['date', 'session_name', 'student_id', 'name', 'class_name', 'status', 'timestamp', 'is_demo']
        existing_cols = [c for c in display_cols if c in recent_display.columns]
        
        st.dataframe(
            recent_display[existing_cols].style.map(style_status, subset=['status']),
            use_container_width=True
        )
    else:
        st.info("No attendance records to display.")

    # Classroom Summary Card
    st.markdown("""
    <div class="pulse-card">
        <h3>Classroom Performance Summary</h3>
        <p>ClassPulse AI monitors continuous classroom indicators to assist educators with proactive engagement strategies. 
        Recent metrics reflect healthy student attendance and consistent active participation. For targeted teaching advice, explore the 
        <strong>AI Teaching Assistant</strong> tab.</p>
    </div>
    """, unsafe_allow_html=True)

# ==============================================================================
# PAGE 2: STUDENT MANAGEMENT
# ==============================================================================
elif selected_page == "Student Management":
    st.header("Student Roster Management")
    st.write("Maintain student records, enroll new learners, update details, and filter the roster.")
    
    students_df = load_students()
    
    tab_list, tab_add, tab_edit_delete = st.tabs(["Student Directory", "Add New Student", "Edit / Remove Student"])
    
    with tab_list:
        if students_df.empty:
            st.warning("No students currently registered. Use the 'Add New Student' tab to enroll students.")
        else:
            col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
            with col_s1:
                search_query = st.text_input("Search by Name or Student ID:", placeholder="e.g. Aarav, STU-101")
            with col_s2:
                classes = ["All Classes"] + sorted(list(students_df['class_name'].dropna().unique()))
                selected_class = st.selectbox("Filter Class:", classes)
            with col_s3:
                demo_filter = st.selectbox("Filter Type:", ["All Records", "Real Records Only", "Demo Records Only"])

            filtered_df = students_df.copy()
            if search_query:
                q = search_query.lower()
                filtered_df = filtered_df[
                    filtered_df['name'].str.lower().str.contains(q, na=False) |
                    filtered_df['student_id'].str.lower().str.contains(q, na=False)
                ]
            if selected_class != "All Classes":
                filtered_df = filtered_df[filtered_df['class_name'] == selected_class]
            if demo_filter == "Real Records Only":
                filtered_df = filtered_df[filtered_df['is_demo'] == False]
            elif demo_filter == "Demo Records Only":
                filtered_df = filtered_df[filtered_df['is_demo'] == True]

            st.write(f"Showing **{len(filtered_df)}** of **{len(students_df)}** registered students:")
            st.dataframe(filtered_df, use_container_width=True)

    with tab_add:
        st.subheader("Enroll a New Student")
        with st.form("add_student_form", clear_on_submit=True):
            col_a1, col_a2 = st.columns(2)
            with col_a1:
                new_id = st.text_input("Student ID (Unique)*", placeholder="e.g. STU-201")
                new_name = st.text_input("Full Name*", placeholder="e.g. Maya Lin")
            with col_a2:
                new_class = st.text_input("Class / Section*", value="CS-A 3rd Year")
                new_email = st.text_input("Email (Optional)", placeholder="e.g. maya.lin@campus.edu")
            
            submit_add = st.form_submit_button("Register Student", use_container_width=True)
            if submit_add:
                ok, msg = add_student(new_id, new_name, new_class, new_email, is_demo=False)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)

    with tab_edit_delete:
        st.subheader("Modify Existing Student Details")
        if students_df.empty:
            st.info("No students available to edit.")
        else:
            student_options = [f"{r['student_id']} - {r['name']}" for _, r in students_df.iterrows()]
            chosen_stu = st.selectbox("Select Student to Modify:", student_options)
            chosen_id = chosen_stu.split(" - ")[0]
            curr_row = students_df[students_df['student_id'] == chosen_id].iloc[0]

            with st.form("edit_student_form"):
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    edit_id = st.text_input("Student ID", value=curr_row['student_id'], disabled=True)
                    edit_name = st.text_input("Full Name", value=curr_row['name'])
                with col_e2:
                    edit_class = st.text_input("Class / Section", value=curr_row.get('class_name', ''))
                    edit_email = st.text_input("Email", value=curr_row.get('email', ''))
                
                col_btn1, col_btn2 = st.columns([1, 1])
                with col_btn1:
                    submit_update = st.form_submit_button("Save Changes", use_container_width=True)
                with col_btn2:
                    submit_del = st.form_submit_button("Delete Student", use_container_width=True)

                if submit_update:
                    ok, msg = update_student(chosen_id, edit_name, edit_class, edit_email)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

                if submit_del:
                    ok, msg = delete_student(chosen_id)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

# ==============================================================================
# PAGE 3: ATTENDANCE
# ==============================================================================
elif selected_page == "Attendance":
    st.header("Attendance Management")
    st.write("Mark, update, and review session-wise attendance records with automated percentage calculation.")
    
    students_df = load_students()
    attendance_df = load_attendance()
    
    tab_mark, tab_history = st.tabs(["Record / Update Attendance", "Attendance History & Filtering"])
    
    with tab_mark:
        if students_df.empty:
            st.error("No registered students found. Please register students in 'Student Management' first.")
        else:
            col_m1, col_m2, col_m3 = st.columns([1, 2, 1])
            with col_m1:
                session_date = st.date_input("Session Date", value=date.today())
            with col_m2:
                session_name = st.text_input("Session / Lecture Title*", value="Machine Learning Lec 16")
            with col_m3:
                filter_class = st.selectbox("Filter Class Roster:", ["All Classes"] + sorted(list(students_df['class_name'].dropna().unique())))
                
            roster = students_df.copy()
            if filter_class != "All Classes":
                roster = roster[roster['class_name'] == filter_class]
                
            date_str = session_date.strftime("%Y-%m-%d")
            
            # Check if attendance was already recorded for this session and date
            existing_records = {}
            if not attendance_df.empty:
                existing_for_session = attendance_df[
                    (attendance_df['date'] == date_str) & 
                    (attendance_df['session_name'].str.lower() == session_name.strip().lower())
                ]
                for _, r in existing_for_session.iterrows():
                    existing_records[str(r['student_id'])] = r['status']
                    
            if existing_records:
                st.info(f"Editing previously recorded attendance for '{session_name}' on {date_str}. Changes will update records.")
            
            # Interactive Attendance Input
            st.markdown("#### Student Attendance Roster")
            
            # Quick toggle buttons
            q1, q2, q3 = st.columns([1, 1, 2])
            with q1:
                mark_all_p = st.button("Mark All Present", use_container_width=True)
            with q2:
                mark_all_a = st.button("Mark All Absent", use_container_width=True)

            status_selections = {}
            status_options = ["Present", "Absent", "Late"]
            
            # Display roster with clean radio/selectbox controls
            cols_per_row = 2
            roster_list = roster.to_dict('records')
            
            for i in range(0, len(roster_list), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    if i + j < len(roster_list):
                        stu = roster_list[i + j]
                        s_id = str(stu['student_id'])
                        s_name = stu['name']
                        c_name = stu['class_name']
                        
                        # Determine default
                        if mark_all_p:
                            default_idx = 0
                        elif mark_all_a:
                            default_idx = 1
                        elif s_id in existing_records:
                            default_idx = status_options.index(existing_records[s_id]) if existing_records[s_id] in status_options else 0
                        else:
                            default_idx = 0
                            
                        with row_cols[j]:
                            st.markdown(f"**{s_name}** (`{s_id}`) — *{c_name}*")
                            status = st.radio(
                                f"Status for {s_id}",
                                status_options,
                                index=default_idx,
                                horizontal=True,
                                key=f"att_status_{s_id}_{date_str}_{session_name}",
                                label_visibility="collapsed"
                            )
                            status_selections[s_id] = status
                            st.write("")

            # Live preview calculation
            total_students_in_roster = len(status_selections)
            present_count = sum(1 for s in status_selections.values() if s == "Present")
            absent_count = sum(1 for s in status_selections.values() if s == "Absent")
            late_count = sum(1 for s in status_selections.values() if s == "Late")
            calc_rate = calculate_attendance_percentage(present_count, total_students_in_roster)
            
            st.markdown(f"""
            <div class="pulse-card" style="display:flex; justify-content:space-around; text-align:center;">
                <div><strong>Total Roster:</strong> {total_students_in_roster}</div>
                <div><strong style="color:#10B981;">Present:</strong> {present_count}</div>
                <div><strong style="color:#EF4444;">Absent:</strong> {absent_count}</div>
                <div><strong style="color:#F59E0B;">Late:</strong> {late_count}</div>
                <div><strong>Attendance Rate:</strong> <span style="font-size:1.15rem; font-weight:700; color:#2563EB;">{calc_rate}%</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Save Attendance Record", type="primary", use_container_width=True):
                if not session_name.strip():
                    st.error("Please enter a valid session / lecture title.")
                else:
                    ok, msg = record_attendance_batch(date_str, session_name, status_selections, is_demo=False)
                    if ok:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)

    with tab_history:
        st.subheader("Attendance History & Filtering")
        if attendance_df.empty:
            st.info("No attendance records found.")
        else:
            col_h1, col_h2, col_h3 = st.columns([1, 1, 1])
            with col_h1:
                all_dates = ["All Dates"] + sorted(list(attendance_df['date'].dropna().unique()), reverse=True)
                filt_date = st.selectbox("Select Date:", all_dates)
            with col_h2:
                all_sessions = ["All Sessions"] + sorted(list(attendance_df['session_name'].dropna().unique()))
                filt_session = st.selectbox("Select Session:", all_sessions)
            with col_h3:
                all_statuses = ["All Statuses", "Present", "Absent", "Late"]
                filt_status = st.selectbox("Status Filter:", all_statuses)
                
            filtered_history = attendance_df.copy()
            if filt_date != "All Dates":
                filtered_history = filtered_history[filtered_history['date'] == filt_date]
            if filt_session != "All Sessions":
                filtered_history = filtered_history[filtered_history['session_name'] == filt_session]
            if filt_status != "All Statuses":
                filtered_history = filtered_history[filtered_history['status'] == filt_status]
                
            # Merge with student names
            if not students_df.empty:
                filtered_history = filtered_history.merge(students_df[['student_id', 'name', 'class_name']], on='student_id', how='left')
                
            st.write(f"Displaying **{len(filtered_history)}** records:")
            st.dataframe(filtered_history, use_container_width=True)
            
            # Quick CSV Download
            csv_data = filtered_history.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Export Filtered Attendance to CSV",
                data=csv_data,
                file_name=f"attendance_export_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# ==============================================================================
# PAGE 4: ENGAGEMENT ANALYTICS
# ==============================================================================
elif selected_page == "Engagement Analytics":
    st.header("Classroom Engagement Analytics")
    st.write("Evaluate multimodal classroom indicators, configure indicator weightings, and track engagement trends.")
    
    engagement_df = load_engagement()
    
    # Engagement calculation methodology disclosure
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Transparent Methodology Notice:</strong> The composite engagement score (0–100) is a transparent, configurable indicator based on observable classroom metrics:
        optical presence rate, acoustic discussion activity, interaction frequency, and punctuality.
        <br><br>
        <strong>Ethical Clarification:</strong> These proxy scores <em>do NOT</em> measure individual student attention, comprehension, emotions, cognitive state, or learning outcomes.
    </div>
    """, unsafe_allow_html=True)
    
    tab_calc, tab_logs = st.tabs(["Interactive Engagement Calculator & Logger", "Session History & Trends"])
    
    with tab_calc:
        st.subheader("Session Engagement Evaluator")
        
        col_inp1, col_inp2 = st.columns(2)
        with col_inp1:
            eng_date = st.date_input("Session Date", value=date.today(), key="eng_date_picker")
            eng_session_name = st.text_input("Session Name", value="AI Lab Session 09", key="eng_sess_name")
            eng_duration = st.slider("Session Duration (minutes)", min_value=15, max_value=180, value=60, step=5)
            
        with col_inp2:
            st.markdown("**Observable Classroom Indicators (0-100%):**")
            opt_presence = st.slider("1. Optical Presence Rate (%) [Camera proxy]", 0.0, 100.0, 92.0, 1.0)
            acoustic_act = st.slider("2. Acoustic Discussion Activity (%) [Audio energy proxy]", 0.0, 100.0, 78.0, 1.0)
            interaction_rate = st.slider("3. Interactive Participation Rate (%) [Q&A / Polls]", 0.0, 100.0, 85.0, 1.0)
            punctuality_rate = st.slider("4. Punctuality / On-Time Arrival (%) [Attendance proxy]", 0.0, 100.0, 88.0, 1.0)

        # Configurable Weights
        with st.expander("Customize Indicator Weights (Transparent Formula)"):
            st.caption("Adjust the contribution of each observable factor to the composite score:")
            w_opt = st.slider("Weight: Optical Presence", 0.0, 1.0, 0.35, 0.05)
            w_aco = st.slider("Weight: Acoustic Activity", 0.0, 1.0, 0.25, 0.05)
            w_int = st.slider("Weight: Interaction & Questions", 0.0, 1.0, 0.25, 0.05)
            w_punc = st.slider("Weight: On-Time Attendance", 0.0, 1.0, 0.15, 0.05)

        # Calculate score
        res = calculate_composite_engagement(
            opt_presence, acoustic_act, interaction_rate, punctuality_rate,
            weights=(w_opt, w_aco, w_int, w_punc)
        )
        
        # Display Results
        c_res1, c_res2, c_res3 = st.columns([1, 1, 2])
        with c_res1:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: {res['color']};">
                <div class="kpi-label">Calculated Score</div>
                <div class="kpi-value" style="color: {res['color']};">{res['score']}</div>
                <div class="kpi-subtext">Scale: 0 - 100</div>
            </div>
            """, unsafe_allow_html=True)
        with c_res2:
            st.markdown(f"""
            <div class="kpi-card" style="border-left-color: {res['color']};">
                <div class="kpi-label">Classification</div>
                <div class="kpi-value" style="font-size: 1.45rem; color: {res['color']};">{res['classification']}</div>
                <div class="kpi-subtext">{res['badge']}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_res3:
            eng_notes = st.text_area("Educator Observations & Context Notes", placeholder="e.g. Conducted group problem solving lab. Active peer collaboration observed.")

        if st.button("Log Session Engagement Record", type="primary", use_container_width=True):
            ok, msg = add_engagement_record(
                str(eng_date),
                eng_session_name,
                eng_duration,
                res['score'],
                f"{res['classification']} Activity",
                res['classification'],
                notes=eng_notes,
                is_demo=False
            )
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)

    with tab_logs:
        st.subheader("Historical Engagement Sessions")
        if engagement_df.empty:
            st.info("No engagement sessions logged yet.")
        else:
            st.dataframe(engagement_df, use_container_width=True)
            
            # Trend visualization
            fig, ax = plt.subplots(figsize=(10, 4))
            dates_names = [f"{r['session_name']} ({r['date']})" for _, r in engagement_df.iterrows()]
            scores = engagement_df['avg_engagement_score'].values
            
            bar_colors = ['#10B981' if s >= 75 else '#F59E0B' if s >= 50 else '#EF4444' for s in scores]
            bars = ax.bar(dates_names, scores, color=bar_colors, width=0.5)
            ax.set_ylim(0, 105)
            ax.set_ylabel("Score (0-100)")
            ax.set_title("Session-Wise Engagement Scores and Classification")
            plt.xticks(rotation=25, ha='right')
            ax.grid(axis='y', linestyle=':', alpha=0.7)
            
            # Value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontweight='bold', fontsize=9)
                            
            st.pyplot(fig)
            plt.close(fig)

# ==============================================================================
# PAGE 5: VIDEO ANALYSIS
# ==============================================================================
elif selected_page == "Video Analysis":
    st.header("Video-Based Engagement Analysis")
    st.write("A practical, lightweight computer vision pipeline for sampling frames, detecting optical face presence, and calculating frame statistics.")
    
    cv_info = get_cv_status()
    st.caption(f"**Current Vision Engine:** `{cv_info['active_backend']}`")
    
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Computer Vision Scope & Ethical Safeguards:</strong>
        <ul>
            <li>Face detection operates solely as an <strong>optical presence indicator</strong> in the camera's field of view.</li>
            <li>This system <strong>does NOT identify individuals</strong>, does NOT perform biometric recognition, and does NOT claim face presence proves attendance.</li>
            <li>This system <strong>does NOT attempt emotion recognition</strong> or infer internal cognitive states from appearance.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col_v1, col_v2 = st.columns([2, 1])
    with col_v1:
        uploaded_video = st.file_uploader(
            "Upload Classroom Video Clip (MP4, AVI, MOV)", 
            type=["mp4", "avi", "mov", "mkv"],
            help="Select a short video clip (e.g. 10-60 seconds) from a classroom session."
        )
    with col_v2:
        st.markdown("**No video file available right now?**")
        use_demo_video = st.button("Load Demo Classroom Video Simulator", use_container_width=True)
        
    analysis_result = None
    
    if uploaded_video is not None:
        file_bytes = uploaded_video.read()
        with st.spinner("Processing video frames and running local face detection..."):
            analysis_result = process_video_file(file_bytes, max_samples=8)
            
    elif use_demo_video or st.session_state.get('demo_video_loaded', False):
        st.session_state['demo_video_loaded'] = True
        analysis_result = generate_synthetic_demo_video_analysis()

    if analysis_result:
        if not analysis_result["success"]:
            st.error(analysis_result.get("error", "Failed to process video."))
        else:
            meta = analysis_result.get("metadata", {})
            stats = analysis_result.get("stats", {})
            frames = analysis_result.get("frames", [])
            
            st.success("Video processing and frame analysis completed successfully.")
            
            # Metadata & Stats KPIs
            m1, m2, m3, m4, m5 = st.columns(5)
            with m1:
                st.metric("Duration", f"{meta.get('duration_seconds', 0)}s")
            with m2:
                st.metric("Resolution", f"{meta.get('width', 0)}x{meta.get('height', 0)}")
            with m3:
                st.metric("Frames Analyzed", stats.get("frames_analyzed", 0))
            with m4:
                st.metric("Frames with Faces", stats.get("frames_with_faces", 0))
            with m5:
                st.metric("Optical Presence Rate", f"{stats.get('optical_presence_rate', 0)}%")
                
            st.markdown("---")
            st.subheader("Sampled Frame Analysis with Face Detection Bounding Boxes")
            st.caption("Frames are evenly sampled across the recording timeline. Detected faces are outlined in blue.")
            
            # Display frames in a grid
            grid_cols = 3
            for i in range(0, len(frames), grid_cols):
                cols = st.columns(grid_cols)
                for j in range(grid_cols):
                    if i + j < len(frames):
                        f_info = frames[i + j]
                        with cols[j]:
                            st.image(
                                f_info["image"], 
                                caption=f"Frame #{f_info['frame_index']} (t={f_info['time_sec']}s) — {f_info['face_count']} faces detected",
                                use_column_width=True
                            )

# ==============================================================================
# PAGE 6: AUDIO ANALYSIS
# ==============================================================================
elif selected_page == "Audio Analysis":
    st.header("Classroom Audio Activity Analysis")
    st.write("Analyze acoustic energy, calculate RMS signal dynamics, and classify periods of classroom discussion versus quiet reflection.")
    
    audio_info = get_audio_backend_status()
    st.caption(f"**Current Audio Processing Engine:** `{audio_info['active_backend']}`")
    
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Acoustic Analysis Scope Notice:</strong>
        Audio RMS energy measures physical sound variation and acoustic amplitude in the classroom.
        It provides an objective signal of group discussion vs monologue vs silence. 
        It <strong>does NOT measure individual understanding or comprehension</strong>, and requires no speech recognition APIs or external cloud credentials.
    </div>
    """, unsafe_allow_html=True)
    
    col_a1, col_a2 = st.columns([2, 1])
    with col_a1:
        uploaded_audio = st.file_uploader(
            "Upload Classroom Audio File (WAV, MP3)", 
            type=["wav", "mp3", "ogg"],
            help="For zero-dependency environments, standard uncompressed WAV files are natively supported by SciPy."
        )
    with col_a2:
        st.markdown("**No audio recording on hand?**")
        use_demo_audio = st.button("Load Demo Classroom Audio Simulator", use_container_width=True)

    audio_res = None
    if uploaded_audio is not None:
        audio_bytes = uploaded_audio.read()
        with st.spinner("Decoding audio signal and computing RMS energy curve..."):
            audio_res = process_audio_file(audio_bytes, uploaded_audio.name)
    elif use_demo_audio or st.session_state.get('demo_audio_loaded', False):
        st.session_state['demo_audio_loaded'] = True
        audio_res = generate_synthetic_demo_audio()

    if audio_res:
        if not audio_res["success"]:
            st.error(audio_res.get("error", "Failed to process audio."))
        else:
            meta = audio_res.get("metadata", {})
            stats = audio_res.get("stats", {})
            wave_data = audio_res.get("waveform", {})
            rms_data = audio_res.get("rms_curve", {})
            
            st.success(f"Audio processing completed via `{audio_res.get('backend')}`.")
            
            # Metrics
            a1, a2, a3, a4, a5 = st.columns(5)
            with a1:
                st.metric("Duration", f"{meta.get('duration_seconds', 0)}s")
            with a2:
                st.metric("Sample Rate", f"{meta.get('sample_rate', 0)} Hz")
            with a3:
                st.metric("Mean RMS Energy", stats.get("mean_rms", 0))
            with a4:
                st.metric("Discussion Activity", f"{stats.get('high_activity_pct', 0)}%")
            with a5:
                st.metric("Quiet Periods", f"{stats.get('quiet_period_pct', 0)}%")
                
            st.info(f"**Overall Acoustic Classification:** {stats.get('overall_activity')}")
            
            # Waveform Plot
            st.subheader("Audio Signal Waveform & RMS Energy Curve")
            
            times = wave_data.get("times", [])
            amps = wave_data.get("amplitudes", [])
            rms_times = rms_data.get("times", [])
            rms_vals = rms_data.get("values", [])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
            
            # Raw Waveform
            ax1.plot(times, amps, color='#3B82F6', alpha=0.8, linewidth=0.8)
            ax1.set_ylabel("Amplitude")
            ax1.set_title("Classroom Sound Waveform (Normalized Amplitude)")
            ax1.set_facecolor('#F8FAFC')
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            # RMS Energy Curve
            ax2.plot(rms_times, rms_vals, color='#1E40AF', linewidth=1.8, label='RMS Energy Curve')
            ax2.axhline(0.35, color='#10B981', linestyle='--', label='High Discussion Threshold')
            ax2.axhline(0.12, color='#F59E0B', linestyle='--', label='Moderate Speech Threshold')
            ax2.set_ylabel("RMS Energy")
            ax2.set_xlabel("Time (seconds)")
            ax2.set_facecolor('#F8FAFC')
            ax2.grid(True, linestyle=':', alpha=0.6)
            ax2.legend(loc='upper right', fontsize=8)
            
            fig.patch.set_facecolor('#FFFFFF')
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

# ==============================================================================
# PAGE 7: AI TEACHING ASSISTANT
# ==============================================================================
elif selected_page == "AI Teaching Assistant":
    st.header("AI-Assisted Teaching Recommendations")
    st.write("Contextual pedagogical interventions derived deterministically from actual classroom attendance and engagement records.")
    
    st.markdown("""
    <div class="disclaimer-box">
        <strong>Engine Transparency Disclosure:</strong>
        This module uses a <strong>local rule-based pedagogical recommendation engine</strong>. It requires zero API keys, consumes zero cloud credits, and guarantees complete data privacy. It does not fabricate or hallucinate responses.
    </div>
    """, unsafe_allow_html=True)
    
    students_df = load_students()
    attendance_df = load_attendance()
    engagement_df = load_engagement()
    
    att_stats = get_overall_attendance_stats(students_df, attendance_df)
    
    # Calculate engagement trend
    avg_eng = 75.0
    trend = "stable"
    if not engagement_df.empty:
        avg_eng = round(float(engagement_df['avg_engagement_score'].mean()), 1)
        if len(engagement_df) >= 2:
            first_half = engagement_df.iloc[:len(engagement_df)//2]['avg_engagement_score'].mean()
            second_half = engagement_df.iloc[len(engagement_df)//2:]['avg_engagement_score'].mean()
            if second_half - first_half >= 5:
                trend = "improving"
            elif first_half - second_half >= 5:
                trend = "declining"
                
    eng_stats = {
        "avg_engagement": avg_eng,
        "trend": trend
    }
    
    # Generate recommendations
    recommendations = generate_teaching_recommendations(att_stats, eng_stats)
    
    st.subheader(f"Actionable Recommendations ({len(recommendations)} Generated)")
    
    for idx, rec in enumerate(recommendations):
        priority_class = "low-priority"
        if rec["priority"] == "High":
            priority_class = "high-priority"
        elif rec["priority"] == "Medium":
            priority_class = "medium-priority"
            
        st.markdown(f"""
        <div class="rec-card {priority_class}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <span style="font-weight:700; font-size:1.05rem; color:#1E293B;">{rec['badge']}</span>
                <span style="font-size:0.8rem; font-weight:600; text-transform:uppercase; color:#64748B;">{rec['category']} • Priority: {rec['priority']}</span>
            </div>
            <p><strong>Observed Data:</strong> {rec['observed_data']}</p>
            <p><strong>Recommendation:</strong> {rec['recommendation']}</p>
            <p><strong>Pedagogical Reason:</strong> {rec['reason']}</p>
            <div style="background:#FFFFFF; border:1px solid #E2E8F0; padding:0.6rem 0.8rem; border-radius:6px; margin-top:0.5rem;">
                <strong>Suggested Next Action:</strong><br>
                <span style="white-space: pre-line; color:#2563EB; font-size:0.9rem;">{rec['next_action']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    with st.expander("Recommendation Engine Scenario Simulator"):
        st.caption("Test how the rule-based recommendation engine responds under different hypothetical classroom scenarios:")
        sim_col1, sim_col2 = st.columns(2)
        with sim_col1:
            sim_rate = st.slider("Simulate Attendance Rate (%)", 30.0, 100.0, 62.0)
            sim_late = st.slider("Simulate Late Rate (%)", 0.0, 40.0, 22.0)
        with sim_col2:
            sim_eng = st.slider("Simulate Engagement Score (0-100)", 20.0, 100.0, 52.0)
            sim_trend = st.selectbox("Simulate Trend", ["declining", "stable", "improving"])
            
        sim_att = {"overall_rate": sim_rate, "sessions_count": 5, "late_rate": sim_late, "absent_count": 18}
        sim_eng_dict = {"avg_engagement": sim_eng, "trend": sim_trend}
        sim_recs = generate_teaching_recommendations(sim_att, sim_eng_dict)
        
        st.write(f"**Simulated Engine Output ({len(sim_recs)} suggestions):**")
        for s in sim_recs:
            st.info(f"**{s['badge']}** ({s['priority']} Priority)\n\n• **Recommendation:** {s['recommendation']}\n\n• **Action:** {s['next_action']}")

# ==============================================================================
# PAGE 8: REPORTS & EXPORT
# ==============================================================================
elif selected_page == "Reports & Export":
    st.header("Reports, Export & Data Administration")
    st.write("Generate consolidated CSV datasets, preview printable administrative summaries, and manage application storage.")
    
    students_df = load_students()
    attendance_df = load_attendance()
    engagement_df = load_engagement()
    
    overall_stats = get_overall_attendance_stats(students_df, attendance_df)
    student_summary = get_student_wise_summary(students_df, attendance_df)
    session_summary = get_session_wise_summary(attendance_df)
    
    tab_exports, tab_print, tab_admin = st.tabs(["CSV Exports", "Printable Summary Report", "Data Administration"])
    
    with tab_exports:
        st.subheader("Data Export Hub")
        st.write("Download certified CSV reports containing all classroom data:")
        
        c_exp1, c_exp2, c_exp3 = st.columns(3)
        with c_exp1:
            st.markdown("#### 1. Attendance Log")
            st.write(f"Total entries: **{len(attendance_df)}**")
            att_csv = attendance_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Attendance CSV",
                data=att_csv,
                file_name=f"classpulse_attendance_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with c_exp2:
            st.markdown("#### 2. Engagement Log")
            st.write(f"Total sessions: **{len(engagement_df)}**")
            eng_csv = engagement_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Engagement CSV",
                data=eng_csv,
                file_name=f"classpulse_engagement_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        with c_exp3:
            st.markdown("#### 3. Student Summary")
            st.write(f"Total students: **{len(student_summary)}**")
            stu_csv = student_summary.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Student Summary CSV",
                data=stu_csv,
                file_name=f"classpulse_student_summary_{date.today().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

        st.markdown("---")
        st.subheader("Student-Wise Attendance Breakdown")
        st.dataframe(student_summary, use_container_width=True)

    with tab_print:
        st.subheader("Printable Institutional Report View")
        st.caption("Press Ctrl+P (or Cmd+P on macOS) in your browser to print this summary or save it directly as a clean PDF.")
        
        # Formatted HTML Report Card
        report_html = f"""
        <div style="background:#FFFFFF; border:1px solid #CBD5E1; border-radius:8px; padding:2rem; font-family:'Inter', sans-serif;">
            <div style="border-bottom:2px solid #2563EB; padding-bottom:1rem; margin-bottom:1.5rem;">
                <h2 style="color:#1E40AF; margin:0;">ClassPulse AI — Executive Attendance & Engagement Report</h2>
                <p style="color:#64748B; margin:0.3rem 0 0 0;">Generated on: {datetime.now().strftime('%B %d, %Y - %I:%M %p')}</p>
            </div>
            
            <h3 style="color:#1E293B;">1. Executive Summary</h3>
            <table style="width:100%; border-collapse:collapse; margin-bottom:1.5rem;">
                <tr style="background:#F8FAFC;">
                    <td style="padding:8px; border:1px solid #E2E8F0;"><strong>Total Registered Students</strong></td>
                    <td style="padding:8px; border:1px solid #E2E8F0;">{overall_stats['total_students']}</td>
                    <td style="padding:8px; border:1px solid #E2E8F0;"><strong>Overall Attendance Rate</strong></td>
                    <td style="padding:8px; border:1px solid #E2E8F0; font-weight:bold; color:#2563EB;">{overall_stats['overall_rate']}%</td>
                </tr>
                <tr>
                    <td style="padding:8px; border:1px solid #E2E8F0;"><strong>Total Classroom Sessions</strong></td>
                    <td style="padding:8px; border:1px solid #E2E8F0;">{overall_stats['sessions_count']}</td>
                    <td style="padding:8px; border:1px solid #E2E8F0;"><strong>Late Arrival Rate</strong></td>
                    <td style="padding:8px; border:1px solid #E2E8F0;">{overall_stats['late_rate']}%</td>
                </tr>
                <tr style="background:#F8FAFC;">
                    <td style="padding:8px; border:1px solid #E2E8F0;"><strong>Total Marks Recorded</strong></td>
                    <td style="padding:8px; border:1px solid #E2E8F0;">{overall_stats.get('total_records', 0)}</td>
                    <td style="padding:8px; border:1px solid #E2E8F0;"><strong>System Status</strong></td>
                    <td style="padding:8px; border:1px solid #E2E8F0; color:#10B981; font-weight:bold;">Verified Local Storage</td>
                </tr>
            </table>

            <h3 style="color:#1E293B;">2. Student Performance Highlights</h3>
            <p>Summary of student cohort attendance rates and academic support classifications:</p>
            <ul>
                <li><strong>Good Standing (>=80%):</strong> {(student_summary['status_band'] == 'Good').sum()} students</li>
                <li><strong>Moderate Standing (65-79%):</strong> {(student_summary['status_band'] == 'Moderate').sum()} students</li>
                <li><strong>Attention Recommended (&lt;65%):</strong> {(student_summary['status_band'] == 'Needs Attention').sum()} students</li>
            </ul>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)

    with tab_admin:
        st.subheader("Data Reset & Storage Maintenance")
        st.write("Reset data back to the default demo state or refresh cache:")
        
        if st.button("Reset Datasets to Default Demo State", type="secondary"):
            ok, msg = reset_all_to_demo()
            if ok:
                st.success(msg)
                st.rerun()
            else:
                st.error("Reset operation failed.")
