"""
Overhead Squat Analysis Page
Main Streamlit page for overhead squat assessment
"""

import streamlit as st
import pandas as pd
from shared.common import initialize_session_state, get_custom_css, calculate_summary_metrics, create_csv_download
from .video_processing import process_overhead_squat_video
from .ai_analysis import analyze_overhead_squat_with_openai
from .visualization import create_overhead_squat_charts

def overhead_squat_page():
    """Main overhead squat analysis page"""
    
    # Sidebar navigation
    st.sidebar.title("🏋️ Exercise Analysis")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📄 Navigation")
    
    if st.sidebar.button("Home", use_container_width=True):
        st.session_state.current_page = "home"
        st.rerun()
    
    if st.sidebar.button("Overhead Squat Analysis", use_container_width=True):
        st.session_state.current_page = "overhead_squat"
        st.rerun()
    
    if st.sidebar.button("Deadlift Analysis", use_container_width=True):
        st.session_state.current_page = "deadlift"
        st.rerun()
    
    # Initialize exercise-specific session state
    if 'overhead_squat_rep_data' not in st.session_state:
        st.session_state.overhead_squat_rep_data = []
    if 'overhead_squat_video_processed' not in st.session_state:
        st.session_state.overhead_squat_video_processed = False
    if 'overhead_squat_output_video_path' not in st.session_state:
        st.session_state.overhead_squat_output_video_path = None
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header"> Overhead Squat Assessment</h1>', unsafe_allow_html=True)
    
    # Assessment info
    st.markdown("""
    <div style="
        background-color: #e8f4fd;
        border: 1px solid #bee5eb;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    ">
        <h3 style="color: #0c5460; margin: 0 0 15px 0;">📋 What is the Overhead Squat Assessment?</h3>
        <p style="color: #0c5460; margin: 0 0 10px 0; font-size: 16px;">
            The <strong>Overhead Squat Assessment (OHSA)</strong> is a fundamental movement screening tool used by 
            fitness professionals, physical therapists, and coaches to evaluate:
        </p>
        <ul style="color: #0c5460; margin: 0; padding-left: 20px;">
            <li><strong>Ankle Mobility</strong> - How well your ankles flex during the squat</li>
            <li><strong>Hip Mobility</strong> - Hip flexibility and range of motion</li>
            <li><strong>Thoracic Spine Extension</strong> - Upper back mobility</li>
            <li><strong>Shoulder Mobility</strong> - Overhead position maintenance</li>
            <li><strong>Core Stability</strong> - Trunk control during movement</li>
            <li><strong>Balance & Coordination</strong> - Movement quality and symmetry</li>
        </ul>
        <p style="color: #0c5460; margin: 10px 0 0 0; font-size: 14px;">
            <em>This AI-powered assessment analyzes your movement patterns and provides corrective exercise recommendations 
            to improve mobility, strength, and movement quality.</em>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filming tips
    st.markdown("""
    <div style="
        background-color: #e8f5e8;
        border: 1px solid #c3e6c3;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    ">
        <h3 style="color: #2d5a2d; margin: 0 0 10px 0;">📱 Filming Tips for Best Results</h3>
        <p style="color: #2d5a2d; margin: 0; font-size: 16px;">
            <strong>📏 Make sure the entire body is visible in the camera view</strong> - from head to feet - 
            so the AI can properly track all key points for accurate movement analysis.
            <br><br>
            <strong>💡 Pro tip:</strong> Ensure good lighting and a clear background for the best pose detection results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Analysis Settings")
    
    squat_down_threshold = st.sidebar.slider(
        "Squat Down Threshold (degrees)", 
        min_value=90, max_value=160, value=130, step=5,
        help="Knee angle threshold to detect start of squatting down"
    )
    
    squat_up_threshold = st.sidebar.slider(
        "Squat Up Threshold (degrees)", 
        min_value=140, max_value=180, value=150, step=5,
        help="Knee angle threshold to detect return to standing"
    )
    
    manual_rotation = st.sidebar.selectbox(
        "Manual Rotation Override",
        options=[0, 90, 180, 270],
        format_func=lambda x: f"{x}°" if x > 0 else "Auto-detect",
        help="Override automatic rotation detection if video appears rotated"
    )
    
    # Video upload
    st.markdown("### 📹 Upload Your Overhead Squat Video")
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video of you performing overhead squats"
    )
    
    if uploaded_file is not None:
        if st.button("🎬 Process Video", type="primary"):
            with st.spinner("Processing video... This may take a few minutes."):
                rep_data, output_path = process_overhead_squat_video(
                    uploaded_file, 
                    squat_down_threshold, 
                    squat_up_threshold, 
                    manual_rotation
                )
            
            if rep_data:
                st.session_state.overhead_squat_rep_data = rep_data
                st.session_state.overhead_squat_video_processed = True
                st.session_state.overhead_squat_output_video_path = output_path
                st.success("✅ Video processed successfully!")
            else:
                st.error("❌ Failed to process video. Please try again.")
    
    # Display results
    if st.session_state.overhead_squat_video_processed and st.session_state.overhead_squat_rep_data:
        rep_data = st.session_state.overhead_squat_rep_data
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Summary metrics
        summary_metrics = calculate_summary_metrics(rep_data, "overhead_squat")
        if summary_metrics:  # Check if metrics exist
            # First row - Basic metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Reps", summary_metrics.get('total_reps', 0))
            with col_b:
                st.metric("Avg Knee Angle", f"{summary_metrics.get('avg_knee_angle', 0):.1f}°", 
                         help="Average of the smallest knee angles reached during each rep (deepest squat positions). Lower values indicate deeper squats.")
            with col_c:
                st.metric("Avg Duration", f"{summary_metrics.get('avg_duration', 0):.1f}s",
                         help="Average time taken for each complete rep (from highest knee angle to next highest)")
            
            # Second row - ROM metrics
            if any(key in summary_metrics for key in ['avg_knee_rom', 'avg_hip_rom', 'avg_shoulder_rom']):
                st.markdown("#### 🎯 Range of Motion (ROM)")
                col_d, col_e, col_f = st.columns(3)
                with col_d:
                    if 'avg_knee_rom' in summary_metrics:
                        st.metric("Avg Knee ROM", f"{summary_metrics['avg_knee_rom']:.1f}°",
                                 help="Average range of motion for knee angles (from lowest to highest standing position)")
                with col_e:
                    if 'avg_hip_rom' in summary_metrics:
                        st.metric("Avg Hip ROM", f"{summary_metrics['avg_hip_rom']:.1f}°",
                                 help="Average range of motion for hip angles (from lowest to highest standing position)")
                with col_f:
                    if 'avg_shoulder_rom' in summary_metrics:
                        st.metric("Avg Shoulder ROM", f"{summary_metrics['avg_shoulder_rom']:.1f}°",
                                 help="Average range of motion for shoulder angles (from lowest to highest standing position)")
        else:
            st.warning("No metrics available for this data.")
        
        # Data table
        st.markdown("### 📋 Detailed Rep Data")
        df = pd.DataFrame(rep_data)
        st.dataframe(df, width='stretch')
        
        # Annotated video playback
        st.markdown("### 🎬 Annotated Video")
        if st.session_state.overhead_squat_output_video_path:
            try:
                with open(st.session_state.overhead_squat_output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")
        
        # Charts
        st.markdown("### 📈 Movement Analysis Charts")
        fig = create_overhead_squat_charts(rep_data)
        if fig:
            st.pyplot(fig)
        
        # AI Analysis
        st.markdown("---")
        st.markdown("### 🤖 AI-Powered Analysis")
        
        # OpenAI API key input
        openai_api_key = st.text_input(
            "OpenAI API Key (Optional)", 
            type="password",
            help="Enter your OpenAI API key for AI-powered analysis and recommendations"
        )
        
        if st.button("🧠 Get AI Analysis", type="secondary"):
            if openai_api_key:
                with st.spinner("AI is analyzing your movement..."):
                    analysis = analyze_overhead_squat_with_openai(rep_data, openai_api_key)
                
                st.markdown("### 💡 AI Analysis & Recommendations")
                st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter your OpenAI API key to get AI analysis.")
        
        # Download options
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        # Download processed video
        if st.session_state.overhead_squat_output_video_path:
            with open(st.session_state.overhead_squat_output_video_path, "rb") as video_file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_file.read(),
                    file_name=st.session_state.overhead_squat_output_video_path,
                    mime="video/mp4"
                )
        
        # Download CSV data
        csv, filename = create_csv_download(rep_data, "overhead_squat")
        if csv:
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
