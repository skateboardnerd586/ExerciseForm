"""
Main Streamlit Application for Overhead Squat Assessment
Refactored into modular structure for better maintainability
"""

import streamlit as st
import numpy as np
import pandas as pd
import os
from datetime import datetime

# Import our custom modules
from utils import (
    initialize_session_state, 
    get_custom_css, 
    get_assessment_info_html, 
    get_filming_tips_html,
    calculate_summary_metrics,
    create_csv_download
)
from video_processing import process_video
from ai_analysis import analyze_with_openai
from visualization import create_matplotlib_charts

# Page configuration
st.set_page_config(
    page_title="Overhead Squat Assessment AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)

# Initialize session state
initialize_session_state()

def main():
    st.markdown('<h1 class="main-header">🏋️ Overhead Squat Assessment AI</h1>', unsafe_allow_html=True)
    
    # Overhead Squat Assessment Information
    st.markdown(get_assessment_info_html(), unsafe_allow_html=True)
    
    # How to Perform Overhead Squat Assessment
    with st.container():
        st.markdown("### 🏋️ How to Perform the Overhead Squat Assessment")
        
        st.markdown("#### 📋 Setup Instructions:")
        st.markdown("""
        1. **Stand tall** with feet shoulder-width apart, toes pointing forward
        2. **Raise arms overhead** with elbows fully extended
        3. **Keep arms parallel** to each other and perpendicular to the floor
        4. **Maintain this overhead position** throughout the entire movement
        """)
        
        st.markdown("#### ⬇️ Movement Instructions:")
        st.markdown("""
        1. **Begin descent** by pushing hips back and down
        2. **Squat down** as low as possible while maintaining overhead position
        3. **Keep chest up** and maintain neutral spine
        4. **Hold briefly** at the bottom position
        5. **Return to standing** while keeping arms overhead
        """)
        
        st.markdown("💡 **Tip:** Perform 3-5 reps for best assessment results. The AI will analyze your movement patterns and provide corrective exercise recommendations.")
    
    # Important filming note
    st.markdown(get_filming_tips_html(), unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # OpenAI API Key
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your OpenAI API key for AI analysis"
    )
    
    # Video settings
    st.sidebar.subheader("Video Settings")
    
    manual_rotation = st.sidebar.selectbox(
        "Manual Rotation Override",
        options=[0, 90, 180, 270],
        format_func=lambda x: f"{x}° ({'Auto-detect only' if x == 0 else 'Override with ' + str(x) + '°'})",
        help="Override automatic rotation detection if needed. Most mobile videos need 90° or 270° rotation."
    )
    
    # Threshold settings
    st.sidebar.subheader("Detection Thresholds")
    st.sidebar.info("💡 **Note**: Due to model imperfections, maintain a ~20-degree difference between thresholds for reliable detection.")
    
    squat_down_threshold = st.sidebar.slider(
        "Squat Down Threshold (degrees)", 
        min_value=100, 
        max_value=160, 
        value=130,
        help="Angle below which squat phase begins"
    )
    squat_up_threshold = st.sidebar.slider(
        "Squat Up Threshold (degrees)", 
        min_value=140, 
        max_value=180, 
        value=150,
        help="Angle above which squat phase ends"
    )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Upload Overhead Squat Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'mov', 'avi', 'mkv'],
            help="Upload a video of someone performing overhead squats for movement assessment"
        )
        
        if uploaded_file is not None:
            if st.button("🚀 Process Video", type="primary"):
                with st.spinner("Processing video... This may take a few minutes."):
                    rep_data, output_video_path = process_video(
                        uploaded_file, 
                        squat_down_threshold, 
                        squat_up_threshold,
                        manual_rotation
                    )
                    st.session_state.rep_data = rep_data
                    st.session_state.output_video_path = output_video_path
                    st.session_state.video_processed = True
                    st.success(f"✅ Processing complete! Found {len(rep_data)} squats.")
    
    with col2:
        st.header("📊 Results")
        
        if st.session_state.video_processed and st.session_state.rep_data:
            rep_data = st.session_state.rep_data
            
            # Summary metrics
            metrics = calculate_summary_metrics(rep_data)
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Total Reps", metrics['total_reps'])
            
            with col2_2:
                st.metric("Avg Knee Angle", f"{metrics['avg_knee_angle']:.1f} deg")
            
            with col2_3:
                st.metric("Avg Duration", f"{metrics['avg_duration']:.1f}s")
            
            # Data table
            st.subheader("📋 Rep Details")
            st.info("📊 **Note**: All angles shown are measured at the deepest point of each overhead squat rep (bottom of squat when knee angles are smallest). This is the most critical position for assessing movement quality and identifying limitations.")
            df = pd.DataFrame(rep_data)
            st.dataframe(df, width='stretch')
            
            # Download data
            csv, filename = create_csv_download(rep_data)
            if csv:
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                    file_name=filename,
                mime="text/csv"
            )
    
    # Video Output Section
    if st.session_state.video_processed and st.session_state.output_video_path:
        st.header("🎥 Processed Video")
        
        col_vid1, col_vid2 = st.columns([2, 1])
        
        with col_vid1:
            st.subheader("📹 Annotated Video")
            if os.path.exists(st.session_state.output_video_path):
                # Display video
                video_file = open(st.session_state.output_video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                video_file.close()
            else:
                st.error("Video file not found. Please reprocess the video.")
        
        with col_vid2:
            st.subheader("📥 Download Video")
            if os.path.exists(st.session_state.output_video_path):
                with open(st.session_state.output_video_path, 'rb') as video_file:
                    st.download_button(
                        label="🎬 Download Annotated Video",
                        data=video_file.read(),
                        file_name=f"squat_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4",
                        mime="video/mp4",
                        type="primary"
                    )
            else:
                st.error("Video file not available for download.")
    
    # Visualizations
    if st.session_state.video_processed and st.session_state.rep_data:
        st.header("📈 Analysis Charts")
        
        fig = create_matplotlib_charts(st.session_state.rep_data)
        
        if fig:
            st.pyplot(fig)
    
    # AI Analysis
    if st.session_state.video_processed and st.session_state.rep_data:
        st.header("🤖 AI Analysis")
        
        if st.button("🧠 Analyze with AI", type="primary"):
            with st.spinner("AI is analyzing your squat data..."):
                analysis = analyze_with_openai(st.session_state.rep_data, openai_api_key)
                st.session_state.analysis_complete = True
        
        if st.session_state.analysis_complete:
            analysis_text = analyze_with_openai(st.session_state.rep_data, openai_api_key)
            st.markdown(f"""
            <div class="analysis-box" style="
                background-color: #f0f2f6;
                padding: 20px;
                border-radius: 10px;
                border-left: 4px solid #667eea;
                margin: 20px 0;
                max-height: none;
                overflow: visible;
                white-space: pre-wrap;
            ">
            {analysis_text}
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()