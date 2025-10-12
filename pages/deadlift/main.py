"""
Deadlift Analysis Page
Main Streamlit page for deadlift assessment
"""

import streamlit as st
import pandas as pd
from shared.common import initialize_session_state, get_custom_css, calculate_summary_metrics, create_csv_download
from .video_processing import process_deadlift_video
from .ai_analysis import analyze_deadlift_with_openai
from .visualization import create_deadlift_charts

def deadlift_page():
    """Main deadlift analysis page"""
    
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
    if 'deadlift_rep_data' not in st.session_state:
        st.session_state.deadlift_rep_data = []
    if 'deadlift_video_processed' not in st.session_state:
        st.session_state.deadlift_video_processed = False
    if 'deadlift_output_video_path' not in st.session_state:
        st.session_state.deadlift_output_video_path = None
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏋️ Deadlift Assessment</h1>', unsafe_allow_html=True)
    
    # Assessment info
    st.markdown("""
    <div style="
        background-color: #f8d7da;
        border: 1px solid #2196f3;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    ">
        <h3 style="color: #721c24; margin: 0 0 15px 0;">🏋️ What is the Deadlift Assessment?</h3>
        <p style="color: #721c24; margin: 0 0 10px 0; font-size: 16px;">
            The <strong>Deadlift Assessment</strong> is a fundamental movement screening tool used by 
            fitness professionals, physical therapists, and coaches to evaluate:
        </p>
        <ul style="color: #721c24; margin: 0; padding-left: 20px;">
            <li><strong>Hip Hinge Mechanics</strong> - Proper hip movement pattern</li>
            <li><strong>Back Position</strong> - Spinal alignment and stability</li>
            <li><strong>Knee Tracking</strong> - Knee positioning and stability</li>
            <li><strong>Posterior Chain Strength</strong> - Glutes, hamstrings, and back</li>
            <li><strong>Core Stability</strong> - Trunk control during movement</li>
            <li><strong>Bilateral Symmetry</strong> - Left vs right side balance</li>
        </ul>
        <p style="color: #721c24; margin: 10px 0 0 0; font-size: 14px;">
            <em>This AI-powered assessment analyzes your movement patterns and provides corrective exercise recommendations 
            to improve hip hinge mechanics, back position, and lifting form.</em>
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
    
    # How to perform deadlift instructions
    st.markdown("### 🏋️ How to Perform a Deadlift")
    st.markdown("Follow these step-by-step instructions to perform a proper deadlift for assessment:")
    
    with st.container():
        st.markdown("**📋 Setup Phase:**")
        st.markdown("""
        1. **Stand with feet hip-width apart** - toes pointing forward or slightly outward
        2. **Position the barbell over mid-foot** - bar should be directly over the center of your foot
        3. **Bend at the hips and knees** - keep your chest up and back straight
        4. **Grip the bar** - hands just outside your legs, palms facing you
        5. **Engage your core** - brace your abdominals and maintain neutral spine
        """)
        
        st.markdown("**⬆️ Lifting Phase:**")
        st.markdown("""
        1. **Drive through your heels** - push the floor away with your feet
        2. **Extend your hips and knees simultaneously** - keep the bar close to your body
        3. **Stand up tall** - fully extend your hips and knees at the top
        4. **Keep your shoulders back** - maintain good posture at the top
        """)
        
        st.markdown("**⬇️ Lowering Phase:**")
        st.markdown("""
        1. **Hinge at the hips first** - push your hips back
        2. **Bend your knees** - lower the bar in a controlled manner
        3. **Keep the bar close** - maintain contact with your legs
        4. **Return to starting position** - ready for the next repetition
        """)
        
        st.markdown("**⚠️ Key Points to Remember:**")
        st.markdown("""
        - **Keep your back straight** - avoid rounding your spine
        - **Maintain neutral head position** - look forward, not up or down
        - **Breathe properly** - inhale on the way down, exhale on the way up
        - **Move slowly and controlled** - avoid jerky movements
        - **Perform 3-5 repetitions** - enough for analysis but not fatiguing
        """)
        
        st.info("💡 **Safety Note:** If you experience any pain or discomfort, stop immediately and consult a healthcare professional. This assessment is for movement analysis only and should not replace proper coaching or medical advice.")
    
    # Sidebar controls
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Analysis Settings")
    
    deadlift_down_threshold = st.sidebar.slider(
        "Deadlift Down Threshold (degrees)", 
        min_value=100, max_value=160, value=140, step=5,
        help="Hip angle threshold to detect start of lowering phase (hip hinge)"
    )
    
    deadlift_up_threshold = st.sidebar.slider(
        "Deadlift Up Threshold (degrees)", 
        min_value=150, max_value=180, value=160, step=5,
        help="Hip angle threshold to detect return to standing (hip hinge)"
    )
    
    manual_rotation = st.sidebar.selectbox(
        "Manual Rotation Override",
        options=[0, 90, 180, 270],
        format_func=lambda x: f"{x}°" if x > 0 else "Auto-detect",
        help="Override automatic rotation detection if video appears rotated"
    )
    
    # Video upload
    st.markdown("### 📹 Upload Your Deadlift Video")
    uploaded_file = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Upload a video of you performing deadlifts"
    )
    
    if uploaded_file is not None:
        if st.button("🎬 Process Video", type="primary"):
            with st.spinner("Processing video... This may take a few minutes."):
                rep_data, output_path = process_deadlift_video(
                    uploaded_file, 
                    deadlift_down_threshold, 
                    deadlift_up_threshold, 
                    manual_rotation
                )
            
            if rep_data:
                st.session_state.deadlift_rep_data = rep_data
                st.session_state.deadlift_video_processed = True
                st.session_state.deadlift_output_video_path = output_path
                st.success("✅ Video processed successfully!")
            else:
                st.error("❌ Failed to process video. Please try again.")
    
    # Display results
    if st.session_state.deadlift_video_processed and st.session_state.deadlift_rep_data:
        rep_data = st.session_state.deadlift_rep_data
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        # Summary metrics
        summary_metrics = calculate_summary_metrics(rep_data, "deadlift")
        if summary_metrics:  # Check if metrics exist
            # First row - Basic metrics
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total Reps", summary_metrics.get('total_reps', 0))
            with col_b:
                st.metric("Avg Hip Angle", f"{summary_metrics.get('avg_hip_angle', 0):.1f}°",
                         help="Average of the smallest hip angles reached during each rep (most bent positions). Lower values indicate more hip flexion.")
            with col_c:
                st.metric("Avg Duration", f"{summary_metrics.get('avg_duration', 0):.1f}s",
                         help="Average time taken for each complete rep (from lowest hip angle to highest hip angle)")
            
            # Second row - ROM metrics
            if any(key in summary_metrics for key in ['avg_hip_rom', 'avg_knee_rom']):
                st.markdown("#### 🎯 Range of Motion (ROM)")
                col_d, col_e = st.columns(2)
                with col_d:
                    if 'avg_hip_rom' in summary_metrics:
                        st.metric("Avg Hip ROM", f"{summary_metrics['avg_hip_rom']:.1f}°",
                                 help="Average range of motion for hip angles (from lowest to highest standing position)")
                with col_e:
                    if 'avg_knee_rom' in summary_metrics:
                        st.metric("Avg Knee ROM", f"{summary_metrics['avg_knee_rom']:.1f}°",
                                 help="Average range of motion for knee angles (from lowest to highest standing position)")
        else:
            st.warning("No metrics available for this data.")
        
        # Data table
        st.markdown("### 📋 Detailed Rep Data")
        df = pd.DataFrame(rep_data)
        st.dataframe(df, width='stretch')
        
        # Annotated video playback
        st.markdown("### 🎬 Annotated Video")
        if st.session_state.deadlift_output_video_path:
            try:
                with open(st.session_state.deadlift_output_video_path, "rb") as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)
            except Exception as e:
                st.error(f"Error loading video: {str(e)}")
        
        # Charts
        st.markdown("### 📈 Movement Analysis Charts")
        fig = create_deadlift_charts(rep_data)
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
                    analysis = analyze_deadlift_with_openai(rep_data, openai_api_key)
                
                st.markdown("### 💡 AI Analysis & Recommendations")
                st.markdown(f'<div class="analysis-box">{analysis}</div>', unsafe_allow_html=True)
            else:
                st.warning("Please enter your OpenAI API key to get AI analysis.")
        
        # Download options
        st.markdown("---")
        st.markdown("### 💾 Download Results")
        
        # Download processed video
        if st.session_state.deadlift_output_video_path:
            with open(st.session_state.deadlift_output_video_path, "rb") as video_file:
                st.download_button(
                    label="📥 Download Processed Video",
                    data=video_file.read(),
                    file_name=st.session_state.deadlift_output_video_path,
                    mime="video/mp4"
                )
        
        # Download CSV data
        csv, filename = create_csv_download(rep_data, "deadlift")
        if csv:
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
