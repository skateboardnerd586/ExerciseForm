"""
Exercise Analysis Platform - Home Page
Main landing page with navigation to different exercise analyses
"""

import streamlit as st
from shared.common import initialize_session_state, get_custom_css

# Page configuration
st.set_page_config(
    page_title="Exercise Analysis Platform",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)

def main():
    """Main home page function"""
    
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
    
    # Initialize session state
    initialize_session_state()
    
    # Apply custom CSS
    st.markdown(get_custom_css(), unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🏋️ Exercise Analysis Platform</h1>', unsafe_allow_html=True)
    
    # Welcome message
    st.markdown("""
    <div style="
        background-color: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 30px;
        margin: 20px 0;
        text-align: center;
    ">
        <h2 style="color: #495057; margin: 0 0 20px 0;">Welcome to Your AI-Powered Exercise Analysis Platform</h2>
        <p style="color: #6c757d; margin: 0; font-size: 18px; line-height: 1.6;">
            Analyze your movement patterns with cutting-edge computer vision and AI technology. 
            Get detailed insights into your exercise form, identify imbalances, and receive 
            personalized recommendations for improvement.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Exercise selection cards
    st.markdown("### 🎯 Choose Your Exercise Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="
            background-color: #d1ecf1;
            border: 1px solid #2196f3;
            border-radius: 10px;
            padding: 20px;
            margin: 10px 0;
            text-align: center;
            height: 200px;
            display: flex;
            flex-direction: column;
            justify-content: center;
        ">
            <h3 style="color: #0c5460; margin: 0 0 15px 0;"> Overhead Squat Analysis</h3>
            <p style="color: #0c5460; margin: 0 0 15px 0;">
                Analyze ankle mobility, hip flexibility, shoulder mobility, and core stability
            </p>
            <p style="color: #0c5460; margin: 0; font-size: 14px;">
                <em>Perfect for assessing full-body movement patterns and mobility</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="
                background-color: #f8d7da;
                border: 1px solid #2196f3;
            border-radius: 10px;
            padding: 20px;
                margin: 10px 0;
            text-align: center;
                height: 200px;
                display: flex;
                flex-direction: column;
                justify-content: center;
            ">
                <h3 style="color: #721c24; margin: 0 0 15px 0;"> Deadlift Analysis</h3>
                <p style="color: #721c24; margin: 0 0 15px 0;">
                    Analyze hip hinge mechanics, back position, and lifting form
                </p>
                <p style="color: #721c24; margin: 0; font-size: 14px;">
                    <em>Perfect for assessing posterior chain strength and technique</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Navigation buttons
    st.markdown("### 🚀 Get Started")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        if st.button("Start Overhead Squat Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = "overhead_squat"
            st.rerun()
    
    with col_b:
        if st.button("Start Deadlift Analysis", use_container_width=True, type="primary"):
            st.session_state.current_page = "deadlift"
            st.rerun()
    
    # Features section
    st.markdown("---")
    st.markdown("### ✨ Platform Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h4 style="color: #495057;">🎥 Computer Vision</h4>
            <p style="color: #6c757d;">
                Advanced pose detection using YOLOv11 for accurate movement tracking
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h4 style="color: #495057;">🤖 AI Analysis</h4>
            <p style="color: #6c757d;">
                OpenAI-powered analysis with personalized recommendations
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px;">
            <h4 style="color: #495057;">📊 Detailed Reports</h4>
            <p style="color: #6c757d;">
                Comprehensive charts and data export for tracking progress
            </p>
            </div>
            """, unsafe_allow_html=True)

# Main execution
if __name__ == "__main__":
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "home"
    
    # Route to appropriate page
    current_page = st.session_state.current_page
    
    if current_page == 'home':
        main()
    elif current_page == 'overhead_squat':
        from pages.overhead_squat.main import overhead_squat_page
        overhead_squat_page()
    elif current_page == 'deadlift':
        from pages.deadlift.main import deadlift_page
        deadlift_page()
    else:
        main()