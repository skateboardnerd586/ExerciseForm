"""
Utility Module for Overhead Squat Assessment
Contains helper functions and constants
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime

def initialize_session_state():
    """Initialize session state variables"""
    if 'rep_data' not in st.session_state:
        st.session_state.rep_data = []
    if 'video_processed' not in st.session_state:
        st.session_state.video_processed = False
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'output_video_path' not in st.session_state:
        st.session_state.output_video_path = None

def get_custom_css():
    """Return custom CSS for the application"""
    return """
    <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            text-align: center;
            margin-bottom: 2rem;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #667eea;
        }
        .analysis-box {
            background-color: #f8f9fa;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e9ecef;
            margin: 1rem 0;
            max-height: none;
            overflow: visible;
            white-space: pre-wrap;
            word-wrap: break-word;
            color: #333333 !important;
            font-size: 16px;
            line-height: 1.6;
        }
        
        /* Mobile-specific improvements */
        @media (max-width: 768px) {
            .analysis-box {
                font-size: 18px !important;
                line-height: 1.7 !important;
                padding: 1rem !important;
                color: #2c3e50 !important;
            }
        }
    </style>
    """

def get_assessment_info_html():
    """Return HTML for assessment information section"""
    return """
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
    """

def get_filming_tips_html():
    """Return HTML for filming tips section"""
    return """
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
    """

def calculate_summary_metrics(rep_data):
    """Calculate summary metrics from rep data"""
    if not rep_data:
        return {}
    
    avg_left_knee = np.mean([rep['left_knee_angle'] for rep in rep_data])
    avg_right_knee = np.mean([rep['right_knee_angle'] for rep in rep_data])
    avg_duration = np.mean([rep['duration_seconds'] for rep in rep_data])
    
    return {
        'total_reps': len(rep_data),
        'avg_knee_angle': (avg_left_knee + avg_right_knee) / 2,
        'avg_duration': avg_duration
    }

def create_csv_download(rep_data):
    """Create CSV data for download"""
    if not rep_data:
        return None, None
    
    df = pd.DataFrame(rep_data)
    csv = df.to_csv(index=False)
    filename = f"squat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return csv, filename
