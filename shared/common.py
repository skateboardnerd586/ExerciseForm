"""
Shared Common Utilities Module
Contains helper functions, CSS styling, and common utilities used across exercises
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

def calculate_summary_metrics(rep_data, exercise_type="overhead_squat"):
    """Calculate summary metrics from rep data"""
    if not rep_data:
        return {}
    
    if exercise_type == "overhead_squat":
        return calculate_overhead_squat_summary(rep_data)
    elif exercise_type == "deadlift":
        return calculate_deadlift_summary(rep_data)
    else:
        # Default fallback
        avg_left_knee = np.mean([rep['left_knee_angle'] for rep in rep_data])
        avg_right_knee = np.mean([rep['right_knee_angle'] for rep in rep_data])
        avg_duration = np.mean([rep['duration_seconds'] for rep in rep_data])
        
        return {
            'total_reps': len(rep_data),
            'avg_knee_angle': (avg_left_knee + avg_right_knee) / 2,
            'avg_duration': avg_duration
        }

def calculate_overhead_squat_summary(rep_data):
    """Calculate summary metrics for overhead squat"""
    if not rep_data:
        return {}
    
    # Calculate averages for all metrics
    avg_left_knee = np.mean([rep['left_knee_angle'] for rep in rep_data])
    avg_right_knee = np.mean([rep['right_knee_angle'] for rep in rep_data])
    
    # Handle duration calculation safely (some reps might not have duration yet)
    durations = [rep.get('duration_seconds') for rep in rep_data if rep.get('duration_seconds') is not None]
    avg_duration = np.mean(durations) if durations else 0
    
    # Calculate ROM averages
    knee_roms = []
    hip_roms = []
    shoulder_roms = []
    
    for rep in rep_data:
        if rep.get('left_knee_rom') is not None and rep.get('right_knee_rom') is not None:
            knee_roms.append((rep['left_knee_rom'] + rep['right_knee_rom']) / 2)
        if rep.get('left_hip_rom') is not None and rep.get('right_hip_rom') is not None:
            hip_roms.append((rep['left_hip_rom'] + rep['right_hip_rom']) / 2)
        if rep.get('left_shoulder_rom') is not None and rep.get('right_shoulder_rom') is not None:
            shoulder_roms.append((rep['left_shoulder_rom'] + rep['right_shoulder_rom']) / 2)
    
    summary = {
        'total_reps': len(rep_data),
        'avg_knee_angle': (avg_left_knee + avg_right_knee) / 2,
        'avg_duration': avg_duration
    }
    
    if knee_roms:
        summary['avg_knee_rom'] = np.mean(knee_roms)
    if hip_roms:
        summary['avg_hip_rom'] = np.mean(hip_roms)
    if shoulder_roms:
        summary['avg_shoulder_rom'] = np.mean(shoulder_roms)
    
    return summary

def calculate_deadlift_summary(rep_data):
    """Calculate summary metrics for deadlift"""
    if not rep_data:
        return {}
    
    # Calculate averages for all metrics
    avg_left_hip = np.mean([rep['left_hip_angle'] for rep in rep_data])
    avg_right_hip = np.mean([rep['right_hip_angle'] for rep in rep_data])
    
    # Handle duration calculation safely (some reps might not have duration yet)
    durations = [rep.get('duration_seconds') for rep in rep_data if rep.get('duration_seconds') is not None]
    avg_duration = np.mean(durations) if durations else 0
    
    # Calculate ROM averages
    hip_roms = []
    knee_roms = []
    
    for rep in rep_data:
        if rep.get('left_hip_rom') is not None and rep.get('right_hip_rom') is not None:
            hip_roms.append((rep['left_hip_rom'] + rep['right_hip_rom']) / 2)
        if rep.get('left_knee_rom') is not None and rep.get('right_knee_rom') is not None:
            knee_roms.append((rep['left_knee_rom'] + rep['right_knee_rom']) / 2)
    
    summary = {
        'total_reps': len(rep_data),
        'avg_hip_angle': (avg_left_hip + avg_right_hip) / 2,
        'avg_duration': avg_duration
    }
    
    if hip_roms:
        summary['avg_hip_rom'] = np.mean(hip_roms)
    if knee_roms:
        summary['avg_knee_rom'] = np.mean(knee_roms)
    
    return summary

def create_csv_download(rep_data, exercise_type="exercise"):
    """Create CSV data for download"""
    if not rep_data:
        return None, None
    
    df = pd.DataFrame(rep_data)
    csv = df.to_csv(index=False)
    filename = f"{exercise_type}_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    
    return csv, filename
