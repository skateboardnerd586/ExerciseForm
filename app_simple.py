import streamlit as st
import cv2
import numpy as np
import math
import subprocess
import sys
import tempfile
import os
from ultralytics import YOLO
import json
import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
from datetime import datetime
import openai
from io import BytesIO
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Overhead Squat Assessment AI",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
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
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rep_data' not in st.session_state:
    st.session_state.rep_data = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False
if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
if 'output_video_path' not in st.session_state:
    st.session_state.output_video_path = None

# YOLOv11 pose keypoint connections
POSE_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),  # nose-eyes-ears
    (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # upper body
    (5, 11), (6, 12), (11, 12),  # torso
    (11, 13), (12, 14), (13, 15), (14, 16),  # lower body
]

def angle_between(v1, v2):
    """Calculate angle between two vectors in degrees"""
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
    if mag1 * mag2 == 0:
        return 0
    return math.degrees(math.acos(dot / (mag1 * mag2)))

def calculate_knee_angles(keypoints):
    """Calculate knee angles for both legs"""
    LEFT_HIP = 11
    LEFT_KNEE = 13
    LEFT_ANKLE = 15
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    RIGHT_ANKLE = 16
    
    left_hip = keypoints[LEFT_HIP]
    left_knee = keypoints[LEFT_KNEE]
    left_ankle = keypoints[LEFT_ANKLE]
    right_hip = keypoints[RIGHT_HIP]
    right_knee = keypoints[RIGHT_KNEE]
    right_ankle = keypoints[RIGHT_ANKLE]
    
    left_thigh = (left_hip[0] - left_knee[0], left_hip[1] - left_knee[1])
    left_shin = (left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1])
    left_angle = angle_between(left_thigh, left_shin)
    
    right_thigh = (right_hip[0] - right_knee[0], right_hip[1] - right_knee[1])
    right_shin = (right_ankle[0] - right_knee[0], right_ankle[1] - right_knee[1])
    right_angle = angle_between(right_thigh, right_shin)
    
    return left_angle, right_angle

def calculate_hip_angles(keypoints):
    """Calculate hip angles for both legs"""
    LEFT_SHOULDER = 5
    LEFT_HIP = 11
    LEFT_KNEE = 13
    RIGHT_SHOULDER = 6
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    
    left_shoulder = keypoints[LEFT_SHOULDER]
    left_hip = keypoints[LEFT_HIP]
    left_knee = keypoints[LEFT_KNEE]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    right_hip = keypoints[RIGHT_HIP]
    right_knee = keypoints[RIGHT_KNEE]
    
    left_torso = (left_shoulder[0] - left_hip[0], left_shoulder[1] - left_hip[1])
    left_thigh = (left_knee[0] - left_hip[0], left_knee[1] - left_hip[1])
    left_angle = angle_between(left_torso, left_thigh)
    
    right_torso = (right_shoulder[0] - right_hip[0], right_shoulder[1] - right_hip[1])
    right_thigh = (right_knee[0] - right_hip[0], right_knee[1] - right_hip[1])
    right_angle = angle_between(right_torso, right_thigh)
    
    return left_angle, right_angle

def calculate_shoulder_angles(keypoints):
    """Calculate shoulder angles for overhead squat assessment"""
    LEFT_SHOULDER = 5
    LEFT_ELBOW = 7
    LEFT_HIP = 11
    RIGHT_SHOULDER = 6
    RIGHT_ELBOW = 8
    RIGHT_HIP = 12
    
    left_shoulder = keypoints[LEFT_SHOULDER]
    left_elbow = keypoints[LEFT_ELBOW]
    left_hip = keypoints[LEFT_HIP]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    right_elbow = keypoints[RIGHT_ELBOW]
    right_hip = keypoints[RIGHT_HIP]
    
    # Left shoulder-to-elbow angle (shoulder mobility)
    left_shoulder_elbow = (left_elbow[0] - left_shoulder[0], left_elbow[1] - left_shoulder[1])
    left_shoulder_hip = (left_hip[0] - left_shoulder[0], left_hip[1] - left_shoulder[1])
    left_shoulder_elbow_angle = angle_between(left_shoulder_elbow, left_shoulder_hip)
    
    # Right shoulder-to-elbow angle (shoulder mobility)
    right_shoulder_elbow = (right_elbow[0] - right_shoulder[0], right_elbow[1] - right_shoulder[1])
    right_shoulder_hip = (right_hip[0] - right_shoulder[0], right_hip[1] - right_shoulder[1])
    right_shoulder_elbow_angle = angle_between(right_shoulder_elbow, right_shoulder_hip)
    
    return left_shoulder_elbow_angle, right_shoulder_elbow_angle

def detect_pose_yolo(model, frame):
    """Detect pose using YOLOv11 Pose"""
    try:
        results = model(frame, verbose=False)
        
        if results and len(results) > 0:
            result = results[0]
            
            if result.keypoints is not None and len(result.keypoints.xy) > 0:
                keypoints = result.keypoints.xy[0].cpu().numpy()
                confidences = result.keypoints.conf[0].cpu().numpy()
                
                keypoints_formatted = []
                for i, (kp, conf) in enumerate(zip(keypoints, confidences)):
                    x, y = kp
                    keypoints_formatted.append((float(x), float(y), float(conf)))
                
                return keypoints_formatted
        
        return None
    
    except Exception as e:
        st.error(f"YOLOv11 Pose inference error: {e}")
        return None

def draw_pose(frame, keypoints):
    """Draw pose on frame using correct YOLOv11 connections"""
    # Draw keypoints
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > 0.3:  # Only draw confident keypoints
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i), (int(x), int(y) - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
    
    # Draw connections using YOLOv11 pose connections
    for connection in POSE_CONNECTIONS:
        start_idx, end_idx = connection
        if start_idx < len(keypoints) and end_idx < len(keypoints):
            start_point = keypoints[start_idx]
            end_point = keypoints[end_idx]
            
            if start_point[2] > 0.3 and end_point[2] > 0.3:
                cv2.line(frame, 
                        (int(start_point[0]), int(start_point[1])),
                        (int(end_point[0]), int(end_point[1])),
                        (255, 0, 0), 2)

def process_video(video_file, squat_down_threshold=130, squat_up_threshold=150):
    """Process uploaded video and extract squat data"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load YOLOv11 model
        with st.spinner("Loading YOLOv11 Pose model..."):
            model = YOLO('yolo11n-pose.pt')
        
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            st.error("Could not open video file")
            return [], None
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Setup video writer for output
        output_path = f"squat_tracker_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize tracking variables
        rep_count = 0
        phase = "standing"
        frame_count = 0
        lowest_left_knee_angle_this_rep = None
        lowest_right_knee_angle_this_rep = None
        lowest_left_hip_angle_this_rep = None
        lowest_right_hip_angle_this_rep = None
        lowest_left_shoulder_angle_this_rep = None
        lowest_right_shoulder_angle_this_rep = None
        rep_history = []
        squat_start_frame = None
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Update progress
            progress = frame_count / total_frames
            progress_bar.progress(progress)
            status_text.text(f"Processing frame {frame_count}/{total_frames}")
            
            # Detect pose
            keypoints = detect_pose_yolo(model, frame)
            
            if keypoints:
                # Draw pose
                draw_pose(frame, keypoints)
                
                left_knee_angle, right_knee_angle = calculate_knee_angles(keypoints)
                left_hip_angle, right_hip_angle = calculate_hip_angles(keypoints)
                left_shoulder_angle, right_shoulder_angle = calculate_shoulder_angles(keypoints)
                
                if left_knee_angle is not None and right_knee_angle is not None:
                    min_knee_angle = min(left_knee_angle, right_knee_angle)
                    
                    if phase == "standing" and min_knee_angle < squat_down_threshold:
                        phase = "squatting"
                        lowest_left_knee_angle_this_rep = left_knee_angle
                        lowest_right_knee_angle_this_rep = right_knee_angle
                        lowest_left_hip_angle_this_rep = left_hip_angle if left_hip_angle is not None else None
                        lowest_right_hip_angle_this_rep = right_hip_angle if right_hip_angle is not None else None
                        lowest_left_shoulder_angle_this_rep = left_shoulder_angle if left_shoulder_angle is not None else None
                        lowest_right_shoulder_angle_this_rep = right_shoulder_angle if right_shoulder_angle is not None else None
                        squat_start_frame = frame_count
                    
                    elif phase == "squatting":
                        # Update lowest angles
                        if left_knee_angle < lowest_left_knee_angle_this_rep:
                            lowest_left_knee_angle_this_rep = left_knee_angle
                        if right_knee_angle < lowest_right_knee_angle_this_rep:
                            lowest_right_knee_angle_this_rep = right_knee_angle
                        
                        if left_hip_angle is not None and (lowest_left_hip_angle_this_rep is None or left_hip_angle < lowest_left_hip_angle_this_rep):
                            lowest_left_hip_angle_this_rep = left_hip_angle
                        if right_hip_angle is not None and (lowest_right_hip_angle_this_rep is None or right_hip_angle < lowest_right_hip_angle_this_rep):
                            lowest_right_hip_angle_this_rep = right_hip_angle
                        
                        # Update shoulder angles at the bottom of squat (when knee angles are smallest)
                        if left_shoulder_angle is not None and (lowest_left_shoulder_angle_this_rep is None or left_knee_angle < lowest_left_knee_angle_this_rep):
                            lowest_left_shoulder_angle_this_rep = left_shoulder_angle
                        if right_shoulder_angle is not None and (lowest_right_shoulder_angle_this_rep is None or right_knee_angle < lowest_right_knee_angle_this_rep):
                            lowest_right_shoulder_angle_this_rep = right_shoulder_angle
                        
                        # Check if back to standing
                        if min_knee_angle >= squat_up_threshold:
                            phase = "standing"
                            rep_count += 1
                            squat_duration_frames = frame_count - squat_start_frame
                            squat_duration_seconds = squat_duration_frames / fps
                            
                            rep_data = {
                                'rep_number': rep_count,
                                'left_knee_angle': lowest_left_knee_angle_this_rep,
                                'right_knee_angle': lowest_right_knee_angle_this_rep,
                                'left_hip_angle': lowest_left_hip_angle_this_rep,
                                'right_hip_angle': lowest_right_hip_angle_this_rep,
                                'left_shoulder_angle': lowest_left_shoulder_angle_this_rep,
                                'right_shoulder_angle': lowest_right_shoulder_angle_this_rep,
                                'duration_seconds': squat_duration_seconds,
                                'timestamp': frame_count / fps
                            }
                            rep_history.append(rep_data)
                            
                            # Reset for next rep
                            lowest_left_knee_angle_this_rep = None
                            lowest_right_knee_angle_this_rep = None
                            lowest_left_hip_angle_this_rep = None
                            lowest_right_hip_angle_this_rep = None
                            lowest_left_shoulder_angle_this_rep = None
                            lowest_right_shoulder_angle_this_rep = None
                            squat_start_frame = None
                
                # Draw angle info on frame
                cv2.putText(frame, f'Left Knee: {left_knee_angle:.1f} deg', 
                           (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.putText(frame, f'Right Knee: {right_knee_angle:.1f} deg', 
                           (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                
                if left_hip_angle is not None and right_hip_angle is not None:
                    cv2.putText(frame, f'Left Hip: {left_hip_angle:.1f} deg', 
                               (50, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                    cv2.putText(frame, f'Right Hip: {right_hip_angle:.1f} deg', 
                               (50, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                
                # Show shoulder angles
                if left_shoulder_angle is not None and right_shoulder_angle is not None:
                    cv2.putText(frame, f'Left Shoulder: {left_shoulder_angle:.1f} deg', 
                               (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                    cv2.putText(frame, f'Right Shoulder: {right_shoulder_angle:.1f} deg', 
                               (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                
                # Show current phase and rep count
                cv2.putText(frame, f'Squats: {rep_count}', (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                cv2.putText(frame, f'Phase: {phase}', (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                # Draw rep history on right side with hip angles and better spacing
                right_x = width - 350  # Moved further left to accommodate more text
                y_start = 50
                cv2.putText(frame, "Rep History:", (right_x, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                for i, rep_data in enumerate(rep_history[-5:]):  # Show last 5 reps
                    y_pos = y_start + 30 + (i * 160)  # Increased spacing to 160 pixels between reps for shoulder angles
                    cv2.putText(frame, f"Rep {rep_data['rep_number']}:", (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    cv2.putText(frame, f"  L-Knee: {rep_data['left_knee_angle']:.1f} deg", (right_x, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(frame, f"  R-Knee: {rep_data['right_knee_angle']:.1f} deg", (right_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add hip angles if available
                    if rep_data['left_hip_angle'] is not None and rep_data['right_hip_angle'] is not None:
                        cv2.putText(frame, f"  L-Hip: {rep_data['left_hip_angle']:.1f} deg", (right_x, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"  R-Hip: {rep_data['right_hip_angle']:.1f} deg", (right_x, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Add shoulder angles if available
                        if rep_data['left_shoulder_angle'] is not None and rep_data['right_shoulder_angle'] is not None:
                            cv2.putText(frame, f"  L-Shoulder: {rep_data['left_shoulder_angle']:.1f} deg", (right_x, y_pos + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            cv2.putText(frame, f"  R-Shoulder: {rep_data['right_shoulder_angle']:.1f} deg", (right_x, y_pos + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            cv2.putText(frame, f"  Duration: {rep_data['duration_seconds']:.1f}s", (right_x, y_pos + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        else:
                            cv2.putText(frame, f"  Duration: {rep_data['duration_seconds']:.1f}s", (right_x, y_pos + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    else:
                        cv2.putText(frame, f"  Duration: {rep_data['duration_seconds']:.1f}s", (right_x, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame to output video
            out.write(frame)
        
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        return rep_history, output_path
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)

def analyze_with_openai(rep_data, api_key):
    """Analyze squat data using OpenAI LLM"""
    if not api_key:
        return "Please provide your OpenAI API key in the sidebar to get AI analysis."
    
    try:
        # Initialize OpenAI client with new API
        client = openai.OpenAI(api_key=api_key)
        
        # Prepare data summary
        total_reps = len(rep_data)
        avg_left_knee = np.mean([rep['left_knee_angle'] for rep in rep_data])
        avg_right_knee = np.mean([rep['right_knee_angle'] for rep in rep_data])
        avg_left_hip = np.mean([rep['left_hip_angle'] for rep in rep_data if rep['left_hip_angle'] is not None])
        avg_right_hip = np.mean([rep['right_hip_angle'] for rep in rep_data if rep['right_hip_angle'] is not None])
        avg_left_shoulder = np.mean([rep['left_shoulder_angle'] for rep in rep_data if rep['left_shoulder_angle'] is not None])
        avg_right_shoulder = np.mean([rep['right_shoulder_angle'] for rep in rep_data if rep['right_shoulder_angle'] is not None])
        avg_duration = np.mean([rep['duration_seconds'] for rep in rep_data])
        
        data_summary = f"""
        Overhead Squat Assessment Data:
        - Total Reps: {total_reps}
        - Average Left Knee Angle: {avg_left_knee:.1f} deg (at bottom of squat)
        - Average Right Knee Angle: {avg_right_knee:.1f} deg (at bottom of squat)
        - Average Left Hip Angle: {avg_left_hip:.1f} deg (at bottom of squat)
        - Average Right Hip Angle: {avg_right_hip:.1f} deg (at bottom of squat)
        - Average Left Shoulder Angle: {avg_left_shoulder:.1f} deg (at bottom of squat)
        - Average Right Shoulder Angle: {avg_right_shoulder:.1f} deg (at bottom of squat)
        - Average Duration: {avg_duration:.1f} seconds
        
        Individual Rep Details:
        """
        
        for i, rep in enumerate(rep_data):
            data_summary += f"""
        Rep {rep['rep_number']}:
        - Left Knee: {rep['left_knee_angle']:.1f} deg (at bottom of squat)
        - Right Knee: {rep['right_knee_angle']:.1f} deg (at bottom of squat)
        - Left Hip: {rep['left_hip_angle']:.1f} deg (at bottom of squat, if available)
        - Right Hip: {rep['right_hip_angle']:.1f} deg (at bottom of squat, if available)
        - Left Shoulder: {rep['left_shoulder_angle']:.1f} deg (at bottom of squat, if available)
        - Right Shoulder: {rep['right_shoulder_angle']:.1f} deg (at bottom of squat, if available)
        - Duration: {rep['duration_seconds']:.1f}s
        """
        
        prompt = f"""
        As a professional fitness trainer and biomechanics expert, analyze this squat tracking data and provide concise insights. 
        
        IMPORTANT DISCLAIMERS:
        - This analysis is based on limited movement data and should not replace professional medical advice
        - I am not a doctor or licensed physical therapist
        - This assessment is for educational and training purposes only
        - Consult healthcare professionals for any pain, injury concerns, or medical conditions
        
        {data_summary}
        
        Please provide a concise analysis (under 300 words) covering:
        1. Overall squat form and depth assessment
        2. Bilateral symmetry analysis (left vs right leg)
        3. Key corrective exercises for identified limitations
        4. Training recommendations for improvement
        
        Focus on practical, actionable exercises and training strategies the person can implement safely.
        Please use second person to address the user and emphasize consulting professionals for any concerns.
        Keep your response under 300 words.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional fitness trainer and biomechanics expert with extensive experience in movement analysis and corrective exercise. You provide educational guidance but always recommend consulting healthcare professionals for medical concerns."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,  # Reduced to save tokens
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except openai.APIConnectionError as e:
        return f"Connection error with OpenAI API. Please check your internet connection and try again. Error: {str(e)}"
    except openai.APIError as e:
        return f"OpenAI API error. Please check your API key and try again. Error: {str(e)}"
    except openai.RateLimitError as e:
        return f"Rate limit exceeded. Please wait a moment and try again. Error: {str(e)}"
    except openai.APITimeoutError as e:
        return f"Request timed out. Please try again. Error: {str(e)}"
    except Exception as e:
        error_str = str(e)
        if "quota" in error_str.lower() or "insufficient" in error_str.lower():
            return f"⚠️ **Quota Exceeded**: You've reached your OpenAI usage limit. This is common on the free tier. You can:\n\n1. **Wait** for your monthly quota to reset\n2. **Upgrade** to a paid plan at https://platform.openai.com/account/billing\n3. **Use the app without AI analysis** - all other features work perfectly!\n\nError details: {error_str}"
        return f"Error analyzing with OpenAI: {str(e)}"

def create_matplotlib_charts(rep_data):
    """Create matplotlib charts instead of plotly"""
    if not rep_data:
        return None, None, None
    
    df = pd.DataFrame(rep_data)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Overhead Squat Assessment Charts', fontsize=16)
    
    # Knee angles over time
    axes[0, 0].plot(df['rep_number'], df['left_knee_angle'], 'b-o', label='Left Knee', linewidth=2)
    axes[0, 0].plot(df['rep_number'], df['right_knee_angle'], 'r-o', label='Right Knee', linewidth=2)
    axes[0, 0].set_title('Knee Angles by Rep')
    axes[0, 0].set_xlabel('Rep Number')
    axes[0, 0].set_ylabel('Angle (degrees)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Shoulder angles over time
    if df['left_shoulder_angle'].notna().any():
        axes[0, 1].plot(df['rep_number'], df['left_shoulder_angle'], 'g-o', label='Left Shoulder', linewidth=2)
    if df['right_shoulder_angle'].notna().any():
        axes[0, 1].plot(df['rep_number'], df['right_shoulder_angle'], 'm-o', label='Right Shoulder', linewidth=2)
    axes[0, 1].set_title('Shoulder Angles by Rep')
    axes[0, 1].set_xlabel('Rep Number')
    axes[0, 1].set_ylabel('Angle (degrees)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hip angles over time
    if df['left_hip_angle'].notna().any():
        axes[1, 0].plot(df['rep_number'], df['left_hip_angle'], 'g-o', label='Left Hip', linewidth=2)
    if df['right_hip_angle'].notna().any():
        axes[1, 0].plot(df['rep_number'], df['right_hip_angle'], 'm-o', label='Right Hip', linewidth=2)
    axes[1, 0].set_title('Hip Angles by Rep')
    axes[1, 0].set_xlabel('Rep Number')
    axes[1, 0].set_ylabel('Angle (degrees)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Duration over time
    axes[1, 1].plot(df['rep_number'], df['duration_seconds'], 'purple', marker='o', linewidth=2)
    axes[1, 1].set_title('Squat Duration by Rep')
    axes[1, 1].set_xlabel('Rep Number')
    axes[1, 1].set_ylabel('Duration (seconds)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main app
def main():
    st.markdown('<h1 class="main-header">🏋️ Overhead Squat Assessment AI</h1>', unsafe_allow_html=True)
    
    # Overhead Squat Assessment Information
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
        
        st.markdown("#### 🎯 What We're Assessing:")
        st.markdown("""
        - **Ankle Mobility:** Can you squat deep without heels lifting?
        - **Hip Mobility:** Can you achieve full depth with good form?
        - **Thoracic Spine:** Can you maintain overhead position?
        - **Shoulder Mobility:** Can you keep arms overhead throughout?
        - **Core Stability:** Can you maintain neutral spine?
        - **Balance:** Can you perform the movement symmetrically?
        """)
        
        st.markdown("💡 **Tip:** Perform 3-5 reps for best assessment results. The AI will analyze your movement patterns and provide corrective exercise recommendations.")
    
    # Important filming note
    st.markdown("""
    <div style="
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        text-align: center;
    ">
        <h3 style="color: #856404; margin: 0 0 10px 0;">📱 Important: Film Your Video Horizontally!</h3>
        <p style="color: #856404; margin: 0; font-size: 16px;">
            For best results, please film your overhead squat video in <strong>landscape/horizontal orientation</strong>. 
            The AI works best when the video is recorded with the camera held sideways (landscape mode).
            <br><br>
            <strong>📏 Make sure the entire body is visible in the camera view</strong> - from head to feet - 
            so the AI can properly track all key points for accurate movement analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("Settings")
    
    # OpenAI API Key
    openai_api_key = st.sidebar.text_input(
        "OpenAI API Key", 
        type="password",
        help="Enter your OpenAI API key for AI analysis"
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
                        squat_up_threshold
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
            col2_1, col2_2, col2_3 = st.columns(3)
            
            with col2_1:
                st.metric("Total Reps", len(rep_data))
            
            with col2_2:
                avg_knee = np.mean([rep['left_knee_angle'] for rep in rep_data] + 
                                 [rep['right_knee_angle'] for rep in rep_data])
                st.metric("Avg Knee Angle", f"{avg_knee:.1f} deg")
            
            with col2_3:
                avg_duration = np.mean([rep['duration_seconds'] for rep in rep_data])
                st.metric("Avg Duration", f"{avg_duration:.1f}s")
            
            # Data table
            st.subheader("📋 Rep Details")
            st.info("📊 **Note**: All angles shown are measured at the deepest point of each overhead squat rep (bottom of squat when knee angles are smallest). This is the most critical position for assessing movement quality and identifying limitations.")
            df = pd.DataFrame(rep_data)
            st.dataframe(df, use_container_width=True)
            
            # Download data
            csv = df.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"squat_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
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
