"""
Video Processing Module for Overhead Squat Assessment
Handles video upload, rotation, processing, and output generation
"""

import cv2
import tempfile
import os
import streamlit as st
from datetime import datetime
from ultralytics import YOLO
from pose_analysis import detect_pose_yolo, draw_pose, calculate_knee_angles, calculate_hip_angles, calculate_shoulder_angles

def get_video_orientation_from_metadata(video_path):
    """Detect video orientation from OpenCV metadata"""
    debug_info = []
    
    try:
        # Use OpenCV to read rotation metadata
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            rotation_prop = cap.get(cv2.CAP_PROP_ORIENTATION_META)
            debug_info.append(f"OpenCV rotation property: {rotation_prop}")
            
            if rotation_prop > 0:
                cap.release()
                return int(rotation_prop), debug_info
            
            cap.release()
        
        debug_info.append("No rotation metadata detected")
        return 0, debug_info
    except Exception as e:
        debug_info.append(f"Error in orientation detection: {str(e)}")
        return 0, debug_info

def get_video_dimensions(video_path):
    """Get video dimensions to help user decide on rotation"""
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return width, height
        return None, None
    except Exception:
        return None, None

def rotate_frame(frame, rotation_angle):
    """Rotate frame by specified angle"""
    if rotation_angle == 0:
        return frame
    elif rotation_angle == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif rotation_angle == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        return frame

def is_cloud_environment():
    """Detect if running in cloud environment (Streamlit Cloud, Heroku, etc.)"""
    import os
    # Check for common cloud environment indicators
    cloud_indicators = [
        'STREAMLIT_SHARING_MODE',  # Streamlit Cloud
        'DYNO',  # Heroku
        'PORT',  # Many cloud platforms
        'CLOUD',  # Generic cloud indicator
        'AWS_',  # AWS
        'GOOGLE_CLOUD',  # Google Cloud
        'AZURE_',  # Azure
    ]
    
    for indicator in cloud_indicators:
        if os.environ.get(indicator):
            return True
    
    # Check if running on Streamlit Cloud specifically
    if 'streamlit.app' in os.environ.get('STREAMLIT_SERVER_HEADLESS', ''):
        return True
        
    return False

def process_video(video_file, squat_down_threshold=130, squat_up_threshold=150, manual_rotation=0):
    """Process uploaded video and extract squat data"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load YOLOv11 model
        with st.spinner("Loading YOLOv11 Pose model..."):
            model = YOLO('yolo11n-pose.pt')
        
        # Detect video orientation from metadata
        with st.spinner("Detecting video orientation..."):
            detected_rotation, debug_info = get_video_orientation_from_metadata(tmp_path)
            width, height = get_video_dimensions(tmp_path)
            
            # Only apply rotation in cloud environments, never locally
            is_cloud = is_cloud_environment()
            if not is_cloud:
                # Force no rotation when running locally
                detected_rotation = 0
                manual_rotation = 0
                debug_info.append("💻 Local environment - no rotation applied")
            
            total_rotation = (detected_rotation + manual_rotation) % 360
        
        if width and height:
            st.info(f"📱 Video dimensions: {width}x{height} pixels")
        
        # Show debug information to help understand what was detected
        with st.expander("🔍 Orientation Detection Details", expanded=False):
            for info in debug_info:
                st.text(info)
        
        if detected_rotation > 0:
            st.success(f"🎯 **Auto-detected rotation: {detected_rotation}°** - Video will be automatically corrected!")
        elif manual_rotation > 0:
            st.info(f"🔄 Manual rotation applied: {manual_rotation}°")
        else:
            st.warning("⚠️ **Auto-detection failed** - No orientation metadata found in your video.")
            st.info("💡 **Solution**: Use the 'Manual Rotation Override' option in the sidebar. Most videos that appear upside down need **180° rotation**.")
            
            if height and width and height > width:
                st.info("📱 Your video is taller than wide - if it appears sideways, try **90° or 270°** rotation.")
        
        # Open video
        cap = cv2.VideoCapture(tmp_path)
        
        if not cap.isOpened():
            st.error("Could not open video file")
            return [], None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Adjust dimensions if video will be rotated
        if total_rotation in [90, 270]:
            width, height = height, width
        
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
            
            # Apply rotation correction
            if total_rotation > 0:
                frame = rotate_frame(frame, total_rotation)
            
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
