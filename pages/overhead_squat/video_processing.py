"""
Overhead Squat Video Processing Module
Handles overhead squat specific video processing and analysis
"""

import cv2
import tempfile
import os
import streamlit as st
from datetime import datetime
from ultralytics import YOLO
from shared.pose_analysis import detect_pose_yolo, draw_pose, calculate_knee_angles, calculate_hip_angles
from shared.video_processing import setup_video_processing, cleanup_video_processing, rotate_frame
from .analysis import calculate_shoulder_angles

def process_overhead_squat_video(video_file, squat_down_threshold=130, squat_up_threshold=150, manual_rotation=0):
    """Process uploaded video and extract overhead squat data"""
    
    # Setup video processing
    cap, fps, width, height, total_rotation, tmp_path = setup_video_processing(video_file, manual_rotation)
    
    if cap is None:
        return [], None
    
    try:
        # Load YOLOv11 model
        with st.spinner("Loading YOLOv11 Pose model..."):
            model = YOLO('yolo11n-pose.pt')
        
        # Setup video writer for output
        output_path = f"overhead_squat_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
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
        
        # ROM tracking variables - track max standing angles between reps
        max_left_knee_angle_standing = None
        max_right_knee_angle_standing = None
        max_left_hip_angle_standing = None
        max_right_hip_angle_standing = None
        max_left_shoulder_angle_standing = None
        max_right_shoulder_angle_standing = None
        
        # Duration tracking - track when highest knee angle was reached before squat
        highest_knee_frame_before_squat = None
        
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
                    
                    # Track max standing angles for ROM calculation and duration tracking
                    if phase == "standing":
                        # Update max standing angles (highest point while standing)
                        if max_left_knee_angle_standing is None or left_knee_angle > max_left_knee_angle_standing:
                            max_left_knee_angle_standing = left_knee_angle
                        if max_right_knee_angle_standing is None or right_knee_angle > max_right_knee_angle_standing:
                            max_right_knee_angle_standing = right_knee_angle
                        
                        # Track when the overall highest knee angle (max of left and right) was reached
                        current_max_knee = max(left_knee_angle, right_knee_angle)
                        previous_max_knee = max(max_left_knee_angle_standing or 0, max_right_knee_angle_standing or 0)
                        if highest_knee_frame_before_squat is None or current_max_knee > previous_max_knee:
                            highest_knee_frame_before_squat = frame_count
                        
                        if left_hip_angle is not None:
                            if max_left_hip_angle_standing is None or left_hip_angle > max_left_hip_angle_standing:
                                max_left_hip_angle_standing = left_hip_angle
                        if right_hip_angle is not None:
                            if max_right_hip_angle_standing is None or right_hip_angle > max_right_hip_angle_standing:
                                max_right_hip_angle_standing = right_hip_angle
                        
                        if left_shoulder_angle is not None:
                            if max_left_shoulder_angle_standing is None or left_shoulder_angle > max_left_shoulder_angle_standing:
                                max_left_shoulder_angle_standing = left_shoulder_angle
                        if right_shoulder_angle is not None:
                            if max_right_shoulder_angle_standing is None or right_shoulder_angle > max_right_shoulder_angle_standing:
                                max_right_shoulder_angle_standing = right_shoulder_angle
                    
                    if phase == "standing" and min_knee_angle < squat_down_threshold:
                        # Calculate ROM for previous rep if we have data
                        if rep_history:
                            # Calculate ROM = max_standing_angle - lowest_angle_of_previous_rep
                            last_rep_index = len(rep_history) - 1
                            last_rep = rep_history[last_rep_index]
                            
                            left_knee_rom = None
                            right_knee_rom = None
                            left_hip_rom = None
                            right_hip_rom = None
                            left_shoulder_rom = None
                            right_shoulder_rom = None
                            
                            if max_left_knee_angle_standing is not None and last_rep.get('left_knee_angle') is not None:
                                left_knee_rom = max_left_knee_angle_standing - last_rep['left_knee_angle']
                            if max_right_knee_angle_standing is not None and last_rep.get('right_knee_angle') is not None:
                                right_knee_rom = max_right_knee_angle_standing - last_rep['right_knee_angle']
                            
                            if max_left_hip_angle_standing is not None and last_rep.get('left_hip_angle') is not None:
                                left_hip_rom = max_left_hip_angle_standing - last_rep['left_hip_angle']
                            if max_right_hip_angle_standing is not None and last_rep.get('right_hip_angle') is not None:
                                right_hip_rom = max_right_hip_angle_standing - last_rep['right_hip_angle']
                            
                            if max_left_shoulder_angle_standing is not None and last_rep.get('left_shoulder_angle') is not None:
                                left_shoulder_rom = max_left_shoulder_angle_standing - last_rep['left_shoulder_angle']
                            if max_right_shoulder_angle_standing is not None and last_rep.get('right_shoulder_angle') is not None:
                                right_shoulder_rom = max_right_shoulder_angle_standing - last_rep['right_shoulder_angle']
                            
                            # Add ROM data to the last rep
                            rep_history[last_rep_index].update({
                                'left_knee_rom': left_knee_rom,
                                'right_knee_rom': right_knee_rom,
                                'left_hip_rom': left_hip_rom,
                                'right_hip_rom': right_hip_rom,
                                'left_shoulder_rom': left_shoulder_rom,
                                'right_shoulder_rom': right_shoulder_rom
                            })
                            
                            # Calculate duration for previous rep (from lowest knee angle to highest knee angle)
                            if highest_knee_frame_before_squat is not None:
                                duration_frames = frame_count - highest_knee_frame_before_squat
                                duration_seconds = duration_frames / fps
                                rep_history[last_rep_index]['duration_seconds'] = duration_seconds
                        
                        phase = "squatting"
                        lowest_left_knee_angle_this_rep = left_knee_angle
                        lowest_right_knee_angle_this_rep = right_knee_angle
                        lowest_left_hip_angle_this_rep = left_hip_angle if left_hip_angle is not None else None
                        lowest_right_hip_angle_this_rep = right_hip_angle if right_hip_angle is not None else None
                        lowest_left_shoulder_angle_this_rep = left_shoulder_angle if left_shoulder_angle is not None else None
                        lowest_right_shoulder_angle_this_rep = right_shoulder_angle if right_shoulder_angle is not None else None
                        squat_start_frame = frame_count
                        
                        # Reset standing max angles for next rep
                        max_left_knee_angle_standing = None
                        max_right_knee_angle_standing = None
                        max_left_hip_angle_standing = None
                        max_right_hip_angle_standing = None
                        max_left_shoulder_angle_standing = None
                        max_right_shoulder_angle_standing = None
                        highest_knee_frame_before_squat = None
                    
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
                            
                            rep_data = {
                                'rep_number': rep_count,
                                'left_knee_angle': lowest_left_knee_angle_this_rep,
                                'right_knee_angle': lowest_right_knee_angle_this_rep,
                                'left_hip_angle': lowest_left_hip_angle_this_rep,
                                'right_hip_angle': lowest_right_hip_angle_this_rep,
                                'left_shoulder_angle': lowest_left_shoulder_angle_this_rep,
                                'right_shoulder_angle': lowest_right_shoulder_angle_this_rep,
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
                            
                            # Reset standing max angles for next rep
                            max_left_knee_angle_standing = None
                            max_right_knee_angle_standing = None
                            max_left_hip_angle_standing = None
                            max_right_hip_angle_standing = None
                            max_left_shoulder_angle_standing = None
                            max_right_shoulder_angle_standing = None
                
                # Draw angle info on frame
                if left_knee_angle is not None:
                    cv2.putText(frame, f'Left Knee: {left_knee_angle:.1f} deg', 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                if right_knee_angle is not None:
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
                
                # Show current rep count
                cv2.putText(frame, f'Overhead Squats: {rep_count}', (50, 230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                
                # Draw rep history on right side with hip angles and better spacing
                right_x = width - 350  # Moved further left to accommodate more text
                y_start = 50
                cv2.putText(frame, "Rep History:", (right_x, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                for i, rep_data in enumerate(rep_history[-3:]):  # Show last 3 reps to accommodate ROM data
                    y_pos = y_start + 30 + (i * 280)  # Increased spacing to 280 pixels between reps for ROM data and shoulder angles
                    cv2.putText(frame, f"Rep {rep_data['rep_number']}:", (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show minimum angles (bottom of squat)
                    if rep_data['left_knee_angle'] is not None:
                        cv2.putText(frame, f"  L-Knee: {rep_data['left_knee_angle']:.1f} deg", (right_x, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    if rep_data['right_knee_angle'] is not None:
                        cv2.putText(frame, f"  R-Knee: {rep_data['right_knee_angle']:.1f} deg", (right_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add ROM data if available
                    if rep_data.get('left_knee_rom') is not None and rep_data.get('right_knee_rom') is not None:
                        cv2.putText(frame, f"  L-Knee ROM: {rep_data['left_knee_rom']:.1f} deg", (right_x, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"  R-Knee ROM: {rep_data['right_knee_rom']:.1f} deg", (right_x, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Add hip angles and ROM if available
                    if rep_data['left_hip_angle'] is not None and rep_data['right_hip_angle'] is not None:
                        cv2.putText(frame, f"  L-Hip: {rep_data['left_hip_angle']:.1f} deg", (right_x, y_pos + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"  R-Hip: {rep_data['right_hip_angle']:.1f} deg", (right_x, y_pos + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        if rep_data.get('left_hip_rom') is not None and rep_data.get('right_hip_rom') is not None:
                            cv2.putText(frame, f"  L-Hip ROM: {rep_data['left_hip_rom']:.1f} deg", (right_x, y_pos + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                            cv2.putText(frame, f"  R-Hip ROM: {rep_data['right_hip_rom']:.1f} deg", (right_x, y_pos + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        
                        # Add shoulder angles and ROM if available
                        if rep_data['left_shoulder_angle'] is not None and rep_data['right_shoulder_angle'] is not None:
                            cv2.putText(frame, f"  L-Shoulder: {rep_data['left_shoulder_angle']:.1f} deg", (right_x, y_pos + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            cv2.putText(frame, f"  R-Shoulder: {rep_data['right_shoulder_angle']:.1f} deg", (right_x, y_pos + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
                            if rep_data.get('left_shoulder_rom') is not None and rep_data.get('right_shoulder_rom') is not None:
                                cv2.putText(frame, f"  L-Shoulder ROM: {rep_data['left_shoulder_rom']:.1f} deg", (right_x, y_pos + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                                cv2.putText(frame, f"  R-Shoulder ROM: {rep_data['right_shoulder_rom']:.1f} deg", (right_x, y_pos + 240), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                            
                            duration = rep_data.get('duration_seconds')
                            if duration is not None:
                                cv2.putText(frame, f"  Duration: {duration:.1f}s", (right_x, y_pos + 260), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Write frame to output video
            out.write(frame)
        
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()
        
        return rep_history, output_path
        
    finally:
        # Clean up temporary file
        cleanup_video_processing(tmp_path)
