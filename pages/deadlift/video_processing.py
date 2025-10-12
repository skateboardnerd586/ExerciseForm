"""
Deadlift Video Processing Module
Handles deadlift specific video processing and analysis
"""

import cv2
import tempfile
import os
import streamlit as st
from datetime import datetime
from ultralytics import YOLO
from shared.pose_analysis import detect_pose_yolo, draw_pose
from shared.video_processing import setup_video_processing, cleanup_video_processing, rotate_frame
from .analysis import calculate_hip_angles_deadlift, calculate_knee_angles_deadlift

def process_deadlift_video(video_file, deadlift_down_threshold=140, deadlift_up_threshold=160, manual_rotation=0):
    """Process uploaded video and extract deadlift data"""
    
    # Setup video processing
    cap, fps, width, height, total_rotation, tmp_path = setup_video_processing(video_file, manual_rotation)
    
    if cap is None:
        return [], None
    
    try:
        # Load YOLOv11 model
        with st.spinner("Loading YOLOv11 Pose model..."):
            model = YOLO('yolo11n-pose.pt')
        
        # Setup video writer for output
        output_path = f"deadlift_output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize tracking variables
        rep_count = 0
        phase = "standing"
        frame_count = 0
        lowest_left_hip_angle_this_rep = None
        lowest_right_hip_angle_this_rep = None
        lowest_left_knee_angle_this_rep = None
        lowest_right_knee_angle_this_rep = None
        
        # ROM tracking variables - track max standing angles between reps
        max_left_hip_angle_standing = None
        max_right_hip_angle_standing = None
        max_left_knee_angle_standing = None
        max_right_knee_angle_standing = None
        
        # Duration tracking - simple logic
        lowest_hip_frame_current_rep = None
        highest_hip_frame_current_rep = None
        first_rep_started = False
        first_rep_completed = False
        
        rep_history = []
        deadlift_start_frame = None
        
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
                
                left_knee_angle, right_knee_angle = calculate_knee_angles_deadlift(keypoints)
                left_hip_angle, right_hip_angle = calculate_hip_angles_deadlift(keypoints)
                
                if left_hip_angle is not None and right_hip_angle is not None:
                    min_hip_angle = min(left_hip_angle, right_hip_angle)
                    
                    # Track max standing angles for ROM calculation
                    if phase == "standing":
                        # Update max standing angles (highest point while standing)
                        if max_left_hip_angle_standing is None or left_hip_angle > max_left_hip_angle_standing:
                            max_left_hip_angle_standing = left_hip_angle
                        if max_right_hip_angle_standing is None or right_hip_angle > max_right_hip_angle_standing:
                            max_right_hip_angle_standing = right_hip_angle
                        
                        # Track when the overall highest hip angle (max of left and right) was reached
                        # Only start tracking after first rep is completed (ignore initial standing position)
                        if first_rep_completed:
                            current_max_hip = max(left_hip_angle, right_hip_angle)
                            previous_max_hip = max(max_left_hip_angle_standing or 0, max_right_hip_angle_standing or 0)
                            if highest_hip_frame_current_rep is None or current_max_hip > previous_max_hip:
                                highest_hip_frame_current_rep = frame_count
                        
                        if left_knee_angle is not None:
                            if max_left_knee_angle_standing is None or left_knee_angle > max_left_knee_angle_standing:
                                max_left_knee_angle_standing = left_knee_angle
                        if right_knee_angle is not None:
                            if max_right_knee_angle_standing is None or right_knee_angle > max_right_knee_angle_standing:
                                max_right_knee_angle_standing = right_knee_angle
                        
                        # Check if we should start tracking the first rep (if video starts at bottom)
                        if not first_rep_started and min_hip_angle < deadlift_down_threshold:
                            # Video started at bottom, begin tracking first rep
                            first_rep_started = True
                            lowest_left_hip_angle_this_rep = left_hip_angle
                            lowest_right_hip_angle_this_rep = right_hip_angle
                            lowest_left_knee_angle_this_rep = left_knee_angle if left_knee_angle is not None else None
                            lowest_right_knee_angle_this_rep = right_knee_angle if right_knee_angle is not None else None
                            lowest_hip_frame_current_rep = frame_count
                            phase = "lowering"
                    
                    if phase == "standing" and min_hip_angle < deadlift_down_threshold:
                        # Calculate ROM and duration for previous rep if we have data
                        if rep_history:
                            # Calculate ROM = max_standing_angle - lowest_angle_of_previous_rep
                            last_rep_index = len(rep_history) - 1
                            last_rep = rep_history[last_rep_index]
                            
                            # Only add ROM if it hasn't been added yet
                            if last_rep.get('left_hip_rom') is None:
                                left_hip_rom = None
                                right_hip_rom = None
                                left_knee_rom = None
                                right_knee_rom = None
                                
                                if max_left_hip_angle_standing is not None and last_rep.get('left_hip_angle') is not None:
                                    left_hip_rom = max_left_hip_angle_standing - last_rep['left_hip_angle']
                                if max_right_hip_angle_standing is not None and last_rep.get('right_hip_angle') is not None:
                                    right_hip_rom = max_right_hip_angle_standing - last_rep['right_hip_angle']
                                
                                if max_left_knee_angle_standing is not None and last_rep.get('left_knee_angle') is not None:
                                    left_knee_rom = max_left_knee_angle_standing - last_rep['left_knee_angle']
                                if max_right_knee_angle_standing is not None and last_rep.get('right_knee_angle') is not None:
                                    right_knee_rom = max_right_knee_angle_standing - last_rep['right_knee_angle']
                                
                                # Add ROM data to the last rep
                                rep_history[last_rep_index].update({
                                    'left_hip_rom': left_hip_rom,
                                    'right_hip_rom': right_hip_rom,
                                    'left_knee_rom': left_knee_rom,
                                    'right_knee_rom': right_knee_rom
                                })
                            
                            # Calculate duration for previous rep (from lowest hip angle to highest hip angle)
                            if lowest_hip_frame_current_rep is not None and highest_hip_frame_current_rep is not None:
                                duration_frames = highest_hip_frame_current_rep - lowest_hip_frame_current_rep
                                duration_seconds = duration_frames / fps
                                rep_history[last_rep_index]['duration_seconds'] = duration_seconds
                        
                        phase = "lowering"
                        lowest_left_hip_angle_this_rep = left_hip_angle
                        lowest_right_hip_angle_this_rep = right_hip_angle
                        lowest_left_knee_angle_this_rep = left_knee_angle if left_knee_angle is not None else None
                        lowest_right_knee_angle_this_rep = right_knee_angle if right_knee_angle is not None else None
                        deadlift_start_frame = frame_count
                        
                        # Track when we reach the lowest hip angle for duration calculation
                        lowest_hip_frame_current_rep = frame_count
                        
                        # Reset standing max angles for next rep
                        max_left_hip_angle_standing = None
                        max_right_hip_angle_standing = None
                        max_left_knee_angle_standing = None
                        max_right_knee_angle_standing = None
                        # Reset highest frame for next rep
                        highest_hip_frame_current_rep = None
                    
                    elif phase == "lowering":
                        # Update lowest angles and track when lowest hip angle was reached
                        if left_hip_angle < lowest_left_hip_angle_this_rep:
                            lowest_left_hip_angle_this_rep = left_hip_angle
                            lowest_hip_frame_current_rep = frame_count  # Track when lowest hip angle was reached
                        if right_hip_angle < lowest_right_hip_angle_this_rep:
                            lowest_right_hip_angle_this_rep = right_hip_angle
                            lowest_hip_frame_current_rep = frame_count  # Track when lowest hip angle was reached
                        
                        if left_knee_angle is not None and (lowest_left_knee_angle_this_rep is None or left_knee_angle < lowest_left_knee_angle_this_rep):
                            lowest_left_knee_angle_this_rep = left_knee_angle
                        if right_knee_angle is not None and (lowest_right_knee_angle_this_rep is None or right_knee_angle < lowest_right_knee_angle_this_rep):
                            lowest_right_knee_angle_this_rep = right_knee_angle
                        
                        # No need to track max angles during lowering phase anymore
                        
                        # Check if back to standing (using hip angle)
                        if min_hip_angle >= deadlift_up_threshold:
                            phase = "standing"
                            rep_count += 1
                            
                            # Mark first rep as completed so we can start tracking highest frame
                            if not first_rep_completed:
                                first_rep_completed = True
                            
                            rep_data = {
                                'rep_number': rep_count,
                                'left_hip_angle': lowest_left_hip_angle_this_rep,
                                'right_hip_angle': lowest_right_hip_angle_this_rep,
                                'left_knee_angle': lowest_left_knee_angle_this_rep,
                                'right_knee_angle': lowest_right_knee_angle_this_rep,
                                'timestamp': frame_count / fps
                            }
                            rep_history.append(rep_data)
                            
                            # Reset for next rep
                            lowest_left_hip_angle_this_rep = None
                            lowest_right_hip_angle_this_rep = None
                            lowest_left_knee_angle_this_rep = None
                            lowest_right_knee_angle_this_rep = None
                            deadlift_start_frame = None
                
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
                
                # Show current rep count
                cv2.putText(frame, f'Deadlifts: {rep_count}', (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
                
                # Draw rep history on right side
                right_x = width - 300
                y_start = 50
                cv2.putText(frame, "Rep History:", (right_x, y_start), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                for i, rep_data in enumerate(rep_history[-3:]):  # Show last 3 reps to accommodate ROM data
                    y_pos = y_start + 30 + (i * 200)  # Increased spacing to 200 pixels between reps for ROM data
                    cv2.putText(frame, f"Rep {rep_data['rep_number']}:", (right_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show minimum angles (bottom of deadlift)
                    if rep_data['left_hip_angle'] is not None:
                        cv2.putText(frame, f"  L-Hip: {rep_data['left_hip_angle']:.1f} deg", (right_x, y_pos + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    if rep_data['right_hip_angle'] is not None:
                        cv2.putText(frame, f"  R-Hip: {rep_data['right_hip_angle']:.1f} deg", (right_x, y_pos + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Add ROM data if available
                    if rep_data.get('left_hip_rom') is not None and rep_data.get('right_hip_rom') is not None:
                        cv2.putText(frame, f"  L-Hip ROM: {rep_data['left_hip_rom']:.1f} deg", (right_x, y_pos + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                        cv2.putText(frame, f"  R-Hip ROM: {rep_data['right_hip_rom']:.1f} deg", (right_x, y_pos + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # Add knee angles and ROM if available
                    if rep_data['left_knee_angle'] is not None and rep_data['right_knee_angle'] is not None:
                        cv2.putText(frame, f"  L-Knee: {rep_data['left_knee_angle']:.1f} deg", (right_x, y_pos + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(frame, f"  R-Knee: {rep_data['right_knee_angle']:.1f} deg", (right_x, y_pos + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        if rep_data.get('left_knee_rom') is not None and rep_data.get('right_knee_rom') is not None:
                            cv2.putText(frame, f"  L-Knee ROM: {rep_data['left_knee_rom']:.1f} deg", (right_x, y_pos + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            cv2.putText(frame, f"  R-Knee ROM: {rep_data['right_knee_rom']:.1f} deg", (right_x, y_pos + 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        duration = rep_data.get('duration_seconds')
                        if duration is not None:
                            cv2.putText(frame, f"  Duration: {duration:.1f}s", (right_x, y_pos + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
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
