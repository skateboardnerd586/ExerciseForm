"""
Pose Analysis Module for Overhead Squat Assessment
Handles pose detection, keypoint tracking, and angle calculations
"""

import cv2
import math
import numpy as np
from ultralytics import YOLO

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
        print(f"YOLOv11 Pose inference error: {e}")
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
