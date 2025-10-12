"""
Overhead Squat Specific Analysis Module
Contains exercise-specific angle calculations and analysis logic
"""

import math
from shared.pose_analysis import angle_between

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
