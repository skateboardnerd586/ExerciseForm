"""
Deadlift Specific Analysis Module
Contains exercise-specific angle calculations and analysis logic for deadlifts
"""

import math
from shared.pose_analysis import angle_between

def calculate_hip_angles_deadlift(keypoints):
    """Calculate hip angles specifically for deadlift analysis"""
    LEFT_SHOULDER = 5
    LEFT_HIP = 11
    LEFT_KNEE = 13
    RIGHT_SHOULDER = 6
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    
    # Check if all required keypoints exist and have valid confidence
    required_keypoints = [LEFT_SHOULDER, LEFT_HIP, LEFT_KNEE, RIGHT_SHOULDER, RIGHT_HIP, RIGHT_KNEE]
    for kp_idx in required_keypoints:
        if kp_idx >= len(keypoints) or keypoints[kp_idx] is None or len(keypoints[kp_idx]) < 3 or keypoints[kp_idx][2] < 0.5:
            return None, None
    
    left_shoulder = keypoints[LEFT_SHOULDER]
    left_hip = keypoints[LEFT_HIP]
    left_knee = keypoints[LEFT_KNEE]
    right_shoulder = keypoints[RIGHT_SHOULDER]
    right_hip = keypoints[RIGHT_HIP]
    right_knee = keypoints[RIGHT_KNEE]
    
    # Calculate hip angle (torso to thigh angle)
    left_torso = (left_shoulder[0] - left_hip[0], left_shoulder[1] - left_hip[1])
    left_thigh = (left_knee[0] - left_hip[0], left_knee[1] - left_hip[1])
    left_angle = angle_between(left_torso, left_thigh)
    
    right_torso = (right_shoulder[0] - right_hip[0], right_shoulder[1] - right_hip[1])
    right_thigh = (right_knee[0] - right_hip[0], right_knee[1] - right_hip[1])
    right_angle = angle_between(right_torso, right_thigh)
    
    return left_angle, right_angle

def calculate_knee_angles_deadlift(keypoints):
    """Calculate knee angles specifically for deadlift analysis"""
    LEFT_HIP = 11
    LEFT_KNEE = 13
    LEFT_ANKLE = 15
    RIGHT_HIP = 12
    RIGHT_KNEE = 14
    RIGHT_ANKLE = 16
    
    # Check if all required keypoints exist and have valid confidence
    required_keypoints = [LEFT_HIP, LEFT_KNEE, LEFT_ANKLE, RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE]
    for kp_idx in required_keypoints:
        if kp_idx >= len(keypoints) or keypoints[kp_idx] is None or len(keypoints[kp_idx]) < 3 or keypoints[kp_idx][2] < 0.5:
            return None, None
    
    left_hip = keypoints[LEFT_HIP]
    left_knee = keypoints[LEFT_KNEE]
    left_ankle = keypoints[LEFT_ANKLE]
    right_hip = keypoints[RIGHT_HIP]
    right_knee = keypoints[RIGHT_KNEE]
    right_ankle = keypoints[RIGHT_ANKLE]
    
    # Calculate knee angle (thigh to shin angle)
    left_thigh = (left_hip[0] - left_knee[0], left_hip[1] - left_knee[1])
    left_shin = (left_ankle[0] - left_knee[0], left_ankle[1] - left_knee[1])
    left_angle = angle_between(left_thigh, left_shin)
    
    right_thigh = (right_hip[0] - right_knee[0], right_hip[1] - right_knee[1])
    right_shin = (right_ankle[0] - right_knee[0], right_ankle[1] - right_knee[1])
    right_angle = angle_between(right_thigh, right_shin)
    
    return left_angle, right_angle
