"""
Shared Video Processing Module
Handles common video operations like rotation, orientation detection, and basic processing
"""

import cv2
import tempfile
import os
import streamlit as st
from datetime import datetime

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

def setup_video_processing(video_file, manual_rotation=0):
    """Setup video processing with rotation detection and temporary file handling"""
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
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
            return None, None, None, None, None
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Adjust dimensions if video will be rotated
        if total_rotation in [90, 270]:
            width, height = height, width
        
        return cap, fps, width, height, total_rotation, tmp_path
        
    except Exception as e:
        st.error(f"Error setting up video processing: {str(e)}")
        os.unlink(tmp_path)
        return None, None, None, None, None, None

def cleanup_video_processing(tmp_path):
    """Clean up temporary video file"""
    try:
        os.unlink(tmp_path)
    except Exception:
        pass  # Ignore cleanup errors
