"""
Visualization Module for Overhead Squat Assessment
Handles chart creation and data visualization
"""

import matplotlib.pyplot as plt
import pandas as pd

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
