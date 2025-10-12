"""
Deadlift Visualization Module
Handles chart creation and data visualization for deadlift analysis
"""

import matplotlib.pyplot as plt
import pandas as pd

def create_deadlift_charts(rep_data):
    """Create matplotlib charts for deadlift data"""
    if not rep_data:
        return None
    
    df = pd.DataFrame(rep_data)
    
    # Check if we have ROM data
    has_rom_data = any(col in df.columns for col in ['left_hip_rom', 'left_knee_rom'])
    
    if has_rom_data:
        # Create figure with subplots for ROM data
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        fig.suptitle('Deadlift Assessment Charts', fontsize=16)
        
        # Hip angles over time
        axes[0, 0].plot(df['rep_number'], df['left_hip_angle'], 'b-o', label='Left Hip', linewidth=2)
        axes[0, 0].plot(df['rep_number'], df['right_hip_angle'], 'r-o', label='Right Hip', linewidth=2)
        axes[0, 0].set_title('Hip Angles by Rep (Smallest Angles)')
        axes[0, 0].set_xlabel('Rep Number')
        axes[0, 0].set_ylabel('Angle (degrees)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hip ROM over time
        if 'left_hip_rom' in df.columns and df['left_hip_rom'].notna().any():
            axes[0, 1].plot(df['rep_number'], df['left_hip_rom'], 'b-o', label='Left Hip ROM', linewidth=2)
        if 'right_hip_rom' in df.columns and df['right_hip_rom'].notna().any():
            axes[0, 1].plot(df['rep_number'], df['right_hip_rom'], 'r-o', label='Right Hip ROM', linewidth=2)
        axes[0, 1].set_title('Hip Range of Motion by Rep')
        axes[0, 1].set_xlabel('Rep Number')
        axes[0, 1].set_ylabel('ROM (degrees)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Knee angles over time
        axes[1, 0].plot(df['rep_number'], df['left_knee_angle'], 'g-o', label='Left Knee', linewidth=2)
        axes[1, 0].plot(df['rep_number'], df['right_knee_angle'], 'm-o', label='Right Knee', linewidth=2)
        axes[1, 0].set_title('Knee Angles by Rep (Smallest Angles)')
        axes[1, 0].set_xlabel('Rep Number')
        axes[1, 0].set_ylabel('Angle (degrees)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Knee ROM over time
        if 'left_knee_rom' in df.columns and df['left_knee_rom'].notna().any():
            axes[1, 1].plot(df['rep_number'], df['left_knee_rom'], 'g-o', label='Left Knee ROM', linewidth=2)
        if 'right_knee_rom' in df.columns and df['right_knee_rom'].notna().any():
            axes[1, 1].plot(df['rep_number'], df['right_knee_rom'], 'm-o', label='Right Knee ROM', linewidth=2)
        axes[1, 1].set_title('Knee Range of Motion by Rep')
        axes[1, 1].set_xlabel('Rep Number')
        axes[1, 1].set_ylabel('ROM (degrees)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Symmetry analysis (hip)
        hip_symmetry_diff = df['left_hip_angle'] - df['right_hip_angle']
        axes[2, 0].plot(df['rep_number'], hip_symmetry_diff, 'purple', marker='o', linewidth=2)
        axes[2, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[2, 0].set_title('Left vs Right Hip Difference')
        axes[2, 0].set_xlabel('Rep Number')
        axes[2, 0].set_ylabel('Left - Right Hip Angle (degrees)')
        axes[2, 0].grid(True, alpha=0.3)
        
        # Duration over time
        axes[2, 1].plot(df['rep_number'], df['duration_seconds'], 'orange', marker='o', linewidth=2)
        axes[2, 1].set_title('Deadlift Duration by Rep')
        axes[2, 1].set_xlabel('Rep Number')
        axes[2, 1].set_ylabel('Duration (seconds)')
        axes[2, 1].grid(True, alpha=0.3)
        
    else:
        # Original layout without ROM data
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Deadlift Assessment Charts', fontsize=16)
        
        # Hip angles over time
        axes[0, 0].plot(df['rep_number'], df['left_hip_angle'], 'b-o', label='Left Hip', linewidth=2)
        axes[0, 0].plot(df['rep_number'], df['right_hip_angle'], 'r-o', label='Right Hip', linewidth=2)
        axes[0, 0].set_title('Hip Angles by Rep (Smallest Angles)')
        axes[0, 0].set_xlabel('Rep Number')
        axes[0, 0].set_ylabel('Angle (degrees)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Knee angles over time
        axes[0, 1].plot(df['rep_number'], df['left_knee_angle'], 'g-o', label='Left Knee', linewidth=2)
        axes[0, 1].plot(df['rep_number'], df['right_knee_angle'], 'm-o', label='Right Knee', linewidth=2)
        axes[0, 1].set_title('Knee Angles by Rep (Smallest Angles)')
        axes[0, 1].set_xlabel('Rep Number')
        axes[0, 1].set_ylabel('Angle (degrees)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Symmetry analysis (hip)
        hip_symmetry_diff = df['left_hip_angle'] - df['right_hip_angle']
        axes[1, 0].plot(df['rep_number'], hip_symmetry_diff, 'purple', marker='o', linewidth=2)
        axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1, 0].set_title('Left vs Right Hip Difference')
        axes[1, 0].set_xlabel('Rep Number')
        axes[1, 0].set_ylabel('Left - Right Hip Angle (degrees)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Duration over time
        axes[1, 1].plot(df['rep_number'], df['duration_seconds'], 'orange', marker='o', linewidth=2)
        axes[1, 1].set_title('Deadlift Duration by Rep')
        axes[1, 1].set_xlabel('Rep Number')
        axes[1, 1].set_ylabel('Duration (seconds)')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig
