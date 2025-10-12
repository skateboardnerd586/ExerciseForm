"""
Overhead Squat AI Analysis Module
Handles OpenAI integration for overhead squat movement analysis and recommendations
"""

import openai
import numpy as np

def analyze_overhead_squat_with_openai(rep_data, api_key):
    """Analyze overhead squat data using OpenAI LLM"""
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
        
        # Calculate ROM averages
        knee_roms = []
        hip_roms = []
        shoulder_roms = []
        
        for rep in rep_data:
            if rep.get('left_knee_rom') is not None and rep.get('right_knee_rom') is not None:
                knee_roms.append((rep['left_knee_rom'] + rep['right_knee_rom']) / 2)
            if rep.get('left_hip_rom') is not None and rep.get('right_hip_rom') is not None:
                hip_roms.append((rep['left_hip_rom'] + rep['right_hip_rom']) / 2)
            if rep.get('left_shoulder_rom') is not None and rep.get('right_shoulder_rom') is not None:
                shoulder_roms.append((rep['left_shoulder_rom'] + rep['right_shoulder_rom']) / 2)
        
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
        """
        
        # Add ROM data if available
        if knee_roms:
            avg_knee_rom = np.mean(knee_roms)
            data_summary += f"\n        - Average Knee Range of Motion: {avg_knee_rom:.1f} deg"
        if hip_roms:
            avg_hip_rom = np.mean(hip_roms)
            data_summary += f"\n        - Average Hip Range of Motion: {avg_hip_rom:.1f} deg"
        if shoulder_roms:
            avg_shoulder_rom = np.mean(shoulder_roms)
            data_summary += f"\n        - Average Shoulder Range of Motion: {avg_shoulder_rom:.1f} deg"
        
        data_summary += "\n        \n        Individual Rep Details:"
        
        for i, rep in enumerate(rep_data):
            data_summary += f"""
        Rep {rep['rep_number']}:
        - Left Knee: {rep['left_knee_angle']:.1f} deg (at bottom of squat)
        - Right Knee: {rep['right_knee_angle']:.1f} deg (at bottom of squat)
        - Left Hip: {rep['left_hip_angle']:.1f} deg (at bottom of squat, if available)
        - Right Hip: {rep['right_hip_angle']:.1f} deg (at bottom of squat, if available)
        - Left Shoulder: {rep['left_shoulder_angle']:.1f} deg (at bottom of squat, if available)
        - Right Shoulder: {rep['right_shoulder_angle']:.1f} deg (at bottom of squat, if available)
        - Duration: {rep['duration_seconds']:.1f}s"""
            
            # Add ROM data for this rep if available
            if rep.get('left_knee_rom') is not None and rep.get('right_knee_rom') is not None:
                data_summary += f"\n        - Knee ROM: L={rep['left_knee_rom']:.1f}°, R={rep['right_knee_rom']:.1f}°"
            if rep.get('left_hip_rom') is not None and rep.get('right_hip_rom') is not None:
                data_summary += f"\n        - Hip ROM: L={rep['left_hip_rom']:.1f}°, R={rep['right_hip_rom']:.1f}°"
            if rep.get('left_shoulder_rom') is not None and rep.get('right_shoulder_rom') is not None:
                data_summary += f"\n        - Shoulder ROM: L={rep['left_shoulder_rom']:.1f}°, R={rep['right_shoulder_rom']:.1f}°"
            
            data_summary += "\n        "
        
        prompt = f"""
        As a professional fitness trainer and biomechanics expert, analyze this overhead squat tracking data and provide concise insights. 
        
        IMPORTANT DISCLAIMERS:
        - This analysis is based on limited movement data and should not replace professional medical advice
        - I am not a doctor or licensed physical therapist
        - This assessment is for educational and training purposes only
        - Consult healthcare professionals for any pain, injury concerns, or medical conditions
        
        {data_summary}
        
        Please provide a concise analysis (under 500 words) covering:
        1. Overall overhead squat form and depth assessment
        2. Range of motion analysis (knee, hip, shoulder ROM consistency and quality)
        3. Bilateral symmetry analysis (left vs right leg ROM and angles)
        4. Ankle, Hip, Shoulder, and Core stability assessment
        5. Overactive and underactive muscles assessment
        6. Key corrective exercises for identified limitations
        7. Training recommendations for improvement
        
        Focus on practical, actionable exercises and training strategies the person can implement safely.
        Please use second person to address the user and emphasize consulting professionals for any concerns.
        Keep your response under 500 words.
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
