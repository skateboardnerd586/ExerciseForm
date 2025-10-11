# Squat Tracker AI - Streamlit App

A comprehensive web application for analyzing squat form using computer vision and AI.

## Features

- **Video Upload & Processing**: Upload squat videos and automatically detect reps using YOLOv11 Pose
- **Real-time Analysis**: Track knee and hip angles throughout each squat
- **Interactive Visualizations**: Charts showing angle trends and timing consistency
- **AI-Powered Insights**: Get professional analysis and recommendations using OpenAI's GPT
- **Data Export**: Download detailed rep data as CSV files

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the YOLOv11 pose model (will be downloaded automatically on first run):
```bash
# The model will be downloaded automatically when you first run the app
```

3. Get your OpenAI API key:
   - Visit [OpenAI API](https://platform.openai.com/api-keys)
   - Create a new API key
   - Copy the key for use in the app

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your browser to `http://localhost:8501`

3. Upload a video file of someone performing squats

4. Adjust detection thresholds in the sidebar if needed

5. Click "Process Video" to analyze the squats

6. View results, charts, and get AI analysis

## Configuration

### Detection Thresholds
- **Squat Down Threshold**: Angle below which squat phase begins (default: 130°)
- **Squat Up Threshold**: Angle above which squat phase ends (default: 150°)

### AI Analysis
- Requires OpenAI API key
- Provides professional form analysis and recommendations
- Analyzes bilateral symmetry, timing, and potential issues

## Data Output

The app tracks:
- Rep count and timing
- Left/right knee angles at lowest point
- Left/right hip angles at lowest point
- Duration of each squat
- Timestamp for each rep

## Technical Details

- Uses YOLOv11 Pose for accurate keypoint detection
- Processes videos frame-by-frame for precise tracking
- Implements hysteresis to prevent threshold bouncing
- Provides real-time progress updates during processing

## Troubleshooting

- Ensure video files are in supported formats (MP4, MOV, AVI, MKV)
- For best results, use videos with clear visibility of the person
- Adjust thresholds if detection is not working properly
- Check OpenAI API key if AI analysis is not working
