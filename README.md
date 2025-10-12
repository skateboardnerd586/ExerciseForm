# Exercise Analysis Platform - AI-Powered Movement Analysis

A comprehensive web application for analyzing exercise form using computer vision and AI. Currently supports **Overhead Squat** and **Deadlift** analysis with more exercises coming soon.

## 🚀 Streamlit Cloud Deployment

**Main File**: `app_simple.py` (modular version with environment-aware rotation and multi-exercise support)

**Legacy File**: `app.py` (original single-file version)

**Note**: Streamlit Cloud automatically uses `app_simple.py` as the main file, which contains the improved modular version with automatic environment detection, rotation handling, and multi-exercise navigation.

## Features

### 🏋️ Multi-Exercise Support
- **Overhead Squat Analysis**: Analyze ankle mobility, hip flexibility, shoulder mobility, and core stability
- **Deadlift Analysis**: Analyze hip hinge mechanics, back position, and lifting form
- **Modular Design**: Easy to add new exercises

### 🎥 Computer Vision & AI
- **Video Upload & Processing**: Upload exercise videos and automatically detect reps using YOLOv11 Pose
- **Real-time Analysis**: Track joint angles throughout each movement
- **Interactive Visualizations**: Charts showing angle trends, ROM, and timing consistency
- **AI-Powered Insights**: Get professional analysis and recommendations using OpenAI's GPT
- **Data Export**: Download detailed rep data as CSV files

### 🔄 Smart Video Processing
- **Automatic Rotation Detection**: Handles mobile video orientation automatically
- **Environment-Aware**: Works seamlessly in both local and cloud environments
- **Manual Override**: Option to manually correct video orientation if needed

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
streamlit run app_simple.py
```

2. Open your browser to `http://localhost:8501`

3. **Choose Your Exercise**: Select from the home page:
   - **Overhead Squat Analysis**: For full-body mobility assessment
   - **Deadlift Analysis**: For posterior chain strength evaluation

4. Upload a video file of someone performing the selected exercise

5. Adjust detection thresholds in the sidebar if needed

6. Click "Process Video" to analyze the movement

7. View results, charts, and get AI analysis

## Configuration

### Exercise-Specific Detection Thresholds

#### Overhead Squat Analysis
- **Squat Down Threshold**: Angle below which squat phase begins (default: 130°)
- **Squat Up Threshold**: Angle above which squat phase ends (default: 150°)
- Tracks: Knee angles, hip angles, shoulder angles, ROM, timing

#### Deadlift Analysis  
- **Deadlift Down Threshold**: Angle below which deadlift phase begins (default: 140°)
- **Deadlift Up Threshold**: Angle above which deadlift phase ends (default: 160°)
- Tracks: Hip angles, knee angles, timing, form analysis

### AI Analysis
- Requires OpenAI API key
- Provides professional form analysis and recommendations
- Analyzes bilateral symmetry, timing, ROM, and potential issues
- Exercise-specific insights and improvement suggestions

## Data Output

### Overhead Squat Analysis
- Rep count and timing
- Left/right knee angles (min/max/ROM)
- Left/right hip angles (min/max/ROM)  
- Left/right shoulder angles (min/max/ROM)
- Duration of each squat
- Timestamp for each rep

### Deadlift Analysis
- Rep count and timing
- Left/right hip angles (min/max/ROM)
- Left/right knee angles (min/max/ROM)
- Duration of each deadlift
- Timestamp for each rep

## Technical Details

- **Modular Architecture**: Clean separation of concerns with shared utilities
- **YOLOv11 Pose**: State-of-the-art pose detection for accurate keypoint tracking
- **Frame-by-Frame Processing**: Precise movement analysis with real-time progress updates
- **Hysteresis Logic**: Prevents threshold bouncing for reliable rep detection
- **Environment Detection**: Automatic cloud vs local environment handling
- **Smart Rotation**: Automatic mobile video orientation correction

## Project Structure

```
BenchTracker/
├── app_simple.py              # Main application entry point
├── pages/
│   ├── overhead_squat/        # Overhead squat analysis module
│   │   ├── main.py           # Main page logic
│   │   ├── video_processing.py # Video analysis
│   │   ├── analysis.py       # Angle calculations
│   │   ├── ai_analysis.py    # OpenAI integration
│   │   └── visualization.py  # Charts and graphs
│   └── deadlift/             # Deadlift analysis module
│       ├── main.py           # Main page logic
│       ├── video_processing.py # Video analysis
│       ├── analysis.py       # Angle calculations
│       ├── ai_analysis.py    # OpenAI integration
│       └── visualization.py  # Charts and graphs
├── shared/                   # Shared utilities
│   ├── common.py            # Common functions and CSS
│   ├── pose_analysis.py     # Pose detection and basic angles
│   └── video_processing.py  # Video handling and rotation
└── requirements.txt         # Python dependencies
```

## Troubleshooting

- **Video Format**: Ensure video files are in supported formats (MP4, MOV, AVI, MKV)
- **Video Quality**: For best results, use videos with clear visibility of the person
- **Detection Issues**: Adjust thresholds if detection is not working properly
- **AI Analysis**: Check OpenAI API key if AI analysis is not working
- **Rotation Issues**: Use manual rotation override if automatic detection fails
- **Performance**: Large videos may take longer to process - progress is shown in real-time

## Contributing

This platform is designed to be easily extensible. To add new exercises:

1. Create a new module in `pages/[exercise_name]/`
2. Implement the required analysis functions
3. Add navigation to `app_simple.py`
4. Follow the existing patterns for consistency

## License

This project is open source and available under the MIT License.
