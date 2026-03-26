# Restaurant AI - Smart Restaurant Analytics

A complete Python-based AI system for smart restaurant analytics using computer vision. The system uses video feed (CCTV or recorded video) to detect and track human footfall, analyze staff activity, and provide actionable insights.

## Features

- **People Detection**: YOLOv8-based person detection
- **Object Tracking**: DeepSORT-style tracking with unique IDs
- **Footfall Counting**: Virtual entry/exit line counting
- **Zone-Based Tracking**: Track time spent in different zones
- **Wait Time Estimation**: Measure customer wait times
- **Staff Detection**: Color-based staff uniform detection
- **Staff Efficiency Metrics**: Idle time, movement frequency, interactions
- **Analytics Engine**: Generate insights on staffing and service quality
- **Visualization**: Interactive plots and heatmaps

## Project Structure

```
restaurant-ai/
├── notebooks/
│   ├── detection.ipynb      # YOLO person detection
│   ├── tracking.ipynb        # Object tracking
│   ├── analytics.ipynb      # Analytics and insights
│   └── visualization.ipynb  # Data visualization
├── models/                  # YOLO models
├── utils/                   # Utility modules
│   ├── detection.py         # Person detection
│   ├── tracking.py         # Object tracking
│   ├── analytics.py        # Analytics components
│   └── visualization.py    # Plotting functions
├── data/                    # Video input
├── outputs/                 # Analytics output
├── app/                     # FastAPI backend
└── requirements.txt        # Dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Manavarya09/restaurant-ai.git
cd restaurant-ai
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Download YOLOv8 model (will be auto-downloaded on first run):
```bash
yolo export model=yolov8n.pt  # Optional: download model explicitly
```

## Quick Start

### Option 1: Use Sample Video

1. Download a sample video:
```bash
# Download sample video from common sources
# Example using wget (replace with actual video URL)
wget -O data/sample_video.mp4 "https://example.com/sample.mp4"
```

Or use your own video file by placing it in `data/sample_video.mp4`.

### Option 2: Use Camera

The system can use webcam or CCTV camera as input. Just change the video path in notebooks.

## Running the Notebooks

1. Start Jupyter:
```bash
jupyter notebook
```

2. Open and run notebooks in order:
   1. `detection.ipynb` - Test person detection
   2. `tracking.ipynb` - Test object tracking
   3. `analytics.ipynb` - Run complete analytics
   - `visualization.ipynb` - View results

## FastAPI Backend

Start the API server:
```bash
cd app
uvicorn main:app --reload
```

API Endpoints:
- `GET /` - API documentation
- `GET /analytics` - Get current analytics
- `POST /update` - Update analytics data
- `POST /video/analyze` - Analyze video frame
- `GET /health` - Health check
- `POST /reset` - Reset analytics

## Configuration

### Zone Definition

Edit zone coordinates in `analytics.ipynb`:
```python
zone_manager.add_zone('entrance', [(x1, y1), (x2, y2), ...])
zone_manager.add_zone('waiting', [(x1, y1), (x2, y2), ...])
zone_manager.add_zone('dining', [(x1, y1), (x2, y2), ...])
```

### Staff Detection

Adjust HSV thresholds for staff uniform:
```python
staff_detector = StaffDetector(
    hsv_lower=(0, 0, 0),      # Lower bound
    hsv_upper=(180, 50, 80)   # Upper bound
)
```

### Entry/Exit Line

Set counting line position:
```python
footfall_counter = FootfallCounter(
    line_start=(x1, y1),
    line_end=(x2, y2)
)
```

## Sample Analytics Output

```
==================================================
CURRENT STATISTICS
==================================================
Entries: 45
Exits: 38
Current customers: 7
Staff count: 3
Average wait time: 180.5s (3.0 min)
Average idle time: 45.2s
==================================================

INSIGHTS AND RECOMMENDATIONS
==================================================

Recommendations:
  - Customer-to-staff ratio is 2.3:1 - consider reducing staff
```

## Requirements

- Python 3.10+
- OpenCV
- Ultralytics YOLOv8
- NumPy, Pandas
- Matplotlib, Plotly
- FastAPI (for backend)

## Troubleshooting

### CUDA/GPU Issues
For GPU acceleration, install torch with CUDA:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Mac (Apple Silicon) Issues
Install compatible packages:
```bash
pip install opencv-python numpy pandas
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Video Not Found
If video file is not found, the system will automatically use camera (webcam) as fallback.

## License

MIT License

## Author

Manavarya09
