# Sample Video Instructions

## Getting a Sample Video

This system works with any video file (MP4, AVI, etc.) containing people walking. Here are options:

### Option 1: Download from Internet

```bash
# Using yt-dlp (YouTube)
pip install yt-dlp
yt-dlp -f "best[height<=480]" -o "data/sample_video.mp4" "https://www.youtube.com/watch?v=VIDEO_ID"

# Using wget (direct URL)
wget -O data/sample_video.mp4 "https://example.com/video.mp4"
```

### Option 2: Use OpenCV Camera

The notebooks already fall back to webcam if no video file is found:
- `detection.ipynb`: Uses camera if VIDEO_PATH not found
- `tracking.ipynb`: Uses camera if VIDEO_PATH not found
- `analytics.ipynb`: Uses camera if VIDEO_PATH not found

### Option 3: Record Your Own

```python
import cv2

cap = cv2.VideoCapture(0)
out = cv2.VideoWriter('data/sample_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (640, 480))

for _ in range(300):  # Record 10 seconds at 30fps
    ret, frame = cap.read()
    out.write(frame)

cap.release()
out.release()
```

## Video Requirements

- Format: MP4, AVI, or any OpenCV-supported format
- Resolution: Any (zones will need adjustment)
- Content: People walking in frame
- Minimum: 10+ seconds for meaningful analytics

## Recommended Sample Videos

1. Pedestrian walk videos from datasets like Penn-Fudan
2. Store/retail CCTV footage
3. Office common area footage

Note: Make sure you have rights to use the video for analytics purposes.
