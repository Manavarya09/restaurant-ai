# Training Configuration
# Restaurant AI - Custom Model Training

## Quick Start

### 1. Install Requirements
```bash
pip install -r requirements.txt
pip install ultralytics torch torchvision
```

### 2. Create Dataset
```bash
python -c "from utils.train import create_sample_dataset; create_sample_dataset('./data/restaurant')"
```

### 3. Train Model
```python
from utils.train import train_model, TrainingConfig

config = TrainingConfig(model_size='n')
config.create_data_yaml('data.yaml')

results = train_model(
    data_yaml='data.yaml',
    model_size='n',
    epochs=100,
    device='cuda'  # or 'cpu'
)
```

### 4. Evaluate & Export
```python
from utils.train import evaluate_model, export_model

metrics = evaluate_model('runs/train/restaurant_ai/weights/best.pt')
export_model('runs/train/restaurant_ai/weights/best.pt', 'onnx')
```

## Dataset Structure

```
data/
└── restaurant/
    ├── images/
    │   ├── train/
    │   │   ├── img_0001.jpg
    │   │   └── img_0002.jpg
    │   └── val/
    │       ├── img_0045.jpg
    │       └── img_0046.jpg
    └── labels/
        ├── train/
        │   ├── img_0001.txt
        │   └── img_0002.txt
        └── val/
            ├── img_0045.txt
            └── img_0046.txt
```

## Label Format (YOLO)

Each `.txt` file contains one line per object:
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1).

## Classes

| ID | Name | Description |
|----|------|-------------|
| 0 | person | Any person |
| 1 | staff | Staff member (uniform) |
| 2 | customer | Customer |
| 3 | group | Group of people |
| 4 | queue | Queue area |

## Training Options

### Model Sizes
- `yolov8n.pt` - Nano (fastest, lowest accuracy)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - X-Large (slowest, highest accuracy)

### Recommended Settings

| Dataset Size | Model | Epochs | Batch |
|--------------|-------|--------|-------|
| < 100 | n | 50 | 8 |
| 100-500 | n/s | 100 | 16 |
| 500+ | s/m | 200 | 32 |

## Data Collection

Use the dataset module to collect data from your CCTV:
```python
from utils.dataset import VideoDatasetExtractor, DataCollector

# Extract frames from video
extractor = VideoDatasetExtractor('./data/raw')
frames = extractor.extract_frames('video.mp4', interval=30)

# Collect from camera
collector = DataCollector(detector, tracker, './data/collected')
data = collector.collect_from_camera(duration=60)
```

## Annotation Tools

Use LabelImg for manual annotation:
```bash
pip install labelimg
labelImg
```

Or use CVAT for team annotation:
```bash
# Install CVAT
docker run -d -p 8080:8080 cvat/server
```

## Fine-tuning Pretrained Model

For best results, fine-tune from COCO pretrained:
```python
from ultralytics import YOLO

model = YOLO('yolov8n.pt')
model.train(
    data='data.yaml',
    epochs=100,
    freeze=10,  # Freeze backbone
    lr0=0.001   # Lower learning rate for fine-tuning
)
```

## Monitoring

Training results are saved to:
- `runs/train/restaurant_ai/`
  - `weights/best.pt`
  - `weights/last.pt`
  - `results.csv`
  - `confusion_matrix.png`
  - `labels_correlogram.png`
  - `train_batch*.jpg`

## Common Issues

### CUDA Out of Memory
```python
# Reduce batch size
model.train(data='data.yaml', batch=4, imgsz=320)
```

### Low Accuracy
- Increase training data
- Use data augmentation
- Try larger model
- Increase epochs

### Model Too Slow
- Use smaller model (n instead of m)
- Reduce input size (imgsz=320)
- Use GPU
