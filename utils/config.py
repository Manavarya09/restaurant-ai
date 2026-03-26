"""
Configuration file for Restaurant AI Analytics System
"""
import json
from pathlib import Path

CONFIG = {
    "video": {
        "path": "../data/sample_video.mp4",
        "max_frames": 200,
        "fallback_to_camera": True
    },
    "detection": {
        "model": "yolov8n.pt",
        "confidence": 0.5,
        "device": "auto"
    },
    "tracking": {
        "max_age": 30,
        "min_hits": 3,
        "iou_threshold": 0.3
    },
    "zones": {
        "entrance": {
            "points": [[200, 400], [400, 400], [400, 450], [200, 450]],
            "color": [0, 255, 255]
        },
        "waiting": {
            "points": [[100, 250], [250, 250], [250, 350], [100, 350]],
            "color": [0, 255, 0]
        },
        "dining": {
            "points": [[400, 50], [620, 50], [620, 350], [400, 350]],
            "color": [255, 0, 0]
        }
    },
    "footfall_line": {
        "start": [320, 0],
        "end": [320, 480]
    },
    "staff_detection": {
        "enabled": True,
        "hsv_lower": [0, 0, 0],
        "hsv_upper": [180, 50, 80]
    },
    "output": {
        "dir": "../outputs",
        "save_video": True,
        "save_report": True
    }
}


def load_config(config_path: str = None) -> dict:
    """Load configuration from file or return default."""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            return json.load(f)
    return CONFIG.copy()


def save_config(config: dict, config_path: str = "config.json"):
    """Save configuration to file."""
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
