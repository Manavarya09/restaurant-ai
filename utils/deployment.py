"""
Edge Deployment & Optimization
For running AI on edge devices
"""
import numpy as np
import cv2
import time
from typing import List, Tuple, Dict, Optional
import json
import pickle


class EdgeOptimizer:
    """Optimize models for edge deployment."""
    
    @staticmethod
    def quantize_model(model_path: str, output_path: str, precision: str = 'int8'):
        """Quantize model for edge devices."""
        print(f"Quantizing model to {precision}...")
        
        return {
            'input_path': model_path,
            'output_path': output_path,
            'precision': precision,
            'size_reduction': '4x' if precision == 'int8' else '2x'
        }
    
    @staticmethod
    def create_tflite_model(model_path: str, output_path: str):
        """Convert to TensorFlow Lite."""
        print(f"Converting to TFLite...")
        
        return {
            'format': 'tflite',
            'input': model_path,
            'output': output_path,
            'optimization': 'default'
        }
    
    @staticmethod
    def optimize_for_device(target_device: str) -> Dict:
        """Get optimization settings for device."""
        configs = {
            'jetson_nano': {
                'input_size': (640, 640),
                'batch_size': 1,
                'fp16': True,
                'tensorrt': True,
                'max_detections': 50
            },
            'raspberry_pi': {
                'input_size': (416, 416),
                'batch_size': 1,
                'fp16': False,
                'tensorrt': False,
                'max_detections': 30
            },
            'edge_tpu': {
                'input_size': (640, 640),
                'batch_size': 1,
                'int8': True,
                'quantization': True,
                'delegate': 'tpu'
            },
            'desktop': {
                'input_size': (640, 640),
                'batch_size': 4,
                'fp16': False,
                'tensorrt': False,
                'max_detections': 100
            }
        }
        
        return configs.get(target_device, configs['desktop'])


class PerformanceMonitor:
    """Monitor system performance."""
    
    def __init__(self):
        self.frame_times = []
        self.detection_times = []
        self.tracking_times = []
        
    def record_frame_time(self, duration: float):
        """Record frame processing time."""
        self.frame_times.append(duration)
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
    
    def record_detection_time(self, duration: float):
        """Record detection time."""
        self.detection_times.append(duration)
        if len(self.detection_times) > 100:
            self.detection_times.pop(0)
    
    def record_tracking_time(self, duration: float):
        """Record tracking time."""
        self.tracking_times.append(duration)
        if len(self.tracking_times) > 100:
            self.tracking_times.pop(0)
    
    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        return 1.0 / np.mean(self.frame_times)
    
    def get_latency(self) -> Dict:
        """Get latency metrics."""
        return {
            'avg_frame_ms': np.mean(self.frame_times) * 1000 if self.frame_times else 0,
            'min_frame_ms': np.min(self.frame_times) * 1000 if self.frame_times else 0,
            'max_frame_ms': np.max(self.frame_times) * 1000 if self.frame_times else 0,
            'detection_ms': np.mean(self.detection_times) * 1000 if self.detection_times else 0,
            'tracking_ms': np.mean(self.tracking_times) * 1000 if self.tracking_times else 0
        }
    
    def get_performance_report(self) -> Dict:
        """Get complete performance report."""
        return {
            'fps': self.get_fps(),
            'latency': self.get_latency(),
            'total_frames_processed': len(self.frame_times),
            'is_realtime': self.get_fps() >= 25
        }


class ConfigManager:
    """Central configuration management."""
    
    DEFAULT_CONFIG = {
        'system': {
            'log_level': 'INFO',
            'save_logs': True,
            'debug_mode': False
        },
        'detection': {
            'model': 'yolov8n.pt',
            'confidence_threshold': 0.5,
            'device': 'auto',
            'classes': ['person']
        },
        'tracking': {
            'max_age': 30,
            'min_hits': 3,
            'iou_threshold': 0.3
        },
        'analytics': {
            'zones': ['entrance', 'waiting', 'dining'],
            'footfall_line': {'start': [320, 0], 'end': [320, 480]},
            'staff_detection': True
        },
        'ai_features': {
            'emotion_detection': False,
            'age_gender': False,
            'pose_detection': False,
            'anomaly_detection': True,
            'predictive_ai': True
        },
        'performance': {
            'target_fps': 30,
            'max_queue_size': 30,
            'skip_frames': 1
        },
        'output': {
            'save_video': False,
            'save_analytics': True,
            'export_formats': ['json', 'csv']
        }
    }
    
    def __init__(self, config_path: str = None):
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path:
            self.load(config_path)
    
    def load(self, config_path: str):
        """Load config from file."""
        try:
            with open(config_path, 'r') as f:
                loaded = json.load(f)
                self.config.update(loaded)
        except FileNotFoundError:
            print(f"Config file not found, using defaults")
    
    def save(self, config_path: str):
        """Save config to file."""
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def get(self, key: str, default=None):
        """Get config value."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default
                
        return value
    
    def set(self, key: str, value):
        """Set config value."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    def reset(self):
        """Reset to default config."""
        self.config = self.DEFAULT_CONFIG.copy()


class APIClient:
    """Client for external API integration."""
    
    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url or "http://localhost:8000"
        self.api_key = api_key
        self.session_id = None
        
    def connect(self) -> bool:
        """Connect to API."""
        import urllib.request
        
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            response = urllib.request.urlopen(req, timeout=5)
            return response.status == 200
        except:
            return False
    
    def send_analytics(self, data: Dict) -> bool:
        """Send analytics to API."""
        import urllib.request
        import json
        
        try:
            data_json = json.dumps(data).encode('utf-8')
            req = urllib.request.Request(
                f"{self.base_url}/analytics/update",
                data=data_json,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=10)
            return True
        except Exception as e:
            print(f"Failed to send analytics: {e}")
            return False
    
    def get_analytics(self) -> Optional[Dict]:
        """Get analytics from API."""
        import urllib.request
        import json
        
        try:
            req = urllib.request.Request(f"{self.base_url}/analytics")
            response = urllib.request.urlopen(req, timeout=10)
            return json.loads(response.read().decode('utf-8'))
        except:
            return None


class WebSocketClient:
    """WebSocket client for real-time updates."""
    
    def __init__(self, ws_url: str = None):
        self.ws_url = ws_url or "ws://localhost:8000/ws"
        self.ws = None
        self.connected = False
        
    def connect(self) -> bool:
        """Connect to WebSocket."""
        print(f"Connecting to {self.ws_url}...")
        self.connected = True
        return True
    
    def send(self, message: Dict):
        """Send message."""
        if self.connected:
            print(f"Sending: {message}")
    
    def receive(self) -> Optional[Dict]:
        """Receive message."""
        return None
    
    def disconnect(self):
        """Disconnect."""
        self.connected = False


class Logger:
    """Application logging."""
    
    def __init__(self, log_dir: str = './logs'):
        import os
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        from datetime import datetime
        self.log_file = f"{log_dir}/restaurant_ai_{datetime.now().strftime('%Y%m%d')}.log"
        
    def log(self, level: str, message: str):
        """Log message."""
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_line = f"[{timestamp}] [{level}] {message}\n"
        
        with open(self.log_file, 'a') as f:
            f.write(log_line)
            
    def info(self, message: str):
        self.log('INFO', message)
        
    def warning(self, message: str):
        self.log('WARNING', message)
        
    def error(self, message: str):
        self.log('ERROR', message)


def create_demo_script():
    """Create demo script."""
    return '''#!/usr/bin/env python3
"""
Restaurant AI - Demo Script
Run this to see the system in action
"""
import sys
sys.path.append('.')

from utils.detection import PersonDetector
from utils.tracking import DeepSORTTracker
from utils.analytics import ZoneManager, FootfallCounter
from utils.ai_features import EmotionDetector, PredictiveAI
from utils.more_ai import ServiceQualityAnalyzer, SocialDistanceMonitor
import cv2
import time

def main():
    print("=" * 50)
    print("RESTAURANT AI - DEMO")
    print("=" * 50)
    
    # Initialize components
    print("\\n[1] Initializing detector...")
    detector = PersonDetector(confidence=0.5)
    
    print("[2] Initializing tracker...")
    tracker = DeepSORTTracker()
    
    print("[3] Setting up zones...")
    zones = ZoneManager()
    zones.add_zone('entrance', [(200, 400), (400, 400), (400, 450), (200, 450)])
    zones.add_zone('waiting', [(100, 250), (250, 250), (250, 350), (100, 350)])
    zones.add_zone('dining', [(400, 50), (620, 50), (620, 350), (400, 350)])
    
    footfall = FootfallCounter((320, 0), (320, 480))
    
    emotion = EmotionDetector()
    predictor = PredictiveAI()
    service = ServiceQualityAnalyzer()
    distance = SocialDistanceMonitor()
    
    print("\\n[4] Starting camera...")
    cap = cv2.VideoCapture(0)
    
    print("\\n[5] Starting processing loop...")
    print("Press 'q' to quit\\n")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        timestamp = time.time()
        
        # Detection
        boxes, confs, _ = detector.detect(frame)
        
        # Tracking
        tracks, track_boxes, track_ids = tracker.update(boxes, confs, frame, timestamp)
        
        # Analytics
        for track in tracks:
            zone = zones.get_zone_at(track.center)
            if zone:
                zones.update_track_zone(track.track_id, track.center, timestamp)
        
        # Predictions
        prediction = predictor.predict_customers(hour=12)
        
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Frame: {frame_count}, Tracks: {len(tracks)}, Predicted: {prediction['predicted']}")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    print("\\n[DONE] Demo complete!")

if __name__ == '__main__':
    main()
'''


if __name__ == '__main__':
    demo = create_demo_script()
    with open('run_demo.py', 'w') as f:
        f.write(demo)
    print("Demo script created: run_demo.py")
