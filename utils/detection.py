import numpy as np
import cv2
from ultralytics import YOLO
from typing import List, Tuple, Optional
import time


class PersonDetector:
    """YOLOv8-based person detector for restaurant analytics."""
    
    def __init__(self, model_path: str = 'yolov8n.pt', confidence: float = 0.5):
        """
        Initialize the person detector.
        
        Args:
            model_path: Path to YOLO model (default: yolov8n.pt)
            confidence: Minimum confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.class_name = 'person'
        self.person_class_id = 0
        
    def detect(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int]]:
        """
        Detect persons in a frame.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            boxes: List of bounding boxes [x1, y1, x2, y2]
            confidences: List of confidence scores
            class_ids: List of class IDs
        """
        results = self.model(frame, verbose=False)[0]
        
        boxes = []
        confidences = []
        class_ids = []
        
        if results.boxes is not None:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls_id == self.person_class_id and conf >= self.confidence:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    boxes.append(np.array([x1, y1, x2, y2]))
                    confidences.append(conf)
                    class_ids.append(cls_id)
        
        return boxes, confidences, class_ids
    
    def detect_with_colors(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[int], List[np.ndarray]]:
        """
        Detect persons and extract dominant colors from each detection.
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            boxes: List of bounding boxes
            confidences: List of confidence scores
            class_ids: List of class IDs
            colors: List of dominant colors (BGR)
        """
        boxes, confidences, class_ids = self.detect(frame)
        colors = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            
            if roi.size > 0:
                pixels = roi.reshape(-1, 3)
                dominant = np.mean(pixels, axis=0)
                colors.append(dominant.astype(int))
            else:
                colors.append(np.array([0, 0, 0]))
        
        return boxes, confidences, class_ids, colors
    
    def detect_batch(self, frames: List[np.ndarray]) -> List[Tuple[List[np.ndarray], List[float], List[int]]]:
        """
        Detect persons in multiple frames.
        
        Args:
            frames: List of input frames
            
        Returns:
            List of detection results for each frame
        """
        results = self.model(frames, verbose=False)
        detections = []
        
        for result in results:
            boxes = []
            confidences = []
            class_ids = []
            
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    if cls_id == self.person_class_id and conf >= self.confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        boxes.append(np.array([x1, y1, x2, y2]))
                        confidences.append(conf)
                        class_ids.append(cls_id)
            
            detections.append((boxes, confidences, class_ids))
        
        return detections


def load_video(path: str) -> cv2.VideoCapture:
    """Load video file or camera stream."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    return cap


def get_video_info(cap: cv2.VideoCapture) -> dict:
    """Get video properties."""
    return {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    }


def read_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    """Read next frame from video."""
    ret, frame = cap.read()
    return ret, frame
