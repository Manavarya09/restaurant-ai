import cv2
import numpy as np
from typing import List, Tuple
from pathlib import Path


def save_annotated_video(input_path: str, output_path: str, 
                         detections_list: List[Tuple[List[np.ndarray], List[int], List[float]]],
                         zones: dict = None,
                         fps: int = 30):
    """
    Save annotated video with bounding boxes.
    
    Args:
        input_path: Input video path
        output_path: Output video path
        detections_list: List of (boxes, ids, confidences) for each frame
        zones: Optional dict of zone polygons
        fps: Output FPS
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {input_path}")
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx < len(detections_list):
            boxes, ids, confidences = detections_list[frame_idx]
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                label = f"ID:{ids[i]}" if i < len(ids) else ""
                if i < len(confidences):
                    label += f" {confidences[i]:.2f}"
                
                cv2.putText(frame, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        out.write(frame)
        frame_idx += 1
    
    cap.release()
    out.release()
    return output_path


def create_heatmap(track_histories: dict, frame_shape: Tuple[int, int], 
                   bins: int = 20) -> np.ndarray:
    """
    Create heatmap of customer movement.
    
    Args:
        track_histories: Dict of track_id -> list of (x, y, timestamp)
        frame_shape: (height, width)
        bins: Number of bins for histogram
    
    Returns:
        Heatmap array
    """
    heatmap = np.zeros(frame_shape[:2], dtype=np.float32)
    height, width = frame_shape[:2]
    
    bin_size_y = height // bins
    bin_size_x = width // bins
    
    for track_id, history in track_histories.items():
        for x, y, _ in history:
            by = min(y // bin_size_y, bins - 1)
            bx = min(x // bin_size_x, bins - 1)
            heatmap[by * bin_size_y:(by + 1) * bin_size_y, 
                   bx * bin_size_x:(bx + 1) * bin_size_x] += 1
    
    heatmap = cv2.GaussianBlur(heatmap, (21, 21), 0)
    heatmap = (heatmap / heatmap.max() * 255).astype(np.uint8)
    
    return heatmap


def apply_heatmap_to_frame(frame: np.ndarray, heatmap: np.ndarray, 
                          colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """Apply heatmap overlay to frame."""
    heatmap_colored = cv2.applyColorMap(heatmap, colormap)
    output = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
    return output


class QueueDetector:
    """Detect queue/crowd buildup."""
    
    def __init__(self, zone_points: List[Tuple[int, int]], 
                 density_threshold: float = 0.5):
        """
        Initialize queue detector.
        
        Args:
            zone_points: Polygon points defining the queue zone
            density_threshold: Person density threshold for queue alert
        """
        self.zone_points = zone_points
        self.density_threshold = density_threshold
        self.zone_area = self._calculate_zone_area()
        
    def _calculate_zone_area(self) -> float:
        """Calculate zone area using polygon."""
        pts = np.array(self.zone_points, np.float32)
        return cv2.contourArea(pts)
    
    def detect_queue(self, person_positions: List[Tuple[int, int]], 
                    frame_shape: Tuple[int, int]) -> Tuple[bool, float]:
        """
        Detect if queue has formed.
        
        Args:
            person_positions: List of person center positions
            frame_shape: Frame shape (height, width)
            
        Returns:
            (is_queue, density)
        """
        import cv2
        
        persons_in_zone = 0
        for x, y in person_positions:
            if cv2.pointPolygonTest(np.array(self.zone_points, np.float32), 
                                    (x, y), False) >= 0:
                persons_in_zone += 1
        
        if self.zone_area <= 0:
            return False, 0.0
        
        frame_area = frame_shape[0] * frame_shape[1]
        density = (persons_in_zone * 10000) / self.zone_area
        
        is_queue = density > self.density_threshold
        return is_queue, density
    
    def get_queue_length(self, person_positions: List[Tuple[int, int]]) -> int:
        """Estimate queue length based on positions."""
        if not person_positions:
            return 0
        
        import cv2
        in_zone = [p for p in person_positions 
                  if cv2.pointPolygonTest(np.array(self.zone_points, np.float32), 
                                         p, False) >= 0]
        return len(in_zone)
