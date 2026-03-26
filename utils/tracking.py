import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import time


@dataclass
class Track:
    """Track object for person tracking."""
    track_id: int
    bbox: np.ndarray
    center: Tuple[int, int]
    confidence: float
    timestamp: float
    history: List[Tuple[int, int, float]] = field(default_factory=list)
    zones_visited: List[str] = field(default_factory=list)
    first_seen: float = 0
    last_seen: float = 0
    total_idle_time: float = 0
    last_position: Optional[Tuple[int, int]] = None
    is_staff: bool = False
    
    def __post_init__(self):
        if not self.first_seen:
            self.first_seen = self.timestamp
        self.last_seen = self.timestamp
        self.history.append((*self.center, self.timestamp))
        self.last_position = self.center
    
    def update(self, bbox: np.ndarray, center: Tuple[int, int], confidence: float, timestamp: float):
        """Update track with new detection."""
        if self.last_position:
            distance = np.sqrt((center[0] - self.last_position[0])**2 + (center[1] - self.last_position[1])**2)
            if distance < 5:
                self.total_idle_time += timestamp - self.last_seen
        
        self.bbox = bbox
        self.center = center
        self.confidence = confidence
        self.last_seen = timestamp
        self.history.append((*center, timestamp))
        self.last_position = center
    
    def add_zone(self, zone: str):
        """Add visited zone."""
        if zone not in self.zones_visited:
            self.zones_visited.append(zone)
    
    def get_duration(self) -> float:
        """Get total duration track was visible."""
        return self.last_seen - self.first_seen
    
    def get_movement_frequency(self) -> float:
        """Calculate movement frequency (positions per second)."""
        duration = self.get_duration()
        if duration > 0:
            return len(self.history) / duration
        return 0


class DeepSORTTracker:
    """
    Simplified DeepSORT-style tracker using IoU matching.
    
    This is a basic implementation that uses:
    - IoU matching for association
    - Track management (creation, deletion, update)
    - Simple motion model
    """
    
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        """
        Initialize tracker.
        
        Args:
            max_age: Maximum frames to keep lost track alive
            min_hits: Minimum detections before confirming track
            iou_threshold: IOU threshold for matching
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.next_id = 1
        self.tracks: Dict[int, Track] = {}
        self.lost_tracks: Dict[int, Track] = {}
        
    def update(self, boxes: List[np.ndarray], confidences: List[float], 
               frame: np.ndarray = None, timestamp: float = None) -> Tuple[List[Track], List[np.ndarray], List[int]]:
        """
        Update tracker with new detections.
        
        Args:
            boxes: List of bounding boxes
            confidences: List of confidence scores
            frame: Current frame (for future use)
            timestamp: Current timestamp
            
        Returns:
            confirmed_tracks: List of confirmed tracks
            active_boxes: List of boxes for confirmed tracks
            active_ids: List of track IDs for confirmed tracks
        """
        if timestamp is None:
            timestamp = time.time()
        
        if not boxes:
            self._age_tracks()
            return [], [], []
        
        detections = []
        for i, box in enumerate(boxes):
            center = ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
            detections.append({
                'box': box,
                'center': center,
                'confidence': confidences[i] if i < len(confidences) else 1.0
            })
        
        matched_tracks, unmatched_dets = self._match(detections)
        
        for track_id, det_idx in matched_tracks:
            if track_id in self.tracks:
                self.tracks[track_id].update(
                    detections[det_idx]['box'],
                    detections[det_idx]['center'],
                    detections[det_idx]['confidence'],
                    timestamp
                )
                if track_id in self.lost_tracks:
                    del self.lost_tracks[track_id]
        
        for det_idx in unmatched_dets:
            self._create_track(detections[det_idx], timestamp)
        
        self._age_tracks()
        
        confirmed = []
        active_boxes = []
        active_ids = []
        
        for track_id, track in self.tracks.items():
            if track.get_duration() > 0:
                confirmed.append(track)
                active_boxes.append(track.bbox)
                active_ids.append(track.track_id)
        
        return confirmed, active_boxes, active_ids
    
    def _create_track(self, detection: dict, timestamp: float):
        """Create new track."""
        track = Track(
            track_id=self.next_id,
            bbox=detection['box'],
            center=detection['center'],
            confidence=detection['confidence'],
            timestamp=timestamp,
            first_seen=timestamp,
            last_seen=timestamp
        )
        self.tracks[self.next_id] = track
        self.next_id += 1
    
    def _match(self, detections: List[dict]) -> Tuple[List[Tuple[int, int]], List[int]]:
        """Match detections to existing tracks using IoU."""
        if not self.tracks:
            return [], list(range(len(detections)))
        
        iou_matrix = np.zeros((len(self.tracks), len(detections)))
        track_ids = list(self.tracks.keys())
        
        for i, track in enumerate(self.tracks.values()):
            for j, det in enumerate(detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, det['box'])
        
        matched = []
        unmatched_dets = list(range(len(detections)))
        
        for i, track_id in enumerate(track_ids):
            if i < len(iou_matrix) and len(iou_matrix[i]) > 0:
                max_iou = np.max(iou_matrix[i])
                if max_iou >= self.iou_threshold:
                    det_idx = np.argmax(iou_matrix[i])
                    matched.append((track_id, det_idx))
                    if det_idx in unmatched_dets:
                        unmatched_dets.remove(det_idx)
        
        return matched, unmatched_dets
    
    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculate Intersection over Union between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0
        return intersection / union
    
    def _age_tracks(self):
        """Age tracks and remove old ones."""
        to_remove = []
        
        for track_id, track in self.tracks.items():
            time_since_seen = time.time() - track.last_seen
            if time_since_seen > self.max_age / 30:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            self.lost_tracks[track_id] = self.tracks[track_id]
            del self.tracks[track_id]
    
    def get_track(self, track_id: int) -> Optional[Track]:
        """Get track by ID."""
        return self.tracks.get(track_id)
    
    def get_all_tracks(self) -> Dict[int, Track]:
        """Get all active tracks."""
        return self.tracks.copy()
    
    def get_lost_tracks(self) -> Dict[int, Track]:
        """Get all lost tracks."""
        return self.lost_tracks.copy()
    
    def reset(self):
        """Reset tracker."""
        self.tracks.clear()
        self.lost_tracks.clear()
        self.next_id = 1
