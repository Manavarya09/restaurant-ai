import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class Zone:
    """Zone definition for area-based tracking."""
    name: str
    points: List[Tuple[int, int]]
    color: Tuple[int, int, int] = (0, 255, 0)
    
    def contains(self, point: Tuple[int, int]) -> bool:
        """Check if point is inside zone."""
        result = cv2.pointPolygonTest(np.array(self.points, np.float32), point, False)
        return result >= 0
    
    def create_mask(self, frame_shape: Tuple[int, int]) -> np.ndarray:
        """Create binary mask for zone."""
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        pts = np.array(self.points, np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask


class ZoneManager:
    """Manage zones and track person movement through zones."""
    
    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        self.zone_timestamps: Dict[int, Dict[str, float]] = defaultdict(dict)
        
    def add_zone(self, name: str, points: List[Tuple[int, int]], 
                 color: Tuple[int, int, int] = (0, 255, 0)):
        """Add a zone."""
        self.zones[name] = Zone(name, points, color)
        
    def get_zone_at(self, point: Tuple[int, int]) -> Optional[str]:
        """Get zone name at point."""
        for name, zone in self.zones.items():
            if zone.contains(point):
                return name
        return None
    
    def update_track_zone(self, track_id: int, point: Tuple[int, int], timestamp: float):
        """Update track's zone history."""
        zone = self.get_zone_at(point)
        if zone:
            if zone not in self.zone_timestamps[track_id]:
                self.zone_timestamps[track_id][zone] = timestamp
    
    def get_time_in_zone(self, track_id: int, zone_name: str) -> float:
        """Get time spent in a specific zone."""
        if track_id not in self.zone_timestamps:
            return 0
        return self.zone_timestamps[track_id].get(zone_name, 0)


class FootfallCounter:
    """Count people entering and exiting through a line."""
    
    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int]):
        """
        Initialize footfall counter.
        
        Args:
            line_start: Start point of counting line
            line_end: End point of counting line
        """
        self.line_start = line_start
        self.line_end = line_end
        
        dx = line_end[0] - line_start[0]
        dy = line_end[1] - line_start[1]
        self.line_length = np.sqrt(dx**2 + dy**2)
        
        self.normal = (-dy / self.line_length, dx / self.line_length)
        
        self.entries = 0
        self.exits = 0
        
        self.track_crossings: Dict[int, bool] = {}
        
    def _crosses_line(self, prev_point: Tuple[int, int], curr_point: Tuple[int, int]) -> bool:
        """Check if line segment crosses counting line."""
        def sign(p1, p2):
            return (p1[0] - p2[0], p1[1] - p2[1])
        
        s1 = sign(curr_point, self.line_start)
        s2 = sign(prev_point, self.line_start)
        s3 = sign(self.line_end, self.line_start)
        
        cross1 = s1[0] * s2[1] - s1[1] * s2[0]
        cross2 = s3[0] * s2[1] - s3[1] * s2[0]
        
        if cross1 * cross2 < 0:
            return True
        return False
    
    def update(self, track_id: int, prev_point: Tuple[int, int], 
               curr_point: Tuple[int, int]) -> Optional[str]:
        """
        Update footfall count for a track.
        
        Returns:
            'entry', 'exit', or None
        """
        if track_id not in self.track_crossings:
            self.track_crossings[track_id] = False
            
        if self._crosses_line(prev_point, curr_point):
            if not self.track_crossings[track_id]:
                self.entries += 1
                self.track_crossings[track_id] = True
                return 'entry'
        elif self.track_crossings[track_id] and self._crosses_line(curr_point, prev_point):
            self.exits += 1
            self.track_crossings[track_id] = False
            return 'exit'
        
        return None
    
    def get_stats(self) -> Dict[str, int]:
        """Get footfall statistics."""
        return {
            'entries': self.entries,
            'exits': self.exits,
            'net': self.entries - self.exits
        }
    
    def reset(self):
        """Reset counters."""
        self.entries = 0
        self.exits = 0
        self.track_crossings.clear()


class StaffDetector:
    """Detect staff members based on color filtering."""
    
    def __init__(self, hsv_lower: Tuple[int, int, int] = (0, 0, 0), 
                 hsv_upper: Tuple[int, int, int] = (180, 255, 80)):
        """
        Initialize staff detector.
        
        Args:
            hsv_lower: Lower HSV threshold for staff color
            hsv_upper: Upper HSV threshold for staff color
        """
        self.hsv_lower = np.array(hsv_lower)
        self.hsv_upper = np.array(hsv_upper)
        
    def is_staff_color(self, bgr_color: np.ndarray) -> bool:
        """Check if color matches staff uniform."""
        hsv = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        mask = cv2.inRange(hsv, self.hsv_lower, self.hsv_upper)
        return np.sum(mask) > 0
    
    def detect_staff(self, frame: np.ndarray, boxes: List[np.ndarray]) -> List[bool]:
        """Detect which boxes contain staff."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        staff_flags = []
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = hsv[y1:y2, x1:x2]
            
            if roi.size > 0:
                mask = cv2.inRange(roi, self.hsv_lower, self.hsv_upper)
                staff_flags.append(np.sum(mask) > (roi.shape[0] * roi.shape[1] * 0.3))
            else:
                staff_flags.append(False)
        
        return staff_flags


class WaitTimeAnalyzer:
    """Analyze customer wait times."""
    
    def __init__(self):
        self.wait_start: Dict[int, float] = {}
        self.wait_end: Dict[int, float] = {}
        self.wait_times: List[float] = []
        
    def start_wait(self, track_id: int, timestamp: float):
        """Record wait start time."""
        self.wait_start[track_id] = timestamp
        
    def end_wait(self, track_id: int, timestamp: float):
        """Record wait end time and calculate wait time."""
        if track_id in self.wait_start:
            wait_time = timestamp - self.wait_start[track_id]
            self.wait_end[track_id] = timestamp
            self.wait_times.append(wait_time)
            del self.wait_start[track_id]
            return wait_time
        return None
    
    def get_average_wait_time(self) -> float:
        """Get average wait time."""
        if self.wait_times:
            return np.mean(self.wait_times)
        return 0
    
    def get_wait_time_distribution(self) -> Dict[str, float]:
        """Get wait time distribution statistics."""
        if not self.wait_times:
            return {'min': 0, 'max': 0, 'mean': 0, 'median': 0, 'std': 0}
        
        return {
            'min': np.min(self.wait_times),
            'max': np.max(self.wait_times),
            'mean': np.mean(self.wait_times),
            'median': np.median(self.wait_times),
            'std': np.std(self.wait_times)
        }


class StaffEfficiencyAnalyzer:
    """Analyze staff efficiency metrics."""
    
    def __init__(self):
        self.staff_tracks: Dict[int, Dict] = {}
        
    def update_staff(self, track_id: int, center: Tuple[int, int], 
                     timestamp: float, is_staff: bool):
        """Update staff track information."""
        if is_staff:
            if track_id not in self.staff_tracks:
                self.staff_tracks[track_id] = {
                    'positions': [],
                    'idle_time': 0,
                    'last_position': center,
                    'last_timestamp': timestamp,
                    'interactions': 0
                }
            
            staff = self.staff_tracks[track_id]
            staff['positions'].append((*center, timestamp))
            
            if staff['last_position']:
                dist = np.sqrt((center[0] - staff['last_position'][0])**2 + 
                             (center[1] - staff['last_position'][1])**2)
                if dist < 10:
                    staff['idle_time'] += timestamp - staff['last_timestamp']
            
            staff['last_position'] = center
            staff['last_timestamp'] = timestamp
    
    def calculate_interactions(self, customer_tracks: Dict, 
                               staff_track_id: int, distance_threshold: int = 100) -> int:
        """Calculate approximate customer interactions."""
        if staff_track_id not in self.staff_tracks:
            return 0
        
        staff = self.staff_tracks[staff_track_id]
        if not staff['positions']:
            return 0
        
        staff_pos = staff['positions'][-1][:2]
        interactions = 0
        
        for cust_id, cust_track in customer_tracks.items():
            if cust_track.history:
                cust_pos = cust_track.history[-1][:2]
                dist = np.sqrt((staff_pos[0] - cust_pos[0])**2 + 
                             (staff_pos[1] - cust_pos[1])**2)
                if dist < distance_threshold:
                    interactions += 1
        
        return interactions
    
    def get_efficiency_metrics(self) -> Dict:
        """Get overall staff efficiency metrics."""
        if not self.staff_tracks:
            return {}
        
        metrics = {
            'total_staff': len(self.staff_tracks),
            'avg_idle_time': 0,
            'total_movements': 0
        }
        
        idle_times = [s['idle_time'] for s in self.staff_tracks.values()]
        if idle_times:
            metrics['avg_idle_time'] = np.mean(idle_times)
        
        metrics['total_movements'] = sum(
            len(s['positions']) for s in self.staff_tracks.values()
        )
        
        return metrics
