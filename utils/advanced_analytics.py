"""
Advanced Analytics Module for Restaurant AI
Includes: Heatmaps, Dwell Time, Crowd Analysis, Behavior Detection
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from dataclasses import dataclass
import time


@dataclass
class CustomerProfile:
    """Customer behavior profile."""
    track_id: int
    entry_time: float
    zones_visited: List[str]
    zone_times: Dict[str, float]
    total_dwell: float
    is_vip: bool
    group_id: Optional[int]
    repeat_visit: bool


class HeatmapGenerator:
    """Generate customer movement heatmaps."""
    
    def __init__(self, frame_shape: Tuple[int, int], bins: int = 20):
        self.frame_shape = frame_shape
        self.bins = bins
        self.heatmap = np.zeros((bins, bins), dtype=np.float32)
        self.frame_count = 0
        
    def update(self, positions: List[Tuple[int, int]]):
        """Update heatmap with new positions."""
        h, w = self.frame_shape[:2]
        bin_h = h // self.bins
        bin_w = w // self.bins
        
        for x, y in positions:
            bx = min(x // bin_w, self.bins - 1)
            by = min(y // bin_h, self.bins - 1)
            self.heatmap[by, bx] += 1
        
        self.frame_count += 1
        
    def get_heatmap(self, apply_blur: bool = True) -> np.ndarray:
        """Get normalized heatmap."""
        if self.heatmap.max() > 0:
            normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
        else:
            normalized = np.zeros_like(self.heatmap, dtype=np.uint8)
            
        if apply_blur:
            normalized = cv2.GaussianBlur(normalized, (21, 21), 0)
            
        return normalized
    
    def apply_to_frame(self, frame: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
        """Apply heatmap overlay to frame."""
        heatmap = self.get_heatmap()
        heatmap_colored = cv2.applyColorMap(heatmap, colormap)
        
        h, w = frame.shape[:2]
        heatmap_resized = cv2.resize(heatmap_colored, (w, h))
        
        overlay = cv2.addWeighted(frame, 0.5, heatmap_resized, 0.5, 0)
        return overlay
    
    def get_hotspots(self, threshold: float = 0.7) -> List[Tuple[int, int]]:
        """Get hotspot coordinates."""
        max_val = self.heatmap.max()
        if max_val == 0:
            return []
            
        threshold_val = max_val * threshold
        h, w = self.frame_shape[:2]
        bin_h = h // self.bins
        bin_w = w // self.bins
        
        hotspots = []
        for y in range(self.bins):
            for x in range(self.bins):
                if self.heatmap[y, x] >= threshold_val:
                    cx = x * bin_w + bin_w // 2
                    cy = y * bin_h + bin_h // 2
                    hotspots.append((cx, cy))
                    
        return hotspots


class DwellTimeAnalyzer:
    """Analyze customer dwell time in zones."""
    
    def __init__(self):
        self.customer_timers: Dict[int, Dict[str, float]] = {}
        self.dwell_times: Dict[str, List[float]] = defaultdict(list)
        
    def start_tracking(self, track_id: int):
        """Start tracking a customer."""
        self.customer_timers[track_id] = {
            'start_time': time.time(),
            'zone_entry': {}
        }
        
    def enter_zone(self, track_id: int, zone: str):
        """Record zone entry."""
        if track_id not in self.customer_timers:
            self.start_tracking(track_id)
            
        if zone not in self.customer_timers[track_id]['zone_entry']:
            self.customer_timers[track_id]['zone_entry'][zone] = time.time()
            
    def exit_zone(self, track_id: int, zone: str):
        """Record zone exit and calculate dwell time."""
        if track_id not in self.customer_timers:
            return
            
        if zone in self.customer_timers[track_id]['zone_entry']:
            entry_time = self.customer_timers[track_id]['zone_entry'][zone]
            dwell = time.time() - entry_time
            self.dwell_times[zone].append(dwell)
            del self.customer_timers[track_id]['zone_entry'][zone]
            
    def get_average_dwell(self, zone: str) -> float:
        """Get average dwell time for zone."""
        if zone in self.dwell_times and self.dwell_times[zone]:
            return np.mean(self.dwell_times[zone])
        return 0.0
    
    def get_dwell_distribution(self, zone: str) -> Dict[str, float]:
        """Get dwell time distribution."""
        if zone not in self.dwell_times or not self.dwell_times[zone]:
            return {}
            
        times = self.dwell_times[zone]
        return {
            'min': np.min(times),
            'max': np.max(times),
            'mean': np.mean(times),
            'median': np.median(times),
            'std': np.std(times)
        }


class CrowdDensityAnalyzer:
    """Analyze crowd density and detect overcrowding."""
    
    def __init__(self, zone_polygon: List[Tuple[int, int]], 
                 capacity: int, alert_threshold: float = 0.8):
        self.zone_polygon = zone_polygon
        self.capacity = capacity
        self.alert_threshold = alert_threshold
        self.density_history: List[float] = []
        
    def calculate_density(self, positions: List[Tuple[int, int]]) -> float:
        """Calculate current density."""
        import cv2
        
        count = 0
        for x, y in positions:
            if cv2.pointPolygonTest(np.array(self.zone_polygon, np.float32), 
                                   (x, y), False) >= 0:
                count += 1
                
        density = count / self.capacity if self.capacity > 0 else 0
        self.density_history.append(density)
        
        return density
    
    def is_overcrowded(self, density: float = None) -> bool:
        """Check if zone is overcrowded."""
        if density is None:
            if not self.density_history:
                return False
            density = self.density_history[-1]
            
        return density >= self.alert_threshold
    
    def get_trend(self, window: int = 10) -> str:
        """Get density trend."""
        if len(self.density_history) < 2:
            return 'stable'
            
        recent = self.density_history[-window:]
        if len(recent) < 2:
            return 'stable'
            
        if recent[-1] > recent[0] * 1.2:
            return 'increasing'
        elif recent[-1] < recent[0] * 0.8:
            return 'decreasing'
        return 'stable'


class BehaviorAnalyzer:
    """Analyze customer behavior patterns."""
    
    def __init__(self):
        self.behavior_patterns: Dict[int, List[str]] = {}
        self.customer_profiles: Dict[int, CustomerProfile] = {}
        
    def analyze_movement(self, track_id: int, positions: List[Tuple[int, int]]) -> str:
        """Analyze movement pattern."""
        if len(positions) < 2:
            return 'unknown'
            
        displacements = []
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            displacements.append(np.sqrt(dx**2 + dy**2))
            
        avg_speed = np.mean(displacements)
        
        if avg_speed < 2:
            return 'stationary'
        elif avg_speed < 10:
            return 'walking'
        elif avg_speed < 30:
            return 'running'
        return 'unknown'
    
    def detect_waiting_pattern(self, track_id: int, zone_history: List[str]) -> bool:
        """Detect if customer is waiting."""
        if len(zone_history) < 3:
            return False
            
        recent_zones = zone_history[-3:]
        return 'waiting' in recent_zones and 'dining' not in recent_zones
    
    def detect_browsing(self, track_id: int, zone_history: List[str]) -> bool:
        """Detect browsing behavior."""
        unique_zones = len(set(zone_history))
        return unique_zones >= 3
    
    def create_profile(self, track_id: int, entry_time: float, 
                     zones: List[str], dwell_times: Dict[str, float]) -> CustomerProfile:
        """Create customer profile."""
        profile = CustomerProfile(
            track_id=track_id,
            entry_time=entry_time,
            zones_visited=zones,
            zone_times=dwell_times,
            total_dwell=sum(dwell_times.values()),
            is_vip=False,
            group_id=None,
            repeat_visit=False
        )
        self.customer_profiles[track_id] = profile
        return profile


class GroupDetector:
    """Detect groups of customers."""
    
    def __init__(self, distance_threshold: int = 150):
        self.distance_threshold = distance_threshold
        self.group_history: Dict[int, List[int]] = {}
        
    def detect_groups(self, positions: List[Tuple[int, int]], 
                     track_ids: List[int]) -> Dict[int, List[int]]:
        """Detect groups based on proximity."""
        groups = {}
        used = set()
        
        for i, (pos1, id1) in enumerate(zip(positions, track_ids)):
            if id1 in used:
                continue
                
            group = [id1]
            used.add(id1)
            
            for j, (pos2, id2) in enumerate(zip(positions, track_ids)):
                if id2 in used or id1 == id2:
                    continue
                    
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                if dist < self.distance_threshold:
                    group.append(id2)
                    used.add(id2)
            
            if len(group) > 1:
                groups[min(group)] = group
                
        return groups
    
    def is_family_group(self, group_ids: List[int], 
                       zone_history: Dict[int, List[str]]) -> bool:
        """Heuristic to detect family groups."""
        if len(group_ids) < 3:
            return False
            
        return True


class PredictiveAnalytics:
    """Predictive analytics for staffing and capacity."""
    
    def __init__(self):
        self.historical_data: Dict[str, List[float]] = defaultdict(list)
        
    def add_data_point(self, metric: str, value: float):
        """Add historical data point."""
        self.historical_data[metric].append(value)
        
    def predict_peak(self, hour: int) -> float:
        """Predict customer count for given hour."""
        key = f"hour_{hour}"
        if key in self.historical_data and self.historical_data[key]:
            return np.mean(self.historical_data[key])
        return 0.0
    
    def recommend_staff(self, predicted_customers: int, 
                       target_ratio: float = 3.0) -> int:
        """Recommend staff count."""
        return max(1, int(predicted_customers / target_ratio))
    
    def get_capacity_alert(self, current: int, capacity: int) -> str:
        """Get capacity alert level."""
        ratio = current / capacity if capacity > 0 else 0
        
        if ratio >= 0.9:
            return "critical"
        elif ratio >= 0.7:
            return "warning"
        elif ratio >= 0.5:
            return "moderate"
        return "normal"


class ReportGenerator:
    """Generate analytics reports."""
    
    def __init__(self):
        self.data = {}
        
    def add_section(self, name: str, data: dict):
        """Add report section."""
        self.data[name] = data
        
    def generate_summary(self) -> str:
        """Generate text summary."""
        lines = ["=" * 50, "RESTAURANT ANALYTICS REPORT", "=" * 50, ""]
        
        for section, data in self.data.items():
            lines.append(f"\n{section.upper()}")
            lines.append("-" * 30)
            for key, value in data.items():
                lines.append(f"  {key}: {value}")
                
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export as dictionary."""
        return self.data.copy()
    
    def to_json(self) -> str:
        """Export as JSON."""
        import json
        return json.dumps(self.data, indent=2)
