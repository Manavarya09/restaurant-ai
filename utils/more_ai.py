"""
Advanced AI Features - More Intelligence
Includes: Gesture Recognition, ReID, Multi-camera, Service Quality, Voice Commands
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from collections import deque, defaultdict
import time
import json
import base64


@dataclass
class CustomerJourney:
    """Complete customer journey tracking."""
    track_id: int
    entry_time: float
    exit_time: Optional[float]
    zones_sequence: List[str] = field(default_factory=list)
    zone_durations: Dict[str, float] = field(default_factory=dict)
    interactions: List[Dict] = field(default_factory=list)
    wait_time_total: float = 0.0
    satisfaction_score: float = 0.0
    is_vip: bool = False
    repeat_customer: bool = False


class GestureRecognizer:
    """Recognize customer gestures."""
    
    GESTURES = {
        'waving': {'motion': 'horizontal', 'duration': (0.5, 2.0)},
        'pointing': {'motion': 'directional', 'duration': (0.3, 1.5)},
        'checking_phone': {'motion': 'downward', 'duration': (1.0, 5.0)},
        'looking_around': {'motion': 'oscillating', 'duration': (0.5, 2.0)},
        'hand_up': {'motion': 'upward', 'duration': (0.3, 1.0)},
        'shaking_head': {'motion': 'side_to_side', 'duration': (0.5, 2.0)},
        'nodding': {'motion': 'up_down', 'duration': (0.3, 1.5)}
    }
    
    def __init__(self):
        self.gesture_history = {}
        
    def detect_hand_gesture(self, hand_positions: List[Tuple[int, int]], 
                           timestamps: List[float]) -> str:
        """Detect hand gesture from position history."""
        if len(hand_positions) < 5:
            return "unknown"
            
        positions = np.array(hand_positions)
        
        dx = np.diff(positions[:, 0])
        dy = np.diff(positions[:, 1])
        
        horizontal_motion = np.sum(np.abs(dx))
        vertical_motion = np.sum(np.abs(dy))
        
        if horizontal_motion > vertical_motion * 2:
            if np.std(dx) < 10:
                return "waving"
            return "pointing"
        elif vertical_motion > horizontal_motion * 2:
            if np.mean(dy) < 0:
                return "hand_up"
            return "checking_phone"
        else:
            return "looking_around"
    
    def detect_gesture_from_motion(self, positions: List[Tuple[int, int]], 
                                  timestamps: List[float]) -> Dict:
        """Detect gesture from body motion."""
        if len(positions) < 3:
            return {'gesture': 'unknown', 'confidence': 0.0}
            
        pos_array = np.array(positions)
        
        displacements = []
        for i in range(1, len(pos_array)):
            dist = np.sqrt((pos_array[i][0] - pos_array[i-1][0])**2 + 
                          (pos_array[i][1] - pos_array[i-1][1])**2)
            displacements.append(dist)
        
        avg_displacement = np.mean(displacements)
        
        gesture = "standing"
        if avg_displacement > 30:
            gesture = "running"
        elif avg_displacement > 10:
            gesture = "walking"
        elif avg_displacement > 2:
            gesture = "looking_around"
            
        return {
            'gesture': gesture,
            'confidence': 0.8,
            'velocity': avg_displacement
        }


class ReIDSystem:
    """Person Re-Identification for multi-camera tracking."""
    
    def __init__(self):
        self.features_db = {}
        self.track_history = {}
        
    def extract_features(self, bbox: np.ndarray, frame: np.ndarray) -> np.ndarray:
        """Extract ReID features from person."""
        x1, y1, x2, y2 = map(int, bbox)
        
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return np.zeros(128)
        
        roi = cv2.resize(roi, (64, 128))
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        features = []
        
        for i in range(8):
            for j in range(4):
                patch = gray[i*16:(i+1)*16, j*16:(j+1)*16]
                features.append(np.mean(patch))
                features.append(np.std(patch))
        
        features = np.array(features[:128])
        
        if np.linalg.norm(features) > 0:
            features = features / np.linalg.norm(features)
            
        return features
    
    def match_person(self, features: np.ndarray, threshold: float = 0.7) -> Optional[int]:
        """Match against known persons."""
        best_match = None
        best_similarity = 0
        
        for track_id, stored_features in self.features_db.items():
            similarity = np.dot(features, stored_features)
            
            if similarity > best_similarity and similarity > threshold:
                best_similarity = similarity
                best_match = track_id
                
        return best_match
    
    def update_database(self, track_id: int, features: np.ndarray):
        """Update feature database."""
        if track_id not in self.features_db:
            self.features_db[track_id] = features
        else:
            self.features_db[track_id] = (self.features_db[track_id] * 0.7 + features * 0.3)


class ServiceQualityAnalyzer:
    """Analyze service quality metrics."""
    
    def __init__(self):
        self.service_events = []
        self.customer_feedback = {}
        
    def record_interaction(self, customer_id: int, staff_id: int, 
                          interaction_type: str, duration: float):
        """Record customer-staff interaction."""
        event = {
            'customer_id': customer_id,
            'staff_id': staff_id,
            'type': interaction_type,
            'duration': duration,
            'timestamp': time.time()
        }
        self.service_events.append(event)
        
    def calculate_service_score(self, customer_id: int) -> float:
        """Calculate service score for customer."""
        events = [e for e in self.service_events if e['customer_id'] == customer_id]
        
        if not events:
            return 5.0
            
        score = 5.0
        
        for event in events:
            if event['duration'] > 300:
                score -= 1.0
            elif event['duration'] < 60:
                score += 0.5
                
            if event['type'] == 'greeting':
                score += 1.0
            elif event['type'] == 'complaint':
                score -= 2.0
                
        return max(1.0, min(10.0, score))
    
    def get_staff_performance(self, staff_id: int) -> Dict:
        """Get staff performance metrics."""
        events = [e for e in self.service_events if e['staff_id'] == staff_id]
        
        if not events:
            return {'score': 0, 'interactions': 0, 'avg_duration': 0}
            
        total_duration = sum(e['duration'] for e in events)
        
        return {
            'score': np.mean([self.calculate_service_score(e['customer_id']) for e in events]),
            'interactions': len(events),
            'avg_duration': total_duration / len(events),
            'greetings': sum(1 for e in events if e['type'] == 'greeting'),
            'complaints': sum(1 for e in events if e['type'] == 'complaint')
        }
    
    def get_service_metrics(self) -> Dict:
        """Get overall service metrics."""
        if not self.service_events:
            return {}
            
        total_events = len(self.service_events)
        avg_duration = np.mean([e['duration'] for e in self.service_events])
        
        return {
            'total_interactions': total_events,
            'avg_interaction_duration': avg_duration,
            'interactions_per_hour': total_events / max(1, (time.time() - self.service_events[0]['timestamp']) / 3600),
            'greeting_rate': sum(1 for e in self.service_events if e['type'] == 'greeting') / total_events,
            'complaint_rate': sum(1 for e in self.service_events if e['type'] == 'complaint') / total_events
        }


class VoiceCommandProcessor:
    """Process voice commands (simulated)."""
    
    COMMANDS = {
        'start_tracking': ['start', 'begin', 'track'],
        'stop_tracking': ['stop', 'end', 'pause'],
        'show_stats': ['stats', 'statistics', 'metrics'],
        'show_heatmap': ['heatmap', 'heat', 'map'],
        'reset_counts': ['reset', 'clear'],
        'export_data': ['export', 'save', 'download'],
        'switch_camera': ['camera', 'switch', 'view'],
        'fullscreen': ['full', 'screen', 'maximize']
    }
    
    def __init__(self):
        self.command_history = []
        
    def process_text_command(self, text: str) -> Tuple[str, Dict]:
        """Process text command."""
        text = text.lower()
        
        for cmd, keywords in self.COMMANDS.items():
            if any(kw in text for kw in keywords):
                self.command_history.append({
                    'command': cmd,
                    'input': text,
                    'timestamp': time.time()
                })
                return cmd, self._extract_parameters(text)
                
        return 'unknown', {}
    
    def _extract_parameters(self, text: str) -> Dict:
        """Extract parameters from command."""
        params = {}
        
        if 'camera' in text:
            for i in range(5):
                if str(i) in text:
                    params['camera_id'] = i
                    
        if 'export' in text:
            if 'json' in text:
                params['format'] = 'json'
            elif 'csv' in text:
                params['format'] = 'csv'
            elif 'excel' in text or 'xlsx' in text:
                params['format'] = 'xlsx'
                
        return params
    
    def get_command_history(self) -> List[Dict]:
        """Get command history."""
        return self.command_history


class SocialDistanceMonitor:
    """Monitor social distancing compliance."""
    
    def __init__(self, min_distance: float = 100.0):
        self.min_distance = min_distance
        self.violations = []
        
    def check_distances(self, positions: List[Tuple[int, int]], 
                       track_ids: List[int]) -> List[Dict]:
        """Check distances between all persons."""
        violations = []
        
        for i, (pos1, id1) in enumerate(zip(positions, track_ids)):
            for j, (pos2, id2) in enumerate(zip(positions, track_ids)):
                if i >= j:
                    continue
                    
                dist = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                
                if dist < self.min_distance:
                    violations.append({
                        'track_id_1': id1,
                        'track_id_2': id2,
                        'distance': dist,
                        'timestamp': time.time()
                    })
                    
        self.violations.extend(violations)
        return violations
    
    def get_compliance_rate(self, total_persons: int) -> float:
        """Calculate compliance rate."""
        if total_persons < 2:
            return 1.0
            
        max_pairs = total_persons * (total_persons - 1) / 2
        
        recent_violations = [v for v in self.violations 
                           if time.time() - v['timestamp'] < 300]
        
        return 1.0 - len(recent_violations) / max(2, max_pairs)
    
    def get_violation_count(self, window_seconds: int = 60) -> int:
        """Get violation count in time window."""
        cutoff = time.time() - window_seconds
        return sum(1 for v in self.violations if v['timestamp'] > cutoff)


class CustomerChurnPredictor:
    """Predict customer churn/satisfaction."""
    
    def __init__(self):
        self.customer_profiles = {}
        
    def update_profile(self, customer_id: int, data: Dict):
        """Update customer profile."""
        if customer_id not in self.customer_profiles:
            self.customer_profiles[customer_id] = {
                'visits': [],
                'wait_times': [],
                'interactions': [],
                'signals': []
            }
        
        profile = self.customer_profiles[customer_id]
        
        if 'wait_time' in data:
            profile['wait_times'].append(data['wait_time'])
        if 'interaction' in data:
            profile['interactions'].append(data['interaction'])
        if 'signal' in data:
            profile['signals'].append(data['signal'])
            
        profile['visits'].append(time.time())
        
    def predict_churn_risk(self, customer_id: int) -> Tuple[str, float]:
        """Predict churn risk."""
        if customer_id not in self.customer_profiles:
            return 'unknown', 0.0
            
        profile = self.customer_profiles[customer_id]
        
        risk_score = 0.0
        
        wait_times = profile.get('wait_times', [])
        if wait_times:
            avg_wait = np.mean(wait_times[-5:])
            if avg_wait > 600:
                risk_score += 0.4
            elif avg_wait > 300:
                risk_score += 0.2
                
        signals = profile.get('signals', [])
        negative_signals = sum(1 for s in signals if s in ['left_angry', 'complaint', 'short_interaction'])
        risk_score += min(0.4, negative_signals * 0.1)
        
        visit_frequency = len(profile.get('visits', []))
        if visit_frequency > 10 and risk_score > 0.3:
            risk_score *= 0.5
            
        if risk_score > 0.6:
            risk_level = 'high'
        elif risk_score > 0.3:
            risk_level = 'medium'
        else:
            risk_level = 'low'
            
        return risk_level, risk_score
    
    def get_at_risk_customers(self) -> List[int]:
        """Get list of at-risk customers."""
        at_risk = []
        
        for customer_id in self.customer_profiles:
            risk_level, _ = self.predict_churn_risk(customer_id)
            if risk_level in ['high', 'medium']:
                at_risk.append(customer_id)
                
        return at_risk


class MultiCameraManager:
    """Manage multiple camera feeds."""
    
    def __init__(self):
        self.cameras = {}
        self.camera_positions = {}
        
    def add_camera(self, camera_id: str, source, position: str = None):
        """Add camera to system."""
        self.cameras[camera_id] = {
            'source': source,
            'position': position,
            'active': False,
            'frame_count': 0
        }
        
    def get_camera(self, camera_id: str) -> Optional[Dict]:
        """Get camera info."""
        return self.cameras.get(camera_id)
    
    def list_cameras(self) -> List[str]:
        """List all camera IDs."""
        return list(self.cameras.keys())
    
    def get_camera_position(self, camera_id: str) -> Optional[str]:
        """Get camera position."""
        cam = self.cameras.get(camera_id)
        return cam['position'] if cam else None
    
    def merge_tracks(self, track_a: Dict, track_b: Dict) -> Optional[int]:
        """Merge tracks from different cameras (same person)."""
        if track_a.get('features') is not None and track_b.get('features') is not None:
            similarity = np.dot(track_a['features'], track_b['features'])
            if similarity > 0.8:
                return track_a.get('id')
        return None


class ObjectDetector:
    """Detect objects in addition to persons."""
    
    OBJECTS = {
        'bag': {'class_id': 0, 'colors': [(120, 100, 50), (80, 80, 80)]},
        'stroller': {'class_id': 1, 'colors': [(200, 200, 200)]},
        'wheelchair': {'class_id': 2, 'colors': [(150, 150, 150)]},
        'shopping_cart': {'class_id': 3, 'colors': [(180, 170, 150)]},
        'food_tray': {'class_id': 4, 'colors': [(200, 200, 200), (150, 150, 150)]},
        'phone': {'class_id': 5, 'colors': [(50, 50, 50), (100, 100, 100)]}
    }
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects in frame."""
        detected = []
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        for obj_name, obj_info in self.OBJECTS.items():
            for color in obj_info['colors']:
                lower = np.array([max(0, c - 30) for c in color])
                upper = np.array([min(180, c + 30) for c in color])
                
                mask = cv2.inRange(hsv, lower, upper)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 500:
                        x, y, w, h = cv2.boundingRect(contour)
                        detected.append({
                            'object': obj_name,
                            'bbox': (x, y, x + w, y + h),
                            'confidence': 0.7
                        })
                        
        return detected


class HeatmapPredictor:
    """Predict future heatmap based on historical data."""
    
    def __init__(self):
        self.historical_maps = []
        
    def add_heatmap_snapshot(self, heatmap: np.ndarray, timestamp: float):
        """Add heatmap to history."""
        self.historical_maps.append({
            'heatmap': heatmap,
            'timestamp': timestamp,
            'hour': timestamp % 86400 / 3600
        })
        
    def predict_heatmap(self, target_hour: float) -> np.ndarray:
        """Predict heatmap for target hour."""
        if not self.historical_maps:
            return np.zeros((10, 10))
            
        relevant = [m for m in self.historical_maps 
                   if abs(m['hour'] - target_hour) < 2]
        
        if not relevant:
            relevant = self.historical_maps[-5:]
            
        avg_heatmap = np.mean([m['heatmap'] for m in relevant], axis=0)
        
        return avg_heatmap
    
    def get_hotspot_prediction(self, target_hour: float) -> List[Tuple[int, int]]:
        """Predict hotspots for target hour."""
        heatmap = self.predict_heatmap(target_hour)
        
        h, w = heatmap.shape
        threshold = np.percentile(heatmap, 90)
        
        hotspots = []
        for y in range(h):
            for x in range(w):
                if heatmap[y, x] >= threshold:
                    hotspots.append((x * 10, y * 10))
                    
        return hotspots


class SentimentAggregator:
    """Aggregate sentiment across all customers."""
    
    def __init__(self):
        self.sentiment_history = defaultdict(lambda: deque(maxlen=1000))
        
    def add_sentiment(self, track_id: int, sentiment: str, confidence: float):
        """Add sentiment reading."""
        self.sentiment_history[track_id].append({
            'sentiment': sentiment,
            'confidence': confidence,
            'timestamp': time.time()
        })
    
    def get_current_sentiment(self, track_id: int) -> Tuple[str, float]:
        """Get current sentiment for track."""
        if track_id not in self.sentiment_history or not self.sentiment_history[track_id]:
            return 'neutral', 0.0
            
        readings = list(self.sentiment_history[track_id])
        
        sentiments = [r['sentiment'] for r in readings[-10:]]
        
        counts = {}
        for s in sentiments:
            counts[s] = counts.get(s, 0) + 1
            
        dominant = max(counts.items(), key=lambda x: x[1])
        confidence = dominant[1] / len(sentiments)
        
        return dominant[0], confidence
    
    def get_overall_sentiment(self) -> Dict:
        """Get overall sentiment across all tracks."""
        all_sentiments = []
        
        for track_id, history in self.sentiment_history.items():
            if history:
                current = history[-1]
                all_sentiments.append(current['sentiment'])
                
        if not all_sentiments:
            return {'sentiment': 'neutral', 'score': 0.5, 'distribution': {}}
            
        counts = {}
        for s in all_sentiments:
            counts[s] = counts.get(s, 0) + 1
            
        total = len(all_sentiments)
        
        positive = counts.get('happy', 0) / total
        negative = counts.get('sad', 0) / total + counts.get('angry', 0) / total
        
        score = 0.5 + positive * 0.3 - negative * 0.3
        
        return {
            'sentiment': max(counts.items(), key=lambda x: x[1])[0],
            'score': score,
            'distribution': {k: v/total for k, v in counts.items()},
            'total_tracked': len(self.sentiment_history)
        }
