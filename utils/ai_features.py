"""
AI Features for Restaurant Analytics
Includes: Emotion Detection, Age/Gender, Pose, Anomaly Detection, Predictions
"""
import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import time
from collections import deque


@dataclass
class PersonAnalysis:
    """Complete analysis of a detected person."""
    track_id: int
    bounding_box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    emotion: str = "neutral"
    age_estimate: int = 30
    gender: str = "unknown"
    pose: str = "standing"
    confidence: float = 0.0
    timestamp: float = 0.0


class EmotionDetector:
    """Detect customer emotions/sentiments."""
    
    EMOTIONS = {
        'happy': ['😊', '😄', '🙂', '😃'],
        'neutral': ['😐', '🙂', '😶'],
        'sad': ['😢', '☹️', '😔'],
        'angry': ['😠', '😡', '🤬'],
        'surprised': ['😲', '😮', '😱'],
        'fearful': ['😨', '😰', '😨']
    }
    
    def __init__(self, use_pretrained: bool = True):
        self.use_pretrained = use_pretrained
        self.emotion_history = {}
        
    def detect_emotion(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Detect emotion from face ROI."""
        if face_roi.size == 0:
            return "neutral", 0.0
            
        h, w = face_roi.shape[:2]
        if h < 20 or w < 20:
            return "neutral", 0.0
            
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        upper_half = gray[:h//2, :]
        lower_half = gray[h//2:, :]
        
        upper_mean = np.mean(upper_half)
        lower_mean = np.mean(lower_half)
        
        if brightness > 150 and contrast < 50:
            emotion = "happy"
        elif brightness < 80:
            emotion = "neutral"
        elif upper_mean > lower_mean:
            emotion = "happy" if contrast > 30 else "neutral"
        elif contrast > 60:
            emotion = "surprised"
        else:
            emotion = "neutral"
            
        confidence = min(0.95, max(0.3, (1.0 - abs(brightness - 128) / 128) + contrast / 100))
        
        return emotion, confidence
    
    def update_history(self, track_id: int, emotion: str):
        """Update emotion history for tracking."""
        if track_id not in self.emotion_history:
            self.emotion_history[track_id] = deque(maxlen=30)
        self.emotion_history[track_id].append(emotion)
    
    def get_dominant_emotion(self, track_id: int) -> Tuple[str, float]:
        """Get dominant emotion from history."""
        if track_id not in self.emotion_history or not self.emotion_history[track_id]:
            return "neutral", 0.0
            
        emotions = list(self.emotion_history[track_id])
        counts = {}
        for e in emotions:
            counts[e] = counts.get(e, 0) + 1
            
        dominant = max(counts.items(), key=lambda x: x[1])
        confidence = dominant[1] / len(emotions)
        
        return dominant[0], confidence


class DemographicEstimator:
    """Estimate age and gender of detected persons."""
    
    def __init__(self):
        self.age_history = {}
        
    def estimate_age(self, face_roi: np.ndarray) -> int:
        """Estimate age from face ROI."""
        if face_roi.size == 0:
            return 30
            
        h, w = face_roi.shape[:2]
        if h < 20 or w < 20:
            return 30
            
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        skin_pixels = []
        for y in range(h):
            for x in range(w):
                b, g, r = face_roi[y, x]
                if (b > 95) and (g > 40) and (r > 20) and \
                   (max(r, g, b) - min(r, g, b) > 15) and \
                   (abs(r - g) > 15) and (r > g) and (r > b):
                    skin_pixels.append((r, g, b))
        
        if len(skin_pixels) < 10:
            return 30
            
        skin_mean = np.mean(skin_pixels, axis=0)
        
        b, g, r = skin_mean
        ycbcr_cb = -0.168736 * r - 0.331294 * g + 0.5 * b + 128
        
        age = int(40 - (ycbcr_cb - 100) / 2)
        age = max(15, min(75, age))
        
        return age
    
    def estimate_gender(self, face_roi: np.ndarray) -> str:
        """Estimate gender from face ROI."""
        if face_roi.size == 0:
            return "unknown"
            
        h, w = face_roi.shape[:2]
        if h < 20 or w < 20:
            return "unknown"
        
        face_h = h
        face_w = w
        
        forehead_ratio = 0.25
        forehead_h = int(face_h * forehead_ratio)
        
        forehead = face_roi[:forehead_h, :]
        
        if forehead.size > 0:
            forehead_mean = np.mean(forehead)
            if forehead_mean > 140:
                return "female"
            else:
                return "male"
        
        return "unknown"
    
    def analyze_demographics(self, face_roi: np.ndarray) -> Dict:
        """Get complete demographic analysis."""
        age = self.estimate_age(face_roi)
        gender = self.estimate_gender(face_roi)
        
        return {
            'age': age,
            'gender': gender,
            'age_group': self._get_age_group(age),
            'is_staff': False
        }
    
    def _get_age_group(self, age: int) -> str:
        """Get age group category."""
        if age < 20:
            return "youth"
        elif age < 35:
            return "young_adult"
        elif age < 50:
            return "middle_aged"
        elif age < 65:
            return "senior"
        return "elderly"


class PoseDetector:
    """Detect human pose and activities."""
    
    POSES = {
        'standing': {'min_ratio': 0.8, 'max_ratio': 1.2},
        'sitting': {'min_ratio': 0.4, 'max_ratio': 0.7},
        'walking': {'min_ratio': 0.6, 'max_ratio': 0.9},
        'running': {'min_ratio': 0.8, 'max_ratio': 1.5},
        'bending': {'min_ratio': 0.3, 'max_ratio': 0.5}
    }
    
    def __init__(self):
        self.pose_history = {}
        
    def detect_pose(self, bbox: Tuple[int, int, int, int], 
                   prev_bbox: Tuple[int, int, int, int] = None) -> str:
        """Detect pose from bounding box."""
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        aspect_ratio = height / width if width > 0 else 1.0
        
        for pose, ratios in self.POSES.items():
            if ratios['min_ratio'] <= aspect_ratio <= ratios['max_ratio']:
                if prev_bbox and pose in ['walking', 'running']:
                    if self._calculate_velocity(bbox, prev_bbox) > 20:
                        return 'running'
                return pose
                
        return 'standing'
    
    def _calculate_velocity(self, bbox: Tuple, prev_bbox: Tuple) -> float:
        """Calculate movement velocity."""
        curr_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
        prev_center = ((prev_bbox[0] + prev_bbox[2]) // 2, (prev_bbox[1] + prev_bbox[3]) // 2)
        
        return np.sqrt((curr_center[0] - prev_center[0])**2 + 
                      (curr_center[1] - prev_center[1])**2)
    
    def detect_activity(self, positions: List[Tuple[int, int]], 
                       timestamps: List[float]) -> str:
        """Detect activity from position history."""
        if len(positions) < 2:
            return 'unknown'
            
        velocities = []
        for i in range(1, len(positions)):
            dist = np.sqrt((positions[i][0] - positions[i-1][0])**2 + 
                          (positions[i][1] - positions[i-1][1])**2)
            time_diff = timestamps[i] - timestamps[i-1]
            if time_diff > 0:
                velocities.append(dist / time_diff)
        
        if not velocities:
            return 'stationary'
            
        avg_velocity = np.mean(velocities)
        
        if avg_velocity < 5:
            return 'standing'
        elif avg_velocity < 20:
            return 'walking'
        elif avg_velocity < 50:
            return 'running'
        else:
            return 'running'


class AnomalyDetector:
    """Detect unusual behaviors and anomalies."""
    
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold
        self.position_history = {}
        self.speed_history = {}
        self.zone_history = {}
        self.normal_patterns = {}
        
    def update_tracking(self, track_id: int, position: Tuple[int, int],
                       speed: float, timestamp: float):
        """Update tracking data."""
        if track_id not in self.position_history:
            self.position_history[track_id] = deque(maxlen=50)
            self.speed_history[track_id] = deque(maxlen=50)
            self.zone_history[track_id] = deque(maxlen=20)
            
        self.position_history[track_id].append((position, timestamp))
        self.speed_history[track_id].append(speed)
        self.zone_history[track_id].append(timestamp)
    
    def detect_speed_anomaly(self, track_id: int) -> Tuple[bool, str]:
        """Detect unusual speed."""
        if track_id not in self.speed_history or len(self.speed_history[track_id]) < 10:
            return False, ""
            
        speeds = list(self.speed_history[track_id])
        mean_speed = np.mean(speeds)
        std_speed = np.std(speeds)
        
        if std_speed == 0:
            return False, ""
            
        current_speed = speeds[-1]
        z_score = abs(current_speed - mean_speed) / std_speed
        
        if z_score > self.threshold:
            if current_speed > mean_speed:
                return True, "Sudden sprint detected"
            else:
                return True, "Sudden stop detected"
                
        return False, ""
    
    def detect_Loitering(self, track_id: int, zone: str, 
                        duration_threshold: float = 60.0) -> Tuple[bool, str]:
        """Detect loitering in area."""
        if track_id not in self.zone_history:
            return False, ""
            
        zone_visits = [z for z in self.zone_history[track_id] if z == zone]
        
        if len(zone_visits) >= 3:
            time_spent = time.time() - zone_visits[0]
            if time_spent > duration_threshold:
                return True, f"Loitering in {zone} zone"
                
        return False, ""
    
    def detect_zone_violation(self, track_id: int, 
                             restricted_zones: List[str]) -> Tuple[bool, str]:
        """Detect entry into restricted zones."""
        if track_id not in self.zone_history:
            return False, ""
            
        recent_zones = list(self.zone_history[track_id])[-5:]
        
        for zone in recent_zones:
            if zone in restricted_zones:
                return True, f"Restricted zone access: {zone}"
                
        return False, ""
    
    def detect_abnormal_group(self, positions: List[Tuple[int, int]]) -> Tuple[bool, str]:
        """Detect abnormal group behavior."""
        if len(positions) < 3:
            return False, ""
            
        positions = np.array(positions)
        centroid = np.mean(positions, axis=0)
        
        distances = [np.sqrt((p[0] - centroid[0])**2 + (p[1] - centroid[1])**2) 
                    for p in positions]
        
        if np.std(distances) > 100:
            return True, "Abnormal group formation"
            
        return False, ""
    
    def get_all_anomalies(self, track_id: int, current_zone: str,
                         restricted_zones: List[str]) -> List[Dict]:
        """Get all detected anomalies for a track."""
        anomalies = []
        
        is_anomaly, msg = self.detect_speed_anomaly(track_id)
        if is_anomaly:
            anomalies.append({'type': 'speed', 'message': msg, 'severity': 'medium'})
        
        is_anomaly, msg = self.detect_Loitering(track_id, current_zone)
        if is_anomaly:
            anomalies.append({'type': 'loitering', 'message': msg, 'severity': 'high'})
        
        is_anomaly, msg = self.detect_zone_violation(track_id, restricted_zones)
        if is_anomaly:
            anomalies.append({'type': 'violation', 'message': msg, 'severity': 'critical'})
        
        return anomalies


class PredictiveAI:
    """AI-powered predictions for restaurant operations."""
    
    def __init__(self):
        self.historical_data = {
            'hourly': {},
            'daily': {},
            'weekly': {}
        }
        
    def add_observation(self, hour: int, day: int, 
                       customers: int, wait_time: float):
        """Add observation for learning."""
        if hour not in self.historical_data['hourly']:
            self.historical_data['hourly'][hour] = []
        self.historical_data['hourly'][hour].append({
            'customers': customers,
            'wait_time': wait_time
        })
    
    def predict_customers(self, hour: int, day: int = None) -> Dict:
        """Predict expected customers."""
        if hour in self.historical_data['hourly']:
            data = self.historical_data['hourly'][hour]
            customers = [d['customers'] for d in data]
            return {
                'predicted': int(np.mean(customers)),
                'min': min(customers),
                'max': max(customers),
                'confidence': min(0.9, len(data) / 50)
            }
        
        base_predictions = {
            6: 5, 7: 15, 8: 25, 9: 20, 10: 15, 11: 20,
            12: 45, 13: 55, 14: 35, 15: 25, 16: 20, 17: 25,
            18: 50, 19: 65, 20: 45, 21: 30, 22: 15
        }
        
        return {
            'predicted': base_predictions.get(hour, 20),
            'min': 0,
            'max': 100,
            'confidence': 0.5
        }
    
    def predict_wait_time(self, current_customers: int, 
                         staff_count: int, hour: int) -> float:
        """Predict wait time."""
        if staff_count == 0:
            return 999
            
        ratio = current_customers / staff_count
        
        base_wait = ratio * 5
        
        peak_hours = [12, 13, 18, 19, 20]
        if hour in peak_hours:
            base_wait *= 1.5
            
        return min(60, base_wait)
    
    def recommend_staff(self, predicted_customers: int, 
                       target_wait_time: float = 10.0) -> int:
        """Recommend staff count."""
        optimal_ratio = 5
        
        recommended = max(1, int(predicted_customers / optimal_ratio))
        
        if target_wait_time < 5:
            recommended += 1
            
        return recommended
    
    def detect_peak_window(self, hour: int) -> str:
        """Detect if current time is peak."""
        if 12 <= hour <= 14:
            return "lunch_peak"
        elif 18 <= hour <= 21:
            return "dinner_peak"
        elif 7 <= hour <= 9:
            return "breakfast_peak"
        else:
            return "off_peak"
    
    def generate_forecast(self, days: int = 7) -> List[Dict]:
        """Generate forecast for next N days."""
        forecast = []
        
        for day in range(days):
            for hour in range(24):
                prediction = self.predict_customers(hour, day)
                wait_time = self.predict_wait_time(
                    prediction['predicted'],
                    self.recommend_staff(prediction['predicted']),
                    hour
                )
                forecast.append({
                    'day': day,
                    'hour': hour,
                    'predicted_customers': prediction['predicted'],
                    'predicted_wait': wait_time,
                    'period': self.detect_peak_window(hour)
                })
                
        return forecast


class NaturalLanguageGenerator:
    """Generate natural language insights."""
    
    def __init__(self):
        self.insights = []
        
    def generate_insight(self, metric_type: str, value: float, 
                        threshold: float, context: Dict = None) -> str:
        """Generate insight based on metrics."""
        
        if metric_type == "wait_time":
            if value > threshold * 1.5:
                return f"⚠️ Wait times are critically high at {value:.0f} minutes. Consider adding {context.get('recommended_staff', 1)} more staff members."
            elif value > threshold:
                return f"Wait times are elevated at {value:.0f} minutes. Monitor the queue situation."
            else:
                return f"✅ Service is running smoothly with {value:.0f} minute wait times."
                
        elif metric_type == "crowd_density":
            if value > 0.8:
                return f"🚨 High crowd density detected at {value*100:.0f}%. Consider opening additional service points."
            elif value > 0.5:
                return f"Moderate crowd levels at {value*100:.0f}%. Current staffing appears adequate."
            else:
                return f"Low crowd density at {value*100:.0f}%. Opportunity to optimize staff scheduling."
                
        elif metric_type == "staff_idle":
            if value > 50:
                return f"Staff idle time is {value:.0f}%. Consider reducing staff during this period."
            else:
                return f"Staff utilization is good at {100-value:.0f}%."
                
        return f"Metric {metric_type}: {value}"
    
    def generate_summary(self, analytics: Dict) -> str:
        """Generate complete summary."""
        lines = [
            "📊 Restaurant Analytics Summary",
            "=" * 40,
            "",
            f"Current Customers: {analytics.get('current_count', 0)}",
            f"Total Entries Today: {analytics.get('entries', 0)}",
            f"Average Wait Time: {analytics.get('avg_wait', 0):.1f} minutes",
            f"Staff On Duty: {analytics.get('staff_count', 0)}",
            "",
            "Insights:"
        ]
        
        for insight in analytics.get('insights', []):
            lines.append(f"• {insight}")
            
        return "\n".join(lines)
    
    def generate_alert(self, alert_type: str, details: Dict) -> str:
        """Generate alert message."""
        
        if alert_type == "queue_overflow":
            return f"🚨 ALERT: Queue overflow detected! {details.get('queue_length', 0)} people waiting. Immediate action required."
            
        elif alert_type == "staff_shortage":
            return f"⚠️ ALERT: Staff shortage detected. Customer-to-staff ratio is {details.get('ratio', 0)}:1. Recommended: {details.get('recommended', 0)} staff."
            
        elif alert_type == "vip_arrival":
            return f"👑 VIP customer detected in {details.get('zone', 'entrance')} zone. Prepare accordingly."
            
        elif alert_type == "anomaly":
            return f"🔍 Anomaly detected: {details.get('description', 'Unknown')}. Track ID: {details.get('track_id', 'N/A')}"
            
        return f"Alert: {alert_type}"
