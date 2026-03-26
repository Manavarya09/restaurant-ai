"""
UAE Restaurant Analytics Configuration
Localized for Middle East market
"""
import json
from datetime import datetime, timedelta
import numpy as np


UAEC_CONFIG = {
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
            "color": [0, 255, 255],
            "ar_name": "المدخل",
            "en_name": "Entrance"
        },
        "waiting": {
            "points": [[100, 250], [250, 250], [250, 350], [100, 350]],
            "color": [0, 255, 0],
            "ar_name": "منطقة الانتظار",
            "en_name": "Waiting Area"
        },
        "dining": {
            "points": [[400, 50], [620, 50], [620, 350], [400, 350]],
            "color": [255, 0, 0],
            "ar_name": "منطقة الطعام",
            "en_name": "Dining Area"
        },
        "prayer": {
            "points": [[500, 400], [600, 400], [600, 450], [500, 450]],
            "color": [128, 0, 128],
            "ar_name": "مصلى",
            "en_name": "Prayer Area"
        },
        "vip": {
            "points": [[50, 50], [150, 50], [150, 150], [50, 150]],
            "color": [255, 215, 0],
            "ar_name": "منطقة كبار الشخصيات",
            "en_name": "VIP Area"
        }
    },
    "footfall_line": {
        "start": [320, 0],
        "end": [320, 480]
    },
    "staff_detection": {
        "enabled": True,
        "hsv_lower": [0, 0, 0],
        "hsv_upper": [180, 50, 80],
        "uniform_colors": {
            "black": {"lower": [0, 0, 0], "upper": [180, 80, 50]},
            "white": {"lower": [0, 0, 200], "upper": [180, 30, 255]},
            "gold": {"lower": [15, 150, 150], "upper": [35, 255, 255]}
        }
    },
    "output": {
        "dir": "../outputs",
        "save_video": True,
        "save_report": True,
        "export_formats": ["json", "csv", "xlsx"]
    },
    "uae_specific": {
        "language": "en",
        " prayer_times": True,
        "ramadan_mode": False,
        "peak_hours": {
            "fajr": [5, 7],
            "morning": [9, 11],
            "lunch": [12, 15],
            "evening": [18, 21],
            "iftar": [18, 21]
        },
        "metrics": {
            "vip_tracking": True,
            "family_tracking": True,
            "queue_alerts": True,
            "staff_rotation": True
        }
    }
}


class UAERestaurantConfig:
    def __init__(self, config_dict=None):
        self.config = config_dict or UAEC_CONFIG.copy()
        
    def get_zone_names(self, language='en'):
        zones = self.config.get('zones', {})
        return {k: v.get(f'{language}_name', v['en_name']) for k, v in zones.items()}
    
    def get_prayer_times(self, date=None):
        if date is None:
            date = datetime.now()
        return {
            'fajr': date.replace(hour=5, minute=30),
            'dhuhr': date.replace(hour=12, minute=30),
            'asr': date.replace(hour=15, minute=45),
            'maghrib': date.replace(hour=18, minute=15),
            'isha': date.replace(hour=20, minute=0)
        }
    
    def is_ramadan(self):
        return self.config['uae_specific'].get('ramadan_mode', False)
    
    def get_peak_hours(self):
        return self.config['uae_specific']['peak_hours']
    
    def get_current_period(self):
        now = datetime.now()
        hour = now.hour
        peaks = self.get_peak_hours()
        
        if self.is_ramadan() and 17 <= hour <= 21:
            return 'iftar'
        for period, (start, end) in peaks.items():
            if start <= hour < end:
                return period
        return 'off_peak'
    
    def set_language(self, lang):
        if lang in ['en', 'ar']:
            self.config['uae_specific']['language'] = lang
    
    def enable_ramadan_mode(self):
        self.config['uae_specific']['ramadan_mode'] = True
    
    def disable_ramadan_mode(self):
        self.config['uae_specific']['ramadan_mode'] = False


def get_default_config():
    return UAERestaurantConfig()
