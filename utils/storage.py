"""
Storage Module for Restaurant AI
Supports: SQLite, JSON, Excel exports
"""
import sqlite3
import json
import csv
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd


class AnalyticsDatabase:
    """SQLite database for analytics storage."""
    
    def __init__(self, db_path: str = "../outputs/analytics.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS footfall (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                entries INTEGER,
                exits INTEGER,
                current INTEGER,
                hour INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS zones (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                zone_name TEXT,
                count INTEGER
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS wait_times (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id INTEGER,
                start_time REAL,
                end_time REAL,
                wait_seconds REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS staff (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                staff_id INTEGER,
                is_active INTEGER,
                idle_time REAL
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS insights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                insight_type TEXT,
                message TEXT,
                resolved INTEGER DEFAULT 0
            )
        """)
        
        conn.commit()
        conn.close()
        
    def record_footfall(self, entries: int, exits: int, current: int):
        """Record footfall data."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().timestamp()
        hour = datetime.now().hour
        
        cursor.execute("""
            INSERT INTO footfall (timestamp, entries, exits, current, hour)
            VALUES (?, ?, ?, ?, ?)
        """, (timestamp, entries, exits, current, hour))
        
        conn.commit()
        conn.close()
        
    def record_zone_count(self, zone_name: str, count: int):
        """Record zone count."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().timestamp()
        
        cursor.execute("""
            INSERT INTO zones (timestamp, zone_name, count)
            VALUES (?, ?, ?)
        """, (timestamp, zone_name, count))
        
        conn.commit()
        conn.close()
        
    def record_wait_time(self, track_id: int, wait_seconds: float):
        """Record wait time."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().timestamp()
        
        cursor.execute("""
            INSERT INTO wait_times (track_id, end_time, wait_seconds)
            VALUES (?, ?, ?)
        """, (track_id, timestamp, wait_seconds))
        
        conn.commit()
        conn.close()
        
    def get_hourly_stats(self, hours: int = 24) -> Dict[int, Dict]:
        """Get hourly statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT hour, SUM(entries), SUM(exits), AVG(current)
            FROM footfall
            WHERE timestamp > ?
            GROUP BY hour
        """, ((datetime.now() - timedelta(hours=hours)).timestamp(),))
        
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = {
                'entries': row[1],
                'exits': row[2],
                'avg_current': row[3]
            }
            
        conn.close()
        return results
    
    def get_daily_stats(self, days: int = 7) -> Dict:
        """Get daily statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT DATE(timestamp, 'unixepoch') as day,
                   SUM(entries), SUM(exits), AVG(wait_seconds)
            FROM footfall f
            LEFT JOIN wait_times w ON f.timestamp = w.start_time
            WHERE f.timestamp > ?
            GROUP BY day
        """, ((datetime.now() - timedelta(days=days)).timestamp(),))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'date': row[0],
                'entries': row[1],
                'exits': row[2],
                'avg_wait': row[3]
            })
            
        conn.close()
        return results
    
    def export_csv(self, table: str, output_path: str):
        """Export table to CSV."""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        df.to_csv(output_path, index=False)
        conn.close()
        
    def get_zone_metrics(self) -> Dict:
        """Get zone performance metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT zone_name, AVG(count), MAX(count), MIN(count)
            FROM zones
            WHERE timestamp > ?
            GROUP BY zone_name
        """, ((datetime.now() - timedelta(hours=24)).timestamp(),))
        
        results = {}
        for row in cursor.fetchall():
            results[row[0]] = {
                'avg_count': row[1],
                'max_count': row[2],
                'min_count': row[3]
            }
            
        conn.close()
        return results


class JSONExporter:
    """Export analytics to JSON."""
    
    def __init__(self, output_dir: str = "../outputs"):
        self.output_dir = Path(output_dir)
        
    def export_daily_report(self, data: dict):
        """Export daily report."""
        date = datetime.now().strftime("%Y-%m-%d")
        path = self.output_dir / f"report_{date}.json"
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return str(path)
    
    def export_realtime(self, data: dict):
        """Export realtime data."""
        path = self.output_dir / "realtime.json"
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
            
        return str(path)


class ExcelExporter:
    """Export analytics to Excel with formatting."""
    
    def __init__(self, output_dir: str = "../outputs"):
        self.output_dir = Path(output_dir)
        
    def export_report(self, data: dict):
        """Export full report to Excel."""
        date = datetime.now().strftime("%Y-%m-%d")
        path = self.output_dir / f"report_{date}.xlsx"
        
        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            if 'footfall' in data:
                df = pd.DataFrame(data['footfall'])
                df.to_excel(writer, sheet_name='Footfall', index=False)
                
            if 'zones' in data:
                df = pd.DataFrame(data['zones'])
                df.to_excel(writer, sheet_name='Zones', index=False)
                
            if 'wait_times' in data:
                df = pd.DataFrame(data['wait_times'])
                df.to_excel(writer, sheet_name='Wait Times', index=False)
                
        return str(path)


class NotificationManager:
    """Send notifications for alerts."""
    
    def __init__(self):
        self.webhooks: List[str] = []
        self.alerts: List[Dict] = []
        
    def add_webhook(self, url: str):
        """Add webhook URL."""
        self.webhooks.append(url)
        
    def send_alert(self, title: str, message: str, priority: str = "medium"):
        """Send alert notification."""
        alert = {
            'title': title,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        }
        
        self.alerts.append(alert)
        
        for webhook in self.webhooks:
            self._send_webhook(webhook, alert)
            
    def _send_webhook(self, url: str, payload: dict):
        """Send webhook request."""
        import urllib.request
        import json
        
        try:
            data = json.dumps(payload).encode('utf-8')
            req = urllib.request.Request(
                url,
                data=data,
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            print(f"Webhook failed: {e}")
            
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alerts 
                if datetime.fromisoformat(a['timestamp']) > cutoff]


class CachingManager:
    """Cache frequently accessed data."""
    
    def __init__(self, ttl: int = 60):
        self.cache: Dict[str, tuple] = {}
        self.ttl = ttl
        
    def get(self, key: str) -> Optional[any]:
        """Get cached value."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
        
    def set(self, key: str, value: any):
        """Set cached value."""
        self.cache[key] = (value, datetime.now().timestamp())
        
    def clear(self):
        """Clear cache."""
        self.cache.clear()
