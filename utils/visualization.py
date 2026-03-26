import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Dict, List, Optional
from collections import defaultdict
import json
from datetime import datetime


class AnalyticsEngine:
    """Generate insights and analytics from tracked data."""
    
    def __init__(self):
        self.data = {
            'footfall': [],
            'wait_times': [],
            'zone_durations': defaultdict(list),
            'staff_metrics': [],
            'hourly_stats': defaultdict(lambda: {'entries': 0, 'exits': 0, 'customers': 0})
        }
        
    def record_footfall(self, timestamp: float, entries: int, exits: int):
        """Record footfall data point."""
        self.data['footfall'].append({
            'timestamp': timestamp,
            'entries': entries,
            'exits': exits,
            'current': entries - exits
        })
    
    def record_wait_time(self, track_id: int, wait_time: float, zone_sequence: List[str]):
        """Record wait time data."""
        self.data['wait_times'].append({
            'track_id': track_id,
            'wait_time': wait_time,
            'zones': zone_sequence
        })
    
    def record_zone_duration(self, zone: str, duration: float):
        """Record zone duration."""
        self.data['zone_durations'][zone].append(duration)
    
    def record_staff_metrics(self, timestamp: float, staff_count: int, 
                             avg_idle_time: float, interactions: int):
        """Record staff metrics."""
        self.data['staff_metrics'].append({
            'timestamp': timestamp,
            'staff_count': staff_count,
            'avg_idle_time': avg_idle_time,
            'interactions': interactions
        })
    
    def analyze_peak_hours(self) -> Dict:
        """Analyze peak hours."""
        if not self.data['footfall']:
            return {}
        
        hourly_entries = defaultdict(int)
        for entry in self.data['footfall']:
            hour = datetime.fromtimestamp(entry['timestamp']).hour
            hourly_entries[hour] += entry['entries']
        
        if hourly_entries:
            peak_hour = max(hourly_entries.keys(), key=lambda h: hourly_entries[h])
            return {
                'peak_hour': peak_hour,
                'peak_entries': hourly_entries[peak_hour],
                'hourly_distribution': dict(hourly_entries)
            }
        return {}
    
    def detect_overstaffing(self, current_customers: int, staff_count: int, 
                           avg_idle_time: float) -> bool:
        """Detect if restaurant is overstaffed."""
        if staff_count == 0 or current_customers == 0:
            return False
        
        customers_per_staff = current_customers / staff_count
        return customers_per_staff < 1 and avg_idle_time > 300
    
    def detect_understaffing(self, avg_wait_time: float, current_customers: int,
                             staff_count: int) -> bool:
        """Detect if restaurant is understaffed."""
        if staff_count == 0:
            return False
        
        return avg_wait_time > 600 and current_customers / staff_count > 5
    
    def generate_insights(self, current_customers: int, staff_count: int,
                         avg_wait_time: float, avg_idle_time: float) -> Dict:
        """Generate actionable insights."""
        insights = {
            'recommendations': [],
            'warnings': [],
            'metrics': {
                'current_customers': current_customers,
                'staff_count': staff_count,
                'avg_wait_time': avg_wait_time,
                'avg_idle_time': avg_idle_time
            }
        }
        
        if self.detect_overstaffing(current_customers, staff_count, avg_idle_time):
            insights['warnings'].append('Overstaffing detected - consider reducing staff during off-peak hours')
            insights['recommendations'].append('Reduce staff by 1-2 during low customer periods')
        
        if self.detect_understaffing(avg_wait_time, current_customers, staff_count):
            insights['warnings'].append('Understaffing detected - wait times are too high')
            insights['recommendations'].append('Add 1-2 staff members to reduce wait times')
        
        if avg_wait_time > 300:
            insights['warnings'].append(f'High wait times: {avg_wait_time/60:.1f} minutes')
        
        if current_customers > 0 and staff_count > 0:
            ratio = current_customers / staff_count
            if ratio > 5:
                insights['recommendations'].append(f'Customer-to-staff ratio is {ratio:.1f}:1 - consider adding staff')
            elif ratio < 2:
                insights['recommendations'].append(f'Customer-to-staff ratio is {ratio:.1f}:1 - consider reducing staff')
        
        return insights
    
    def export_to_csv(self, filepath: str):
        """Export data to CSV."""
        if self.data['footfall']:
            df = pd.DataFrame(self.data['footfall'])
            df.to_csv(filepath, index=False)
    
    def save_report(self, filepath: str):
        """Save analytics report as JSON."""
        report = {
            'summary': {
                'total_entries': sum(e['entries'] for e in self.data['footfall']),
                'total_exits': sum(e['exits'] for e in self.data['footfall']),
                'avg_wait_time': np.mean([w['wait_time'] for w in self.data['wait_times']]) if self.data['wait_times'] else 0,
                'peak_hours': self.analyze_peak_hours()
            },
            'data': {
                'footfall': self.data['footfall'],
                'wait_times': self.data['wait_times'],
                'zone_durations': {k: list(v) for k, v in self.data['zone_durations'].items()}
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)


def plot_footfall_graph(footfall_data: List[Dict], save_path: Optional[str] = None):
    """Plot footfall over time."""
    if not footfall_data:
        return
    
    df = pd.DataFrame(footfall_data)
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['entries'], name='Entries', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['exits'], name='Exits', fill='tozeroy'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['current'], name='Current', line=dict(color='green', width=2)))
    
    fig.update_layout(title='Footfall Over Time', xaxis_title='Time', yaxis_title='Count')
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_wait_time_distribution(wait_times: List[float], save_path: Optional[str] = None):
    """Plot wait time distribution."""
    if not wait_times:
        return
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wait_times, nbinsx=30, name='Wait Times'))
    
    fig.update_layout(title='Wait Time Distribution', xaxis_title='Wait Time (seconds)', yaxis_title='Frequency')
    
    if save_path:
        fig.write_html(save_path)
    
    return fig


def plot_staff_activity(staff_metrics: List[Dict], save_path: Optional[str] = None):
    """Plot staff activity over time."""
    if not staff_metrics:
        return
    
    df = pd.DataFrame(staff_metrics)
    df['time'] = pd.to_datetime(df['timestamp'], unit='s')
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['time'], y=df['staff_count'], name='Staff Count'))
    fig.add_trace(go.Scatter(x=df['time'], y=df['interactions'], name='Interactions'))
    
    fig.update_layout(title='Staff Activity', xaxis_title='Time', yaxis_title='Count')
    
    if save_path:
        fig.write_html(save_path)
    
    return fig
