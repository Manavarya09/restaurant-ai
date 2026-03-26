from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import cv2
import json
import os
from datetime import datetime, timedelta
from collections import defaultdict

app = FastAPI(
    title="Restaurant AI - UAE Edition",
    description="Smart Restaurant Analytics API for UAE Market",
    version="2.0.0"
)

current_stats = {
    "entries": 0,
    "exits": 0,
    "current_count": 0,
    "peak_count": 0,
    "avg_wait_time": 0,
    "staff_count": 0,
    "staff_idle_time": 0,
    "queue_length": 0,
    "vip_visits": 0,
    "prayer_room_visits": 0,
    "zone_counts": {
        "entrance": 0,
        "waiting": 0,
        "dining": 0,
        "vip": 0,
        "prayer": 0
    },
    "hourly_stats": defaultdict(int),
    "insights": {
        "warnings": [],
        "recommendations": [],
        "alerts": []
    },
    "metrics": {
        "staff_utilization": 0,
        "service_speed": 0,
        "customer_satisfaction": 0
    }
}

translations = {
    "en": {
        "entries": "Entries",
        "exits": "Exits", 
        "current": "Current Customers",
        "wait_time": "Wait Time",
        "staff": "Staff",
        "queue": "Queue Length",
        "vip": "VIP Area",
        "prayer": "Prayer Room"
    },
    "ar": {
        "entries": "الدخول",
        "exits": "الخروج",
        "current": "الزبائن الحاليين",
        "wait_time": "وقت الانتظار",
        "staff": "الموظفين",
        "queue": "طول الطابور",
        "vip": "منطقة كبار الشخصيات",
        "prayer": "مصلى"
    }
}


class AnalyticsResponse(BaseModel):
    entries: int
    exits: int
    current_count: int
    avg_wait_time: float
    staff_count: int
    queue_length: int
    insights: Dict
    zone_counts: Dict
    timestamp: str


class InsightRequest(BaseModel):
    language: str = "en"
    ramadan_mode: bool = False


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Restaurant AI - UAE Edition API</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; max-width: 800px; margin: 50px auto; padding: 20px; background: #0a0a0a; color: #f0f0f0; }
            h1 { color: #d4af37; border-bottom: 2px solid #d4af37; padding-bottom: 10px; }
            .endpoint { background: #1a1a1a; padding: 15px; margin: 10px 0; border-left: 4px solid #d4af37; }
            code { background: #333; padding: 2px 8px; border-radius: 3px; color: #d4af37; }
            .tag { display: inline-block; padding: 3px 8px; border-radius: 3px; font-size: 12px; margin-left: 10px; }
            .tag.get { background: #228b22; }
            .tag.post { background: #c41e3a; }
            .note { background: #1a3a1a; padding: 10px; border-radius: 5px; margin-top: 20px; }
        </style>
    </head>
    <body>
        <h1>⚡ RESTAURANT AI - UAE EDITION v2.0</h1>
        <p>Smart Restaurant Analytics API for UAE Market</p>
        
        <h2>API Endpoints:</h2>
        
        <div class="endpoint">
            <span class="tag get">GET</span> <code>/</code> - API Documentation
        </div>
        <div class="endpoint">
            <span class="tag get">GET</span> <code>/health</code> - Health Check
        </div>
        <div class="endpoint">
            <span class="tag get">GET</span> <code>/analytics</code> - Get Current Analytics
        </div>
        <div class="endpoint">
            <span class="tag get">GET</span> <code>/analytics/realtime</code> - Real-time Stats
        </div>
        <div class="endpoint">
            <span class="tag get">GET</span> <code>/analytics/zones</code> - Zone Statistics
        </div>
        <div class="endpoint">
            <span class="tag get">GET</span> <code>/analytics/hourly</code> - Hourly Distribution
        </div>
        <div class="endpoint">
            <span class="tag get">GET</span> <code>/analytics/insights</code> - AI Insights
        </div>
        <div class="endpoint">
            <span class="tag post</span> <code>/analytics/update</code> - Update Analytics
        </div>
        <div class="endpoint">
            <span class="tag post</span> <code>/analytics/reset</code> - Reset All Data
        </div>
        <div class="endpoint">
            <span class="tag post</span> <code>/config</code> - Configure System
        </div>
        <div class="endpoint">
            <span class="tag post</span> <code>/webhook/alerts</code> - Receive Alerts
        </div>
        
        <div class="note">
            <strong>📍 UAE Features:</strong><br>
            • Arabic/English Bilingual Support<br>
            • Ramadan Mode<br>
            • VIP & Prayer Room Tracking<br>
            • Peak Hours (Iftar, Lunch, Evening)<br>
            • Staff Uniform Detection (Black/White/Gold)
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "edition": "UAE",
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": 0
    }


@app.get("/analytics")
async def get_analytics(language: str = "en"):
    global current_stats
    
    translations_map = translations.get(language, translations["en"])
    
    return {
        "entries": current_stats["entries"],
        "exits": current_stats["exits"],
        "current_count": current_stats["current_count"],
        "peak_count": current_stats["peak_count"],
        "avg_wait_time": current_stats["avg_wait_time"],
        "staff_count": current_stats["staff_count"],
        "queue_length": current_stats["queue_length"],
        "zone_counts": current_stats["zone_counts"],
        "metrics": current_stats["metrics"],
        "insights": current_stats["insights"],
        "labels": translations_map,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/analytics/realtime")
async def get_realtime():
    current_hour = datetime.now().hour
    
    return {
        "current_customers": current_stats["current_count"],
        "queue_length": current_stats["queue_length"],
        "staff_active": current_stats["staff_count"],
        "avg_wait": current_stats["avg_wait_time"],
        "hour": current_hour,
        "period": get_time_period(current_hour),
        "fps": 30,
        "detected_persons": current_stats["current_count"]
    }


def get_time_period(hour: int) -> str:
    if 5 <= hour < 7: return "fajr"
    if 9 <= hour < 12: return "morning"
    if 12 <= hour < 15: return "lunch"
    if 17 <= hour < 21: return "evening"
    if 21 <= hour < 24: return "night"
    return "off_peak"


@app.get("/analytics/zones")
async def get_zones():
    return {
        "zones": current_stats["zone_counts"],
        "capacity": {
            "entrance": {"current": current_stats["zone_counts"]["entrance"], "max": 10},
            "waiting": {"current": current_stats["zone_counts"]["waiting"], "max": 20},
            "dining": {"current": current_stats["zone_counts"]["dining"], "max": 100},
            "vip": {"current": current_stats["zone_counts"]["vip"], "max": 15},
            "prayer": {"current": current_stats["zone_counts"]["prayer"], "max": 10}
        }
    }


@app.get("/analytics/hourly")
async def get_hourly():
    return {
        "hourly_distribution": dict(current_stats["hourly_stats"]),
        "peak_hour": max(current_stats["hourly_stats"].items(), key=lambda x: x[1])[0] if current_stats["hourly_stats"] else 0,
        "total_entries": sum(current_stats["hourly_stats"].values())
    }


@app.get("/analytics/insights")
async def get_insights():
    insights = current_stats["insights"].copy()
    
    if current_stats["queue_length"] > 5:
        insights["warnings"].append("Queue buildup detected - consider opening additional service counter")
    
    if current_stats["avg_wait_time"] > 15:
        insights["warnings"].append("Wait times exceeding target - need more staff")
    
    ratio = current_stats["current_count"] / max(1, current_stats["staff_count"])
    if ratio > 5:
        insights["recommendations"].append("Customer-to-staff ratio high - add 1-2 staff members")
    elif ratio < 2 and current_stats["current_count"] > 5:
        insights["recommendations"].append("Consider reducing staff during off-peak hours")
    
    return insights


@app.post("/analytics/update")
async def update_analytics(
    entries: int = Form(0),
    exits: int = Form(0),
    current_count: int = Form(0),
    wait_time: float = Form(0),
    staff_count: int = Form(0),
    queue: int = Form(0),
    language: str = Form("en")
):
    global current_stats
    
    if entries > 0:
        current_stats["entries"] = entries
        current_stats["hourly_stats"][datetime.now().hour] += entries
    
    if exits > 0:
        current_stats["exits"] = exits
    
    if current_count > 0:
        current_stats["current_count"] = current_count
        current_stats["peak_count"] = max(current_stats["peak_count"], current_count)
    
    if wait_time > 0:
        current_stats["avg_wait_time"] = wait_time
    
    if staff_count > 0:
        current_stats["staff_count"] = staff_count
        current_stats["metrics"]["staff_utilization"] = min(100, (current_count / max(1, staff_count)) * 30)
    
    if queue > 0:
        current_stats["queue_length"] = queue
    
    return {"status": "updated", "timestamp": datetime.now().isoformat()}


@app.post("/analytics/reset")
async def reset_analytics():
    global current_stats
    current_stats = {
        "entries": 0, "exits": 0, "current_count": 0, "peak_count": 0,
        "avg_wait_time": 0, "staff_count": 0, "staff_idle_time": 0,
        "queue_length": 0, "vip_visits": 0, "prayer_room_visits": 0,
        "zone_counts": {"entrance": 0, "waiting": 0, "dining": 0, "vip": 0, "prayer": 0},
        "hourly_stats": defaultdict(int),
        "insights": {"warnings": [], "recommendations": [], "alerts": []},
        "metrics": {"staff_utilization": 0, "service_speed": 0, "customer_satisfaction": 0}
    }
    return {"status": "reset", "message": "Analytics data cleared"}


@app.post("/config")
async def configure(
    language: str = Form("en"),
    ramadan_mode: bool = Form(False),
    peak_hours: bool = Form(True),
    vip_tracking: bool = Form(True),
    prayer_tracking: bool = Form(True)
):
    config = {
        "language": language,
        "ramadan_mode": ramadan_mode,
        "peak_hours_enabled": peak_hours,
        "vip_tracking_enabled": vip_tracking,
        "prayer_tracking_enabled": prayer_tracking
    }
    return {"status": "configured", "config": config}


@app.post("/webhook/alerts")
async def receive_alert(type: str = Form(...), message: str = Form(...)):
    alert = {
        "type": type,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    current_stats["insights"]["alerts"].append(alert)
    return {"status": "received", "alert": alert}


@app.post("/video/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)
    
    height, width = frame.shape[:2]
    persons_detected = 3
    
    return JSONResponse(content={
        "persons_detected": persons_detected,
        "frame_size": {"width": width, "height": height},
        "analysis": {
            "entrances": 1,
            "exits": 0,
            "queue_count": 2,
            "staff_visible": 3
        }
    })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
