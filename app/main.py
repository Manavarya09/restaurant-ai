from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import numpy as np
import cv2
import json
import os
from datetime import datetime

app = FastAPI(title="Restaurant Analytics API")

current_stats = {
    'entries': 0,
    'exits': 0,
    'current_count': 0,
    'avg_wait_time': 0,
    'staff_count': 0,
    'avg_idle_time': 0,
    'insights': {
        'warnings': [],
        'recommendations': [],
        'metrics': {}
    }
}


class AnalyticsResponse(BaseModel):
    entries: int
    exits: int
    current_count: int
    avg_wait_time: float
    staff_count: int
    avg_idle_time: float
    insights: Dict


class VideoFrameRequest(BaseModel):
    frame_data: str
    timestamp: Optional[float] = None


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <head>
            <title>Restaurant Analytics API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                h1 { color: #333; }
                h2 { color: #666; }
                code { background: #e0e0e0; padding: 2px 6px; border-radius: 3px; }
            </style>
        </head>
        <body>
            <h1>Restaurant Analytics API</h1>
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <code>GET /analytics</code> - Get current analytics
            </div>
            <div class="endpoint">
                <code>POST /update</code> - Update analytics data
            </div>
            <div class="endpoint">
                <code>POST /video/analyze</code> - Analyze video frame
            </div>
            <div class="endpoint">
                <code>GET /health</code> - Health check
            </div>
        </body>
    </html>
    """


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@app.get("/analytics", response_model=AnalyticsResponse)
async def get_analytics():
    return current_stats


@app.post("/update")
async def update_analytics(
    entries: int = Form(...),
    exits: int = Form(...),
    current_count: int = Form(...),
    avg_wait_time: float = Form(...),
    staff_count: int = Form(...),
    avg_idle_time: float = Form(...)
):
    current_stats['entries'] = entries
    current_stats['exits'] = exits
    current_stats['current_count'] = current_count
    current_stats['avg_wait_time'] = avg_wait_time
    current_stats['staff_count'] = staff_count
    current_stats['avg_idle_time'] = avg_idle_time
    
    current_stats['insights']['metrics'] = {
        'current_customers': current_count,
        'staff_count': staff_count,
        'avg_wait_time': avg_wait_time,
        'avg_idle_time': avg_idle_time
    }
    
    if staff_count > 0 and current_count > 0:
        ratio = current_count / staff_count
        if ratio < 1 and avg_idle_time > 300:
            current_stats['insights']['warnings'].append('Overstaffing detected')
            current_stats['insights']['recommendations'].append('Consider reducing staff')
        elif ratio > 5 and avg_wait_time > 600:
            current_stats['insights']['warnings'].append('Understaffing detected')
            current_stats['insights']['recommendations'].append('Consider adding staff')
    
    return {"status": "updated", "data": current_stats}


@app.post("/video/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if frame is None:
        return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)
    
    height, width = frame.shape[:2]
    
    return JSONResponse(content={
        "frame_size": {"width": width, "height": height},
        "message": "Frame analyzed successfully"
    })


@app.post("/reset")
async def reset_analytics():
    global current_stats
    current_stats = {
        'entries': 0,
        'exits': 0,
        'current_count': 0,
        'avg_wait_time': 0,
        'staff_count': 0,
        'avg_idle_time': 0,
        'insights': {
            'warnings': [],
            'recommendations': [],
            'metrics': {}
        }
    }
    return {"status": "reset", "message": "Analytics reset successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
