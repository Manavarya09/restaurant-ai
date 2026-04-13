"""
Restaurant AI — UAE Edition v3.0
FastAPI backend with real YOLO detection, DeepSORT tracking, and zone analytics.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import asyncio
import base64
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse,
                                StreamingResponse)
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Restaurant AI — UAE Edition", version="3.0.0")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

_dir = os.path.dirname(os.path.abspath(__file__))
app.mount("/static", StaticFiles(directory=_dir), name="static")

# ── AI STATE ──
ai_ready = False
ai_loading = False
ai_error: Optional[str] = None

detector = None
tracker = None
footfall_counter = None
staff_detector_obj = None
zone_manager = None

# ── TRAIN STATE ──
train_state: dict = {
    "status": "idle",       # idle | running | done | error
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "map50": None,
    "map50_95": None,
    "weights_path": None,
    "error": None,
}

# ── ANALYTICS STATE ──
current_stats: dict = {
    "entries": 0, "exits": 0, "current_count": 0, "peak_count": 0,
    "avg_wait_time": 0.0, "staff_count": 0, "queue_length": 0,
    "zone_counts": {"entrance": 0, "waiting": 0, "dining": 0, "vip": 0, "prayer": 0},
    "hourly_stats": defaultdict(int),
    "insights": {"warnings": [], "recommendations": [], "alerts": []},
    "metrics": {"staff_utilization": 0.0, "service_speed": 0.0, "customer_satisfaction": 0.0},
}

# ── SERVER-SIDE CAMERA ──
server_cam: dict = {"cap": None, "running": False, "latest_frame": None}

# ── ACTIVE WEBSOCKETS ──
ws_clients: set = set()


# ── AI INITIALISATION ──

def _load_ai() -> None:
    global ai_ready, ai_loading, ai_error
    global detector, tracker, footfall_counter, staff_detector_obj, zone_manager

    ai_loading = True
    ai_error = None
    try:
        from utils.analytics import (FootfallCounter, StaffDetector,
                                      WaitTimeAnalyzer, ZoneManager)
        from utils.detection import PersonDetector
        from utils.tracking import DeepSORTTracker

        detector = PersonDetector(model_path="yolov8n.pt", confidence=0.45)
        tracker = DeepSORTTracker(max_age=30, min_hits=2, iou_threshold=0.3)
        footfall_counter = FootfallCounter(
            line_start=(0, 360), line_end=(1280, 360)
        )
        staff_detector_obj = StaffDetector(
            hsv_lower=(0, 0, 0), hsv_upper=(180, 50, 80)
        )

        zone_manager = ZoneManager()
        zone_manager.add_zone("entrance", [(0, 0), (213, 0), (213, 720), (0, 720)])
        zone_manager.add_zone("waiting", [(213, 0), (427, 0), (427, 720), (213, 720)])
        zone_manager.add_zone("dining", [(427, 0), (896, 0), (896, 720), (427, 720)])
        zone_manager.add_zone("vip", [(896, 0), (1280, 0), (1280, 360), (896, 360)])
        zone_manager.add_zone("prayer", [(896, 360), (1280, 360), (1280, 720), (896, 720)])

        ai_ready = True
        print("[AI] Ready — YOLOv8n + DeepSORT loaded")
    except Exception as exc:
        ai_error = str(exc)
        ai_ready = False
        print(f"[AI] Init failed: {exc}")
    finally:
        ai_loading = False


# ── FRAME PROCESSING ──

def _process_frame(frame: np.ndarray) -> dict:
    """Run full detection + tracking pipeline on one frame. Returns result dict."""
    if not ai_ready:
        return {
            "persons_detected": 0, "customers": 0, "staff": 0,
            "annotated": None,
            "footfall": {"entries": 0, "exits": 0, "net": 0},
            "zone_counts": {},
            "analysis": {"entrances": 0, "exits": 0, "queue_count": 0, "staff_visible": 0},
        }

    t = time.time()
    h, w = frame.shape[:2]

    # Scale footfall line to actual frame size
    line_y = int(h * 0.5)
    line_start = (0, line_y)
    line_end = (w, line_y)

    boxes, confidences, _ = detector.detect(frame)
    tracks, track_boxes, track_ids = tracker.update(boxes, confidences, frame, t)

    staff_flags = staff_detector_obj.detect_staff(frame, boxes) if boxes else []
    staff_count = sum(1 for f in staff_flags if f)
    customer_count = max(0, len(tracks) - staff_count)

    zone_counts: dict = defaultdict(int)
    for track in tracks:
        # Normalise center to 1280x720 space for zone lookup
        cx = int(track.center[0] * 1280 / w)
        cy = int(track.center[1] * 720 / h)
        zone = zone_manager.get_zone_at((cx, cy))
        if zone:
            zone_counts[zone] += 1

    # Footfall line crossing (using normalised coords)
    for track in tracks:
        if len(track.history) >= 2:
            px = int(track.history[-2][0] * 1280 / w)
            py = int(track.history[-2][1] * 720 / h)
            cx = int(track.center[0] * 1280 / w)
            cy = int(track.center[1] * 720 / h)
            footfall_counter.update(track.track_id, (px, py), (cx, cy))

    fs = footfall_counter.get_stats()

    # Update global stats
    current_stats["entries"] = fs["entries"]
    current_stats["exits"] = fs["exits"]
    current_stats["current_count"] = len(tracks)
    current_stats["peak_count"] = max(current_stats["peak_count"], len(tracks))
    current_stats["staff_count"] = staff_count
    current_stats["queue_length"] = int(zone_counts.get("waiting", 0))
    for z in ["entrance", "waiting", "dining", "vip", "prayer"]:
        current_stats["zone_counts"][z] = int(zone_counts.get(z, 0))
    current_stats["hourly_stats"][datetime.now().hour] += 1
    util = min(100.0, (customer_count / max(1, staff_count)) * 20.0) if staff_count else 0.0
    current_stats["metrics"]["staff_utilization"] = round(util, 1)
    current_stats["metrics"]["service_speed"] = round(max(0.0, 100.0 - current_stats["queue_length"] * 8.0), 1)
    current_stats["metrics"]["customer_satisfaction"] = round(min(100.0, max(0.0, 95.0 - current_stats["avg_wait_time"] * 1.5)), 1)

    # Draw annotated frame
    annotated = frame.copy()

    # Zone overlays (semi-transparent)
    overlay = annotated.copy()
    zone_colors_bgr = {
        "entrance": (0, 200, 255),
        "waiting": (0, 140, 255),
        "dining": (0, 255, 100),
        "vip": (0, 215, 255),
        "prayer": (255, 200, 0),
    }
    # map 1280x720 zone coords back to frame coords
    for zname, zobj in zone_manager.zones.items():
        pts = np.array(
            [[(int(px * w / 1280), int(py * h / 720)) for px, py in zobj.points]],
            dtype=np.int32,
        )
        color = zone_colors_bgr.get(zname, (100, 100, 100))
        cv2.fillPoly(overlay, pts, color)
    cv2.addWeighted(overlay, 0.12, annotated, 0.88, 0, annotated)

    # Footfall line
    cv2.line(annotated, line_start, line_end, (0, 0, 220), 2)
    cv2.putText(annotated, f"IN:{fs['entries']}  OUT:{fs['exits']}", (8, line_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 220), 1)

    # Bounding boxes
    for i, (box, tid) in enumerate(zip(track_boxes, track_ids)):
        x1, y1, x2, y2 = map(int, box)
        is_staff = i < len(staff_flags) and staff_flags[i]
        color = (60, 80, 255) if is_staff else (60, 255, 100)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"STAFF #{tid}" if is_staff else f"#{tid}"
        cv2.putText(annotated, label, (x1, max(12, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1)

    # Top-left HUD
    hud_lines = [
        (f"PEOPLE: {len(tracks)}", (255, 255, 255)),
        (f"STAFF:  {staff_count}", (60, 80, 255)),
        (f"QUEUE:  {zone_counts.get('waiting', 0)}", (0, 200, 255)),
    ]
    for idx, (text, clr) in enumerate(hud_lines):
        cv2.putText(annotated, text, (8, 22 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(annotated, text, (8, 22 + idx * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, clr, 1)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 68])
    annotated_b64 = base64.b64encode(buf).decode()

    return {
        "persons_detected": len(tracks),
        "customers": customer_count,
        "staff": staff_count,
        "annotated": annotated_b64,
        "footfall": fs,
        "zone_counts": {k: int(v) for k, v in zone_counts.items()},
        "analysis": {
            "entrances": fs["entries"],
            "exits": fs["exits"],
            "queue_count": int(zone_counts.get("waiting", 0)),
            "staff_visible": staff_count,
        },
    }


# ── ROUTES ──

@app.get("/dashboard", response_class=FileResponse)
async def dashboard():
    return FileResponse(os.path.join(_dir, "pro.html"))


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "ai_ready": ai_ready,
        "ai_loading": ai_loading,
        "ai_error": ai_error,
        "version": "3.0.0",
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/ai/init")
async def init_ai():
    if ai_ready:
        return {"status": "already_ready"}
    if ai_loading:
        return {"status": "loading"}
    threading.Thread(target=_load_ai, daemon=True).start()
    return {"status": "loading"}


@app.get("/ai/status")
async def ai_status():
    return {"ready": ai_ready, "loading": ai_loading, "error": ai_error}


@app.get("/analytics")
async def get_analytics():
    return {
        **current_stats,
        "hourly_stats": dict(current_stats["hourly_stats"]),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/analytics/realtime")
async def get_realtime():
    h = datetime.now().hour
    period_map = [(range(5, 7), "fajr"), (range(9, 12), "morning"),
                  (range(12, 15), "lunch"), (range(17, 21), "evening"), (range(21, 24), "night")]
    period = next((p for r, p in period_map if h in r), "off_peak")
    return {
        "current_customers": current_stats["current_count"],
        "queue_length": current_stats["queue_length"],
        "staff_active": current_stats["staff_count"],
        "avg_wait": current_stats["avg_wait_time"],
        "hour": h,
        "period": period,
        "ai_ready": ai_ready,
    }


@app.get("/analytics/zones")
async def get_zones():
    maxes = {"entrance": 10, "waiting": 20, "dining": 100, "vip": 15, "prayer": 10}
    return {
        "zones": current_stats["zone_counts"],
        "capacity": {
            z: {"current": current_stats["zone_counts"].get(z, 0), "max": m}
            for z, m in maxes.items()
        },
    }


@app.get("/analytics/hourly")
async def get_hourly():
    h = dict(current_stats["hourly_stats"])
    return {
        "hourly_distribution": h,
        "peak_hour": max(h, key=h.get) if h else 0,
        "total_entries": sum(h.values()),
    }


@app.get("/analytics/insights")
async def get_insights():
    ins: dict = {
        "warnings": [],
        "recommendations": [],
        "alerts": list(current_stats["insights"]["alerts"][-10:]),
    }
    q = current_stats["queue_length"]
    wait = current_stats["avg_wait_time"]
    curr = current_stats["current_count"]
    staff = current_stats["staff_count"]

    if q > 5:
        ins["warnings"].append(f"Queue at {q} — consider opening additional counter")
    if wait > 15:
        ins["warnings"].append(f"Avg wait {wait:.1f}m exceeds 15m target")
    if staff > 0:
        ratio = curr / staff
        if ratio > 5:
            ins["recommendations"].append(f"Customer:staff ratio {ratio:.1f}:1 — add staff")
        elif ratio < 2 and curr > 5:
            ins["recommendations"].append("Low customer:staff ratio — consider reducing staff")
    if not staff and curr > 10:
        ins["warnings"].append("No staff detected with 10+ customers — check detection")
    return ins


@app.post("/analytics/update")
async def update_analytics(
    entries: int = Form(0), exits: int = Form(0),
    current_count: int = Form(0), wait_time: float = Form(0.0),
    staff_count: int = Form(0), queue: int = Form(0),
):
    if entries > 0:
        current_stats["entries"] = entries
        current_stats["hourly_stats"][datetime.now().hour] += entries
    if exits > 0:
        current_stats["exits"] = exits
    if current_count >= 0:
        current_stats["current_count"] = current_count
        current_stats["peak_count"] = max(current_stats["peak_count"], current_count)
    if wait_time > 0:
        current_stats["avg_wait_time"] = wait_time
    if staff_count >= 0:
        current_stats["staff_count"] = staff_count
        current_stats["metrics"]["staff_utilization"] = round(
            min(100.0, (current_count / max(1, staff_count)) * 20.0), 1
        )
    if queue >= 0:
        current_stats["queue_length"] = queue
    return {"status": "updated", "timestamp": datetime.now().isoformat()}


@app.post("/analytics/reset")
async def reset_analytics():
    global current_stats
    if ai_ready and footfall_counter:
        footfall_counter.reset()
    if ai_ready and tracker:
        tracker.reset()
    current_stats = {
        "entries": 0, "exits": 0, "current_count": 0, "peak_count": 0,
        "avg_wait_time": 0.0, "staff_count": 0, "queue_length": 0,
        "zone_counts": {"entrance": 0, "waiting": 0, "dining": 0, "vip": 0, "prayer": 0},
        "hourly_stats": defaultdict(int),
        "insights": {"warnings": [], "recommendations": [], "alerts": []},
        "metrics": {"staff_utilization": 0.0, "service_speed": 0.0, "customer_satisfaction": 0.0},
    }
    return {"status": "reset"}


@app.post("/video/analyze")
async def analyze_frame(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse(content={"error": "Failed to decode image"}, status_code=400)

    result = await asyncio.get_event_loop().run_in_executor(None, _process_frame, frame)
    result["frame_size"] = {"width": frame.shape[1], "height": frame.shape[0]}
    return JSONResponse(content=result)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    ws_clients.add(websocket)
    try:
        while True:
            raw = await websocket.receive_bytes()
            nparr = np.frombuffer(raw, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_json({"error": "bad_frame"})
                continue
            result = await asyncio.get_event_loop().run_in_executor(None, _process_frame, frame)
            await websocket.send_json(result)
    except WebSocketDisconnect:
        ws_clients.discard(websocket)
    except Exception:
        ws_clients.discard(websocket)


# ── SERVER-SIDE CAMERA ──

def _server_cam_thread():
    cap = server_cam["cap"]
    while server_cam["running"] and cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            server_cam["latest_frame"] = frame
        else:
            time.sleep(0.02)


@app.post("/camera/start")
async def start_server_camera(source: str = Form("0")):
    if server_cam["running"]:
        return {"status": "already_running", "source": server_cam.get("source")}
    src = int(source) if source.strip().isdigit() else source.strip()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return JSONResponse(content={"error": f"Cannot open source: {source}"}, status_code=400)
    server_cam["cap"] = cap
    server_cam["source"] = source
    server_cam["running"] = True
    threading.Thread(target=_server_cam_thread, daemon=True).start()
    return {"status": "started", "source": source}


@app.post("/camera/stop")
async def stop_server_camera():
    server_cam["running"] = False
    if server_cam["cap"]:
        server_cam["cap"].release()
        server_cam["cap"] = None
    return {"status": "stopped"}


@app.get("/camera/stream")
async def mjpeg_stream():
    async def generate():
        while server_cam["running"]:
            frame = server_cam["latest_frame"]
            if frame is not None:
                result = await asyncio.get_event_loop().run_in_executor(None, _process_frame, frame)
                if result.get("annotated"):
                    jpeg_bytes = base64.b64decode(result["annotated"])
                else:
                    _, buf = cv2.imencode(".jpg", frame)
                    jpeg_bytes = buf.tobytes()
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            await asyncio.sleep(0.033)

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.post("/config")
async def configure(language: str = Form("en"), ramadan_mode: bool = Form(False)):
    return {"status": "configured", "language": language, "ramadan_mode": ramadan_mode}


@app.post("/webhook/alerts")
async def receive_alert(type: str = Form(...), message: str = Form(...)):
    alert = {"type": type, "message": message, "timestamp": datetime.now().isoformat()}
    current_stats["insights"]["alerts"].append(alert)
    return {"status": "received", "alert": alert}


# ── TRAINING ──

def _run_training(data_yaml: str, epochs: int, batch: int, imgsz: int,
                  device: str, freeze: int, weights: str) -> None:
    global train_state, detector
    from ultralytics import YOLO

    train_state.update({"status": "running", "epoch": 0, "total_epochs": epochs,
                        "loss": None, "map50": None, "map50_95": None,
                        "weights_path": None, "error": None})

    def _on_epoch_end(trainer):
        loss_val = float(trainer.loss.item()) if hasattr(trainer.loss, "item") else float(trainer.loss)
        train_state["epoch"] = trainer.epoch + 1
        train_state["loss"]  = round(loss_val, 4)

    try:
        model = YOLO(weights)
        model.add_callback("on_train_epoch_end", _on_epoch_end)

        _dir_app = os.path.dirname(os.path.abspath(__file__))
        run_dir  = os.path.join(_dir_app, "..", "runs", "train")

        results = model.train(
            data=data_yaml, epochs=epochs, batch=batch, imgsz=imgsz,
            device=device, freeze=freeze,
            lr0=0.001, lrf=0.01, momentum=0.937, weight_decay=0.0005,
            warmup_epochs=3, patience=30, cos_lr=True,
            project=run_dir, name="restaurant_ai", exist_ok=True,
            save=True, verbose=False, workers=0,
            pretrained=True, optimizer="AdamW", close_mosaic=10,
        )

        best = os.path.join(run_dir, "restaurant_ai", "weights", "best.pt")

        # Evaluate
        if os.path.exists(best):
            val_model = YOLO(best)
            metrics   = val_model.val(data=data_yaml, verbose=False)
            train_state["map50"]    = round(float(metrics.box.map50), 4)
            train_state["map50_95"] = round(float(metrics.box.map),   4)

        train_state["status"]       = "done"
        train_state["weights_path"] = best
        print(f"[train] Done — best weights: {best}")
    except Exception as exc:
        train_state["status"] = "error"
        train_state["error"]  = str(exc)
        print(f"[train] Failed: {exc}")


@app.post("/train/start")
async def start_training(
    data_yaml: str  = Form(None),
    epochs:    int  = Form(100),
    batch:     int  = Form(16),
    imgsz:     int  = Form(640),
    device:    str  = Form("0"),
    freeze:    int  = Form(10),
    weights:   str  = Form("yolov8n.pt"),
):
    if train_state["status"] == "running":
        return {"status": "already_running", "epoch": train_state["epoch"]}

    _dir_app = os.path.dirname(os.path.abspath(__file__))
    resolved_yaml = data_yaml or os.path.join(
        _dir_app, "..", "data", "restaurant", "data.yaml"
    )
    resolved_yaml = os.path.abspath(resolved_yaml)

    if not os.path.exists(resolved_yaml):
        return JSONResponse(
            content={"error": f"data.yaml not found: {resolved_yaml}. "
                               "Run scripts/collect_and_label.py first."},
            status_code=400,
        )

    threading.Thread(
        target=_run_training,
        args=(resolved_yaml, epochs, batch, imgsz, device, freeze, weights),
        daemon=True,
    ).start()
    return {"status": "started", "epochs": epochs, "data_yaml": resolved_yaml}


@app.get("/train/status")
async def get_train_status():
    return {**train_state, "timestamp": datetime.now().isoformat()}


@app.post("/model/switch")
async def switch_model(weights_path: str = Form(...)):
    global detector, ai_ready, ai_error

    if not os.path.exists(weights_path):
        return JSONResponse(
            content={"error": f"Weights not found: {weights_path}"},
            status_code=400,
        )

    try:
        from utils.detection import PersonDetector
        detector  = PersonDetector(model_path=weights_path, confidence=0.45)
        ai_ready  = True
        ai_error  = None
        print(f"[model] Switched to {weights_path}")
        return {"status": "switched", "weights": weights_path}
    except Exception as exc:
        ai_error = str(exc)
        return JSONResponse(content={"error": str(exc)}, status_code=500)


@app.get("/")
async def root():
    return HTMLResponse("""<!DOCTYPE html>
<html><head><meta charset=UTF-8><title>Restaurant AI</title>
<style>body{background:#050505;color:#e0e0e0;font-family:monospace;padding:40px;}</style></head>
<body>
<h2 style="color:#ff4500">Restaurant AI — UAE Edition v3.0</h2>
<p><a href="/dashboard" style="color:#ff4500">→ Open Dashboard</a> &nbsp;
   <a href="/docs" style="color:#00d4ff">→ API Docs</a></p>
<p>AI: <span id="s" style="color:#666">checking...</span></p>
<script>
fetch('/ai/status').then(r=>r.json()).then(d=>{
  const s=document.getElementById('s');
  s.textContent=d.ready?'READY ✓':(d.loading?'LOADING...':(d.error?'ERROR: '+d.error:'Not loaded'));
  s.style.color=d.ready?'#39ff14':d.loading?'#ffbf00':'#ff1a1a';
});
</script>
</body></html>""")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
