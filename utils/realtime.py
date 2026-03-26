"""
Real-time Video Processor
Handles video streams from CCTV, RTSP, Webcam with frame processing pipeline
"""
import cv2
import numpy as np
import threading
import queue
from typing import List, Tuple, Callable, Optional
import time
from dataclasses import dataclass


@dataclass
class FrameData:
    """Container for frame data."""
    frame: np.ndarray
    frame_id: int
    timestamp: float
    boxes: List[np.ndarray] = None
    ids: List[int] = None
    confidences: List[float] = None


class VideoStream:
    """Handle video stream from various sources."""
    
    def __init__(self, source, queue_size: int = 30):
        self.source = source
        self.capture = None
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.running = False
        self.thread = None
        
    def start(self):
        """Start streaming."""
        if isinstance(self.source, int):
            self.capture = cv2.VideoCapture(self.source)
        elif self.source.startswith('rtsp://'):
            self.capture = cv2.VideoCapture(self.source)
        elif self.source.endswith(('.mp4', '.avi')):
            self.capture = cv2.VideoCapture(self.source)
        else:
            self.capture = cv2.VideoCapture(0)
            
        if not self.capture.isOpened():
            raise ValueError(f"Cannot open video source: {self.source}")
            
        self.running = True
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        
    def _read_frames(self):
        """Background thread for reading frames."""
        frame_id = 0
        while self.running:
            ret, frame = self.capture.read()
            if not ret:
                if isinstance(self.source, str) and not self.source.startswith('rtsp'):
                    self.running = False
                    break
                continue
                
            frame_data = FrameData(
                frame=frame,
                frame_id=frame_id,
                timestamp=time.time()
            )
            
            try:
                self.frame_queue.put_nowait(frame_data)
            except queue.Full:
                try:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(frame_data)
                except:
                    pass
                    
            frame_id += 1
            
    def get_frame(self, timeout: float = 1.0) -> Optional[FrameData]:
        """Get next frame."""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def stop(self):
        """Stop streaming."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
        if self.capture:
            self.capture.release()


class ProcessorPipeline:
    """Process frames through detection/tracking pipeline."""
    
    def __init__(self, detector, tracker):
        self.detector = detector
        self.tracker = tracker
        self.processors: List[Callable] = []
        self.output_queue = queue.Queue(maxsize=30)
        
    def add_processor(self, processor: Callable):
        """Add frame processor."""
        self.processors.append(processor)
        
    def process(self, frame_data: FrameData) -> FrameData:
        """Process single frame through pipeline."""
        frame = frame_data.frame
        
        boxes, confidences, _ = self.detector.detect(frame)
        
        tracks, track_boxes, track_ids = self.tracker.update(
            boxes, confidences, frame, frame_data.timestamp
        )
        
        frame_data.boxes = track_boxes
        frame_data.ids = track_ids
        frame_data.confidences = confidences if confidences else []
        
        for processor in self.processors:
            frame = processor(frame, frame_data)
            
        return frame_data
    
    def start_processing(self, stream: VideoStream):
        """Start processing stream."""
        def process_loop():
            while stream.running:
                frame_data = stream.get_frame()
                if frame_data:
                    processed = self.process(frame_data)
                    try:
                        self.output_queue.put_nowait(processed)
                    except queue.Full:
                        pass
                        
        thread = threading.Thread(target=process_loop, daemon=True)
        thread.start()


class OverlayRenderer:
    """Render overlays on frames."""
    
    def __init__(self):
        self.colors = {
            'person': (0, 255, 0),
            'staff': (255, 0, 0),
            'vip': (255, 215, 0),
            'zone': (0, 255, 255)
        }
        
    def draw_boxes(self, frame: np.ndarray, frame_data: FrameData) -> np.ndarray:
        """Draw bounding boxes."""
        output = frame.copy()
        
        if frame_data.boxes:
            for i, box in enumerate(frame_data.boxes):
                x1, y1, x2, y2 = map(int, box)
                
                if frame_data.ids:
                    color = self.colors['person']
                    label = f"ID:{frame_data.ids[i]}"
                else:
                    color = self.colors['person']
                    label = f"{frame_data.confidences[i]:.2f}" if i < len(frame_data.confidences) else ""
                
                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
                
                if label:
                    cv2.putText(output, label, (x1, y1 - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                                      
        return output
    
    def draw_zones(self, frame: np.ndarray, zones: dict) -> np.ndarray:
        """Draw zone polygons."""
        output = frame.copy()
        
        for name, zone in zones.items():
            points = np.array(zone['points'], np.int32)
            color = zone.get('color', (0, 255, 255))
            
            cv2.polylines(output, [points], True, color, 2)
            cv2.fillPoly(output, [points], color=(color[0]//3, color[1]//3, color[2]//3))
            
        return output
    
    def draw_stats(self, frame: np.ndarray, stats: dict, position: str = 'top-left') -> np.ndarray:
        """Draw statistics overlay."""
        output = frame.copy()
        
        if position == 'top-left':
            x, y = 10, 30
        else:
            x, y = 10, frame.shape[0] - 30
            
        for key, value in stats.items():
            text = f"{key}: {value}"
            cv2.putText(output, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output, text, (x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
            y += 25
            
        return output
    
    def draw_footfall_line(self, frame: np.ndarray, start: Tuple[int, int], 
                          end: Tuple[int, int]) -> np.ndarray:
        """Draw entry/exit line."""
        output = frame.copy()
        cv2.line(output, start, end, (0, 0, 255), 3)
        
        mid = ((start[0] + end[0]) // 2, (start[1] + end[1]) // 2)
        cv2.putText(output, "ENTRY/EXIT LINE", (mid[0] - 50, mid[1] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                   
        return output


class VideoRecorder:
    """Record processed video."""
    
    def __init__(self, output_path: str, fps: int = 30, frame_size: Tuple[int, int] = None):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size
        self.writer = None
        self.recording = False
        
    def start(self, frame: np.ndarray):
        """Start recording."""
        if self.frame_size is None:
            self.frame_size = (frame.shape[1], frame.shape[0])
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.frame_size)
        self.recording = True
        
    def write(self, frame: np.ndarray):
        """Write frame."""
        if self.recording and self.writer:
            self.writer.write(frame)
            
    def stop(self):
        """Stop recording."""
        self.recording = False
        if self.writer:
            self.writer.release()
            self.writer = None


class StreamManager:
    """Manage multiple video streams."""
    
    def __init__(self):
        self.streams: dict = {}
        self.processors: dict = {}
        self.overlay_renderers: dict = {}
        
    def add_stream(self, stream_id: str, source, detector, tracker):
        """Add new stream."""
        stream = VideoStream(source)
        processor = ProcessorPipeline(detector, tracker)
        
        self.streams[stream_id] = stream
        self.processors[stream_id] = processor
        self.overlay_renderers[stream_id] = OverlayRenderer()
        
        return stream, processor
    
    def start_stream(self, stream_id: str):
        """Start stream processing."""
        if stream_id in self.streams:
            self.streams[stream_id].start()
            self.processors[stream_id].start_processing(self.streams[stream_id])
            
    def get_frame(self, stream_id: str) -> Optional[FrameData]:
        """Get processed frame."""
        if stream_id in self.processors:
            try:
                return self.processors[stream_id].output_queue.get_nowait()
            except queue.Empty:
                return None
        return None
    
    def stop_stream(self, stream_id: str):
        """Stop stream."""
        if stream_id in self.streams:
            self.streams[stream_id].stop()
            
    def stop_all(self):
        """Stop all streams."""
        for stream_id in self.streams:
            self.stop_stream(stream_id)
