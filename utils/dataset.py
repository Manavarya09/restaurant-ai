"""
Dataset Collection and Management Tools
For building custom restaurant datasets
"""
import os
import cv2
import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
import time
import random


@dataclass
class Annotation:
    """Single bounding box annotation."""
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float = 1.0


class VideoDatasetExtractor:
    """Extract frames from video for dataset creation."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / 'frames'
        self.annotations = []
        
    def extract_frames(self, video_path: str, interval: int = 30,
                      max_frames: int = 100) -> List[str]:
        """Extract frames at regular intervals."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        frame_paths = []
        frame_count = 0
        saved = 0
        
        while saved < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                filename = f"frame_{saved:05d}.jpg"
                path = self.frames_dir / filename
                cv2.imwrite(str(path), frame)
                frame_paths.append(str(path))
                saved += 1
                
            frame_count += 1
            
        cap.release()
        return frame_paths
    
    def extract_from_camera(self, duration: int = 60, 
                          output_prefix: str = 'capture') -> List[str]:
        """Capture frames from camera."""
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Cannot open camera")
        
        frame_paths = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = duration * fps
        frame_count = 0
        saved = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % int(fps) == 0:
                filename = f"{output_prefix}_{saved:05d}.jpg"
                path = self.frames_dir / filename
                cv2.imwrite(str(path), frame)
                frame_paths.append(str(path))
                saved += 1
                
            frame_count += 1
            
        cap.release()
        return frame_paths


class AnnotationTool:
    """Tool for annotating images."""
    
    CATEGORIES = [
        {'id': 0, 'name': 'person', 'supercategory': 'person'},
        {'id': 1, 'name': 'staff', 'supercategory': 'person'},
        {'id': 2, 'name': 'customer', 'supercategory': 'person'},
        {'id': 3, 'name': 'group', 'supercategory': 'group'},
        {'id': 4, 'name': 'queue', 'supercategory': 'area'}
    ]
    
    def __init__(self, output_dir: str = 'annotations'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.annotations = []
        self.current_image = None
        self.current_path = None
        
    def load_image(self, image_path: str):
        """Load image for annotation."""
        self.current_path = image_path
        self.current_image = cv2.imread(image_path)
        self.current_annotations = []
        
    def add_annotation(self, x1: int, y1: int, x2: int, y2: int, 
                       class_id: int = 0):
        """Add bounding box annotation."""
        annotation = Annotation(x1, y1, x2, y2, class_id, 
                               self.CATEGORIES[class_id]['name'])
        self.current_annotations.append(annotation)
        
    def remove_last(self):
        """Remove last annotation."""
        if self.current_annotations:
            self.current_annotations.pop()
            
    def save_yolo_format(self, image_path: str):
        """Save annotations in YOLO format."""
        if not self.current_image:
            return None
            
        h, w = self.current_image.shape[:2]
        
        label_path = self.output_dir / (Path(image_path).stem + '.txt')
        
        with open(label_path, 'w') as f:
            for ann in self.current_annotations:
                cx = ((ann.x1 + ann.x2) / 2) / w
                cy = ((ann.y1 + ann.y2) / 2) / h
                bw = (ann.x2 - ann.x1) / w
                bh = (ann.y2 - ann.y1) / h
                f.write(f"{ann.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                
        return str(label_path)
    
    def save_coco_format(self, output_file: str = 'annotations.json'):
        """Save annotations in COCO format."""
        coco = {
            'images': [],
            'annotations': [],
            'categories': self.CATEGORIES
        }
        
        ann_id = 1
        for img_path in self.output_dir.glob('*.jpg'):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            
            coco['images'].append({
                'id': len(coco['images']) + 1,
                'file_name': img_path.name,
                'width': w,
                'height': h
            })
            
            label_path = self.output_dir / (img_path.stem + '.txt')
            if label_path.exists():
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls, cx, cy, bw, bh = map(float, parts)
                            x1 = int((cx - bw/2) * w)
                            y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w)
                            y2 = int((cy + bh/2) * h)
                            
                            coco['annotations'].append({
                                'id': ann_id,
                                'image_id': len(coco['images']),
                                'category_id': int(cls),
                                'bbox': [x1, y1, x2-x1, y2-y1],
                                'area': (x2-x1) * (y2-y1),
                                'iscrowd': 0
                            })
                            ann_id += 1
                            
        with open(output_file, 'w') as f:
            json.dump(coco, f, indent=2)
            
        return output_file
    
    def draw_annotations(self, image: np.ndarray = None) -> np.ndarray:
        """Draw annotations on image."""
        if image is None:
            image = self.current_image
            
        if image is None:
            return None
            
        output = image.copy()
        
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), 
                  (255, 255, 0), (255, 0, 255)]
        
        for ann in self.current_annotations:
            color = colors[ann.class_id % len(colors)]
            cv2.rectangle(output, (ann.x1, ann.y1), (ann.x2, ann.y2), color, 2)
            cv2.putText(output, ann.class_name, (ann.x1, ann.y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
        return output


class DataCollector:
    """Real-time data collection from video stream."""
    
    def __init__(self, detector, tracker, output_dir: str = 'collected_data'):
        self.detector = detector
        self.tracker = tracker
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.collected_frames = []
        self.collected_annotations = []
        
    def collect_from_video(self, video_path: str, 
                          sample_interval: int = 30,
                          save_images: bool = True) -> Dict:
        """Collect data from video file."""
        cap = cv2.VideoCapture(video_path)
        
        frame_data = {
            'total_frames': 0,
            'detected_persons': 0,
            'unique_tracks': 0,
            'samples': []
        }
        
        frame_count = 0
        all_track_ids = set()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = time.time()
            
            boxes, confidences, _ = self.detector.detect(frame)
            tracks, track_boxes, track_ids = self.tracker.update(
                boxes, confidences, frame, timestamp
            )
            
            all_track_ids.update(track_ids)
            
            if frame_count % sample_interval == 0 and save_images:
                img_path = self.output_dir / f'frame_{frame_count:06d}.jpg'
                cv2.imwrite(str(img_path), frame)
                
                sample = {
                    'frame': str(img_path),
                    'timestamp': timestamp,
                    'num_detections': len(boxes),
                    'num_tracks': len(tracks),
                    'track_ids': list(track_ids)
                }
                frame_data['samples'].append(sample)
                
            frame_data['total_frames'] += 1
            frame_data['detected_persons'] += len(boxes)
            
            frame_count += 1
            
        cap.release()
        
        frame_data['unique_tracks'] = len(all_track_ids)
        
        self.collected_frames = frame_data
        
        return frame_data
    
    def collect_from_camera(self, duration: int = 60) -> Dict:
        """Collect data from camera."""
        cap = cv2.VideoCapture(0)
        
        frame_data = {
            'total_frames': 0,
            'detected_persons': 0,
            'unique_tracks': 0,
            'samples': []
        }
        
        all_track_ids = set()
        start_time = time.time()
        frame_count = 0
        
        while time.time() - start_time < duration:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = time.time()
            
            boxes, confidences, _ = self.detector.detect(frame)
            tracks, track_boxes, track_ids = self.tracker.update(
                boxes, confidences, frame, timestamp
            )
            
            all_track_ids.update(track_ids)
            
            if frame_count % 30 == 0:
                img_path = self.output_dir / f'frame_{frame_count:06d}.jpg'
                cv2.imwrite(str(img_path), frame)
                
                sample = {
                    'frame': str(img_path),
                    'timestamp': timestamp,
                    'num_detections': len(boxes),
                    'track_ids': list(track_ids)
                }
                frame_data['samples'].append(sample)
                
            frame_data['total_frames'] += 1
            frame_data['detected_persons'] += len(boxes)
            frame_count += 1
            
        cap.release()
        
        frame_data['unique_tracks'] = len(all_track_ids)
        
        return frame_data
    
    def export_summary(self, output_file: str = 'collection_summary.json'):
        """Export collection summary."""
        with open(output_file, 'w') as f:
            json.dump(self.collected_frames, f, indent=2)
        return output_file


class QualityChecker:
    """Check dataset quality."""
    
    def __init__(self):
        pass
    
    def check_image_quality(self, image_path: str) -> Dict:
        """Check image quality metrics."""
        img = cv2.imread(image_path)
        if img is None:
            return {'valid': False, 'error': 'Cannot read image'}
            
        h, w = img.shape[:2]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        noisy = False
        if blur_score < 50:
            noisy = True
            
        valid = (h >= 320 and w >= 320 and 
                brightness > 30 and brightness < 225 and
                contrast > 10 and not noisy)
        
        return {
            'valid': valid,
            'width': w,
            'height': h,
            'brightness': float(brightness),
            'contrast': float(contrast),
            'blur_score': float(blur_score),
            'issues': [] if valid else self._get_issues(brightness, contrast, blur_score)
        }
    
    def _get_issues(self, brightness, contrast, blur_score):
        """Get list of quality issues."""
        issues = []
        if brightness < 30 or brightness > 225:
            issues.append('Poor brightness')
        if contrast < 10:
            issues.append('Low contrast')
        if blur_score < 50:
            issues.append('Blurry image')
        return issues
    
    def check_dataset(self, dataset_dir: str) -> Dict:
        """Check entire dataset."""
        dataset_path = Path(dataset_dir)
        
        results = {
            'total_images': 0,
            'valid_images': 0,
            'invalid_images': 0,
            'issues': [],
            'by_quality': {'good': 0, 'fair': 0, 'poor': 0}
        }
        
        for img_path in dataset_path.rglob('*.jpg'):
            results['total_images'] += 1
            
            quality = self.check_image_quality(str(img_path))
            
            if quality['valid']:
                results['valid_images'] += 1
                
                if quality['blur_score'] > 100:
                    results['by_quality']['good'] += 1
                elif quality['blur_score'] > 50:
                    results['by_quality']['fair'] += 1
                else:
                    results['by_quality']['poor'] += 1
            else:
                results['invalid_images'] += 1
                results['issues'].append({
                    'file': str(img_path),
                    'issues': quality['issues']
                })
                
        return results


def download_sample_dataset(output_dir: str = './data/external'):
    """Download sample dataset for training."""
    import urllib.request
    import zipfile
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    urls = [
        ('https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip', 'coco128.zip')
    ]
    
    for url, filename in urls:
        zip_path = output_path / filename
        
        print(f"Downloading {filename}...")
        
        try:
            urllib.request.urlretrieve(url, str(zip_path))
            
            print("Extracting...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(output_path)
                
            print(f"Downloaded to {output_path}")
            
        except Exception as e:
            print(f"Error downloading: {e}")
            
    return str(output_path)
