"""
Custom Model Training for Restaurant AI
Fine-tune YOLOv8 on custom restaurant dataset
"""
import os
import yaml
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np


class RestaurantDataset:
    """Create and manage restaurant dataset for training."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'images'
        self.labels_path = self.dataset_path / 'labels'
        
    def create_structure(self):
        """Create dataset directory structure."""
        for split in ['train', 'val', 'test']:
            (self.images_path / split).mkdir(parents=True, exist_ok=True)
            (self.labels_path / split).mkdir(parents=True, exist_ok=True)
            
    def add_image(self, image_path: str, split: str = 'train',
                  bbox: List = None, class_id: int = 0):
        """Add image with annotations."""
        img = cv2.imread(image_path)
        if img is None:
            return False
            
        h, w = img.shape[:2]
        filename = Path(image_path).name
        
        target_img = self.images_path / split / filename
        cv2.imwrite(str(target_img), img)
        
        if bbox:
            label_file = self.labels_path / split / (Path(image_path).stem + '.txt')
            with open(label_file, 'w') as f:
                x_center = (bbox[0] + bbox[2]) / 2 / w
                y_center = (bbox[1] + bbox[3]) / 2 / h
                width = (bbox[2] - bbox[0]) / w
                height = (bbox[3] - bbox[1]) / h
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
        return True
    
    def convert_coco(self, coco_json: str, split: str = 'train'):
        """Convert COCO format annotations to YOLO format."""
        import json
        
        with open(coco_json, 'r') as f:
            coco = json.load(f)
            
        for img_info in coco['images']:
            img_id = img_info['id']
            img_name = img_info['file_name']
            w, h = img_info['width'], img_info['height']
            
            annotations = [a for a in coco['annotations'] if a['image_id'] == img_id]
            
            label_lines = []
            for ann in annotations:
                bbox = ann['bbox']
                x = bbox[0] / w
                y = bbox[1] / h
                bw = bbox[2] / w
                bh = bbox[3] / h
                cx = x + bw / 2
                cy = y + bh / 2
                
                category_id = ann['category_id'] - 1
                label_lines.append(f"{category_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            
            if label_lines:
                label_file = self.labels_path / split / (Path(img_name).stem + '.txt')
                with open(label_file, 'w') as f:
                    f.writelines(label_lines)
                    

class DataAugmentor:
    """Augment training data."""
    
    def __init__(self):
        pass
    
    def random_brightness(self, image: np.ndarray, factor: float = 0.3) -> np.ndarray:
        """Random brightness adjustment."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv = hsv.astype(np.float32)
        
        delta = np.random.uniform(-factor, factor)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * (1 + delta), 0, 255)
        
        hsv = hsv.astype(np.uint8)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    def random_contrast(self, image: np.ndarray, factor: float = 0.3) -> np.ndarray:
        """Random contrast adjustment."""
        alpha = 1 + np.random.uniform(-factor, factor)
        return np.clip(image.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    
    def random_blur(self, image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """Random blur."""
        if np.random.random() > 0.5:
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return image
    
    def random_rotation(self, image: np.ndarray, angle: float = 15) -> np.ndarray:
        """Random rotation."""
        angle_rad = np.random.uniform(-angle, angle) * np.pi / 180
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        matrix = cv2.getRotationMatrix2D(center, angle * 180 / np.pi, 1.0)
        return cv2.warpAffine(image, matrix, (w, h))
    
    def random_crop(self, image: np.ndarray, bbox: List, 
                   crop_ratio: float = 0.2) -> Tuple[np.ndarray, List]:
        """Random crop around bounding box."""
        if np.random.random() > 0.5:
            return image, bbox
            
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox
        
        crop_w = int(w * crop_ratio)
        crop_h = int(h * crop_ratio)
        
        new_x1 = max(0, x1 - np.random.randint(0, crop_w))
        new_y1 = max(0, y1 - np.random.randint(0, crop_h))
        new_x2 = min(w, x2 + np.random.randint(0, crop_w))
        new_y2 = min(h, y2 + np.random.randint(0, crop_h))
        
        cropped = image[new_y1:new_y2, new_x1:new_x2]
        
        new_bbox = [x1 - new_x1, y1 - new_y1, x2 - new_x1, y2 - new_y1]
        
        return cropped, new_bbox
    
    def flip_horizontal(self, image: np.ndarray, bbox: List = None) -> Tuple[np.ndarray, List]:
        """Horizontal flip."""
        if np.random.random() > 0.5:
            flipped = cv2.flip(image, 1)
            if bbox:
                w = image.shape[1]
                new_bbox = [w - bbox[2], bbox[1], w - bbox[0], bbox[3]]
                return flipped, new_bbox
        return image, bbox
    
    def augment_image(self, image: np.ndarray, bbox: List = None) -> Tuple[np.ndarray, List]:
        """Apply all augmentations."""
        image = self.random_brightness(image)
        image = self.random_contrast(image)
        image = self.random_blur(image)
        image, bbox = self.flip_horizontal(image, bbox)
        
        return image, bbox


class TrainingConfig:
    """Training configuration for YOLO model."""
    
    RESTAURANT_CLASSES = {
        'person': 0,
        'staff': 1,
        'customer': 2,
        'group': 3,
        'queue': 4
    }
    
    def __init__(self, model_size: str = 'n'):
        self.model_size = model_size
        self.data_config = {
            'path': './data/restaurant',
            'train': 'images/train',
            'val': 'images/val',
            'test': 'images/test',
            'names': self.RESTAURANT_CLASSES
        }
        
    def create_data_yaml(self, output_path: str = 'data.yaml'):
        """Create data.yaml for training."""
        with open(output_path, 'w') as f:
            yaml.dump(self.data_config, f, default_flow_style=False)
        return output_path
    
    def get_training_args(self, epochs: int = 100, batch_size: int = 16,
                         image_size: int = 640, device: str = 'cuda'):
        """Get training arguments."""
        return {
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'device': device,
            'name': 'restaurant_ai',
            'exist_ok': True,
            'optimizer': 'SGD',
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3.0,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            'patience': 50,
            'save': True,
            'save_period': -1,
            'cache': False,
            'verbose': True,
            'workers': 8,
            'project': './runs/train',
            'pretrained': True
        }
    

def train_model(data_yaml: str = 'data.yaml', model_size: str = 'n',
               epochs: int = 100, device: str = 'cuda'):
    """Train YOLO model."""
    from ultralytics import YOLO
    
    model = YOLO(f'yolov8{model_size}.pt')
    
    config = TrainingConfig(model_size)
    args = config.get_training_args(epochs=epochs, device=device)
    
    results = model.train(data=data_yaml, **args)
    
    return results


def export_model(weights_path: str, format: str = 'onnx'):
    """Export trained model."""
    from ultralytics import YOLO
    
    model = YOLO(weights_path)
    model.export(format=format)
    

def evaluate_model(weights_path: str, data_yaml: str = 'data.yaml'):
    """Evaluate trained model."""
    from ultralytics import YOLO
    
    model = YOLO(weights_path)
    metrics = model.val(data=data_yaml)
    
    return {
        'map50': metrics.box.map50,
        'map50_95': metrics.box.map,
        'precision': metrics.box.mp,
        'recall': metrics.box.mr
    }


def create_sample_dataset(output_dir: str = './data/restaurant'):
    """Create sample dataset for testing."""
    import random
    
    dataset = RestaurantDataset(output_dir)
    dataset.create_structure()
    
    augmentor = DataAugmentor()
    
    for i in range(50):
        w, h = 640, 480
        img = np.random.randint(100, 200, (h, w, 3), dtype=np.uint8)
        
        num_persons = random.randint(1, 4)
        bboxes = []
        
        for _ in range(num_persons):
            x1 = random.randint(50, w - 150)
            y1 = random.randint(50, h - 150)
            x2 = x1 + random.randint(50, 100)
            y2 = y1 + random.randint(80, 150)
            
            color = (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
            
            bboxes.append([x1, y1, x2, y2])
        
        split = 'train' if i < 40 else 'val'
        
        img_path = dataset.images_path / split / f'image_{i:04d}.jpg'
        cv2.imwrite(str(img_path), img)
        
        label_path = dataset.labels_path / split / f'image_{i:04d}.txt'
        with open(label_path, 'w') as f:
            for bbox in bboxes:
                cx = ((bbox[0] + bbox[2]) / 2) / w
                cy = ((bbox[1] + bbox[3]) / 2) / h
                bw = (bbox[2] - bbox[0]) / w
                bh = (bbox[3] - bbox[1]) / h
                f.write(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    
    return dataset


if __name__ == '__main__':
    print("Creating sample dataset...")
    dataset = create_sample_dataset()
    print(f"Dataset created at {dataset.dataset_path}")
    
    config = TrainingConfig()
    config.create_data_yaml()
    print("data.yaml created")
    
    print("\nTo train the model, run:")
    print("  from utils.train import train_model")
    print("  results = train_model('data.yaml', model_size='n', epochs=100)")
