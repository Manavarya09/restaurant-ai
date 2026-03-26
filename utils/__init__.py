import numpy as np
import cv2


def draw_boxes(frame, boxes, ids=None, confidences=None, labels=None, color=(0, 255, 0), text_color=(255, 255, 255)):
    """Draw bounding boxes on frame with optional IDs and labels."""
    output = frame.copy()
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        label_parts = []
        if ids is not None and i < len(ids):
            label_parts.append(f"ID:{ids[i]}")
        if confidences is not None and i < len(confidences):
            label_parts.append(f"{confidences[i]:.2f}")
        if labels is not None and i < len(labels):
            label_parts.append(labels[i])
        
        if label_parts:
            label = " ".join(label_parts)
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(output, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
            cv2.putText(output, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
    
    return output


def draw_roi(frame, points, color=(0, 255, 255), thickness=2):
    """Draw region of interest polygon on frame."""
    output = frame.copy()
    pts = np.array(points, np.int32)
    cv2.polylines(output, [pts], True, color, thickness)
    cv2.fillPoly(output, [pts], color=(0, 255, 255))
    return output


def draw_line(frame, start, end, color=(0, 0, 255), thickness=3):
    """Draw a line on frame."""
    output = frame.copy()
    cv2.line(output, start, end, color, thickness)
    return output


def create_zone_mask(frame_shape, points):
    """Create a binary mask for a zone."""
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    pts = np.array(points, np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def point_in_polygon(point, polygon):
    """Check if point is inside polygon."""
    return cv2.pointPolygonTest(np.array(polygon, np.float32), point, False) >= 0


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_center(bbox):
    """Get center point of bounding box."""
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def get_timestamp():
    """Get current timestamp."""
    import time
    return time.time()


def format_time(seconds):
    """Format seconds to MM:SS."""
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"
