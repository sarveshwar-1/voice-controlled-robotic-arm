import torch
import logging
import cv2
import numpy as np
from PIL import Image

# Update object mappings to include more possible labels
OBJECT_MAPPINGS = {
    'apple': ['apple', 'fruit', 'sports ball', 'ball']  # Add more possible labels
}

def visualize_detections(image, detections):
    """Debug helper to visualize detections"""
    vis_img = image.copy()
    for obj_name, bbox in detections.items():
        if bbox is not None:
            x1, y1, x2, y2 = [int(c) for c in bbox]
            cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_img, obj_name, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite('debug_detections.jpg', vis_img)

def detect_objects(image, conf_threshold=0.3):
    """Detect objects in the image using YOLOv5."""
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.conf = conf_threshold
        
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        
        # Run inference
        results = model(image)
        
        # Initialize detections dictionary
        detections = {
            'apple': None
        }
        
        # Process results with more flexible matching
        if len(results.xyxy[0]) > 0:
            best_conf = 0
            best_detection = None
            
            for *xyxy, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)].lower()
                logging.info(f"Found {label} with confidence {conf:.2f}")
                
                # Check if label matches any of our target objects
                for obj_name, possible_labels in OBJECT_MAPPINGS.items():
                    if label in possible_labels and conf > conf_threshold:
                        # Update if this is the highest confidence detection
                        if conf > best_conf:
                            best_conf = conf
                            best_detection = (obj_name, [coord.item() for coord in xyxy])
            
            # Use the best detection
            if best_detection:
                obj_name, coords = best_detection
                detections[obj_name] = coords
                logging.info(f"Best match: {obj_name} with confidence {best_conf:.2f}")
        
        # Visualize results
        detection_view = image.copy()
        if any(v is not None for v in detections.values()):
            for obj_name, bbox in detections.items():
                if bbox is not None:
                    cv2.rectangle(detection_view, 
                                (int(bbox[0]), int(bbox[1])),
                                (int(bbox[2]), int(bbox[3])),
                                (0, 255, 0), 2)
                    cv2.putText(detection_view, 
                              f"{obj_name}: {best_conf:.2f}",
                              (int(bbox[0]), int(bbox[1])-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        cv2.imshow('Current Detection', detection_view)
        cv2.waitKey(1)
        
        return detections
        
    except Exception as e:
        logging.error(f"Detection error: {str(e)}", exc_info=True)
        return None