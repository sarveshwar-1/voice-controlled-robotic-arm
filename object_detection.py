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

def detect_red_sphere(image):
    """Detect red sphere using improved adaptive color thresholding"""
    try:
        # Convert to HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Wider HSV ranges for red detection
        lower_red1 = np.array([0, 100, 50])    # More permissive lower bounds
        upper_red1 = np.array([15, 255, 255])  # Wider hue range
        lower_red2 = np.array([160, 100, 50])  # More permissive lower bounds
        upper_red2 = np.array([180, 255, 255])
        
        # Create masks with adaptive thresholding
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Enhance mask with adaptive processing
        blurred = cv2.GaussianBlur(mask, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, 
                                     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)
        
        # Noise reduction
        kernel = np.ones((3,3), np.uint8)  # Smaller kernel
        mask = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find and filter contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Store valid contours with their scores
        valid_contours = []
        min_area = 50  # Reduced minimum area
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                perimeter = cv2.arcLength(cnt, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                # Calculate center and average color
                x, y, w, h = cv2.boundingRect(cnt)
                center = (x + w//2, y + h//2)
                mask_roi = np.zeros_like(mask)
                cv2.drawContours(mask_roi, [cnt], 0, 255, -1)
                avg_color = cv2.mean(hsv, mask=mask_roi)
                
                # Score based on multiple factors
                score = (circularity * 0.4 +  # Circularity importance
                        (min(w, h) / max(w, h)) * 0.3 +  # Aspect ratio
                        (area / (w * h)) * 0.3)  # Fill ratio
                
                if score > 0.6:  # More permissive threshold
                    valid_contours.append((cnt, score, (x, y, w, h)))
        
        # Debug visualization
        debug_view = image.copy()
        cv2.putText(debug_view, f"Found {len(valid_contours)} candidates", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if valid_contours:
            # Sort by score and take the best match
            best_contour = max(valid_contours, key=lambda x: x[1])
            cnt, score, (x, y, w, h) = best_contour
            
            # Draw detection results
            cv2.rectangle(debug_view, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(debug_view, (x + w//2, y + h//2), 5, (0, 0, 255), -1)
            cv2.putText(debug_view, f"Score: {score:.2f}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Show debug views
            cv2.imshow('Mask', mask)
            cv2.imshow('Thresholded', thresh)
            cv2.imshow('Detection Debug', debug_view)
            cv2.waitKey(1)
            
            return {'apple': [float(x), float(y), float(x+w), float(y+h)]}
        
        # Show debug views even when no detection
        cv2.imshow('Mask', mask)
        cv2.imshow('Thresholded', thresh)
        cv2.imshow('Detection Debug', debug_view)
        cv2.waitKey(1)
        return {'apple': None}
        
    except Exception as e:
        logging.error(f"Color detection error: {str(e)}")
        return None

def detect_objects(image, conf_threshold=0.3):
    """Detect objects in the image using color detection"""
    return detect_red_sphere(image)