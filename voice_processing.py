import whisper
import shutil
import logging
import os
from difflib import get_close_matches

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define valid objects and locations with command templates
VALID_OBJECTS = ['apple', 'ball', 'cube', 'box']
VALID_LOCATIONS = ['floor']  # Removed table
COMMAND_TEMPLATES = [
    "pick up the {object}",
    "put the {object} on the {location}",
    "move {object} to {location}"
]

def find_closest_match(word, valid_words, threshold=0.6):
    matches = get_close_matches(word, valid_words, n=1, cutoff=threshold)
    return matches[0] if matches else None

def extract_detection_results(detection_results, confidence_threshold=0.5):
    """Extract bounding boxes from YOLO detection results."""
    try:
        boxes = detection_results[0].boxes
        
        # Use 'xywh' if available; otherwise convert from 'xyxy' to 'xywh'
        if hasattr(boxes, "xywh"):
            coords = boxes.xywh.cpu().numpy()  # Format: [x, y, w, h]
        elif hasattr(boxes, "xyxy"):
            xyxy = boxes.xyxy.cpu().numpy()  # Format: [x_min, y_min, x_max, y_max]
            coords = xyxy.copy()
            coords[:, 2] = xyxy[:, 2] - xyxy[:, 0]  # width = x_max - x_min
            coords[:, 3] = xyxy[:, 3] - xyxy[:, 1]  # height = y_max - y_min
        else:
            logger.error("Detection boxes have no attribute 'xywh' or 'xyxy'")
            return None
        
        # Get coordinates and confidence scores
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        
        # Filter by confidence threshold
        valid_detections = conf >= confidence_threshold
        filtered_coords = coords[valid_detections]
        filtered_cls = cls[valid_detections]
        
        if len(filtered_coords) == 0:
            logger.error("No objects detected with sufficient confidence")
            return None
            
        return {
            'coordinates': filtered_coords,
            'classes': filtered_cls,
            'confidence': conf[valid_detections]
        }
        
    except AttributeError as e:
        logger.error(f"Invalid detection results format: {e}")
        return None
    except Exception as e:
        logger.error(f"Error processing detection results: {e}")
        return None

def process_voice_command(command, model):
    if command is None:
        logger.error("Received empty command")
        return None, None

    if isinstance(command, str):
        text = command.lower()
    else:
        # Validate audio file exists if path provided
        if isinstance(command, str) and not os.path.exists(command):
            logger.error(f"Audio file not found: {command}")
            return None, None

        # Check if ffmpeg is installed
        if not shutil.which('ffmpeg'):
            logger.error("ffmpeg not installed. Please install using 'brew install ffmpeg'")
            return None, None
        
        try:
            result = model.transcribe(command)
            if not result or 'text' not in result:
                logger.error("Failed to transcribe audio")
                return None, None

            text = result['text'].lower()
            logger.info(f"Transcribed text: {text}")
        except Exception as e:
            logger.error(f"Error processing voice command: {str(e)}")
            return None, None
        
    logger.info(f"Processing command: {text}")
    
    # Try to find objects and locations using fuzzy matching
    words = text.split()
    obj_name = None
    location = None
    
    for word in words:
        if not obj_name:
            obj_name = find_closest_match(word, VALID_OBJECTS)
        if not location:
            location = find_closest_match(word, VALID_LOCATIONS)
    
    if not obj_name or not location:
        logger.error(
            f"Invalid command structure. Examples:\n" +
            "\n".join(COMMAND_TEMPLATES) +
            f"\nValid objects: {VALID_OBJECTS}\n" +
            f"Valid locations: {VALID_LOCATIONS}"
        )
        return None, None
        
    logger.info(f"Recognized: object='{obj_name}', location='{location}'")
    return obj_name, location

def process_detection(detection_results):
    """Process YOLO detection results and return object locations."""
    results = extract_detection_results(detection_results)
    if not results:
        return None
        
    detections = {
        'objects': [],
        'locations': []
    }
    
    for coord, cls, conf in zip(results['coordinates'], 
                              results['classes'], 
                              results['confidence']):
        x, y, w, h = coord
        detections['objects'].append({
            'class': int(cls),
            'confidence': float(conf),
            'bbox': [float(x), float(y), float(w), float(h)]
        })
    
    return detections
