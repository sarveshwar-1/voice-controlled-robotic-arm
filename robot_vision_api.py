from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Form, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import os
import time
import shutil
from PIL import Image
import io
import uuid
import json
from typing import List, Optional
from pydantic import BaseModel

# Import the RobotVisionPipeline class
from robot_vision_pipeline import RobotVisionPipeline

app = FastAPI(
    title="Robot Vision API",
    description="API for controlling robot vision and manipulation pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory for storing images
os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize the pipeline
pipeline = RobotVisionPipeline()

# Model definitions
class CalibrationPoint(BaseModel):
    pixel_x: float
    pixel_y: float
    robot_x: float
    robot_y: float

class CalibrationData(BaseModel):
    points: List[CalibrationPoint]

class BoxPosition(BaseModel):
    pixel_x: float
    pixel_y: float

class BoundingBox(BaseModel):
    label: str
    box_2d: List[float]

class DetectionPrompt(BaseModel):
    prompt: str

class PickingPrompt(BaseModel):
    prompt: str

# Task status tracking
task_status = {}

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    # No need to initialize here since we're creating the pipeline instance at module level
    print("Robot Vision API starting up...")

@app.get("/")
async def root():
    """API root endpoint"""
    return {"message": "Robot Vision API is running"}

@app.post("/calibrate")
async def calibrate(data: CalibrationData):
    """Calibrate the system using pixel and robot coordinate pairs"""
    try:
        # Convert calibration points to the format expected by the pipeline
        pixel_coordinates = [[point.pixel_x, point.pixel_y] for point in data.points]
        robot_coordinates = [[point.robot_x, point.robot_y] for point in data.points]
        
        # Perform calibration
        homography_matrix = pipeline.calibrate(pixel_coordinates, robot_coordinates)
        
        # Return serializable version of the matrix
        return {"status": "success", "homography_matrix": homography_matrix.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calibration failed: {str(e)}")

@app.post("/connect_robot")
async def connect_robot():
    """Connect to the robot arm"""
    try:
        pipeline.connect_robot()
        return {"status": "success", "message": "Robot connected and homed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Robot connection failed: {str(e)}")

@app.post("/set_box_position")
async def set_box_position(box: BoxPosition):
    """Set the position of the destination box"""
    try:
        box_position = pipeline.set_box_position(box.pixel_x, box.pixel_y)
        return {"status": "success", "box_position": box_position}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to set box position: {str(e)}")

@app.post("/capture_image")
async def capture_image():
    """Capture an image from the camera (non-interactive API version)"""
    try:
        # Since the original capture_image method is interactive, we need an alternative
        # For API usage, we'll use a simpler capture method without UI
        cap = cv2.VideoCapture(pipeline.camera_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to capture image from camera")
        
        # Generate unique filename
        filename = f"captured_{uuid.uuid4()}.jpg"
        filepath = os.path.join("static", filename)
        
        # Save the image
        cv2.imwrite(filepath, frame)
        
        return {
            "status": "success", 
            "image_path": filepath,
            "image_url": f"/static/{filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image capture failed: {str(e)}")

@app.post("/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image to use instead of capturing one"""
    try:
        # Generate unique filename
        filename = f"uploaded_{uuid.uuid4()}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join("static", filename)
        
        # Save uploaded file
        with open(filepath, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "status": "success",
            "image_path": filepath,
            "image_url": f"/static/{filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image upload failed: {str(e)}")

@app.post("/detect_objects")
async def detect_objects(image_path: str = Form(...), prompt: str = Form(...)):
    """Detect objects in the specified image"""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Run object detection
        bounding_boxes_text, im = pipeline.detect_objects(image_path, prompt)
        
        # Plot bounding boxes on the image
        result_img = pipeline.plot_bounding_boxes(im.copy(), bounding_boxes_text)
        
        # Save the result image
        result_filename = f"detection_result_{uuid.uuid4()}.jpg"
        result_filepath = os.path.join("static", result_filename)
        result_img.save(result_filepath)
        
        # Parse the bounding boxes to return as structured data
        bounding_boxes = json.loads(pipeline.parse_json(bounding_boxes_text))
        
        return {
            "status": "success",
            "bounding_boxes": bounding_boxes,
            "result_image_url": f"/static/{result_filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Object detection failed: {str(e)}")

@app.post("/get_picking_order")
async def get_picking_order(
    image_path: str = Form(...),
    detection_prompt: str = Form(...),
    picking_prompt: str = Form(...)
):
    """Get the order in which objects should be picked"""
    try:
        # Check if image exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Run object detection
        bounding_boxes_text, im = pipeline.detect_objects(image_path, detection_prompt)
        
        # Get centroids
        centroids = pipeline.get_centroids_from_bounding_boxes(bounding_boxes_text, im)
        
        # Get picking order
        response_content = pipeline.get_objects_to_pick(centroids, picking_prompt)
        parsed_response = pipeline.parse_json_groq(response_content)
        result = json.loads(parsed_response)
        
        # Map pixel coordinates to robot coordinates
        for centroid in result["centroids"]:
            pixel_x, pixel_y = centroid["centroid"]
            robot_x, robot_y = pipeline.pixel_to_robot(pixel_x, pixel_y)
            centroid["robot_coordinates"] = [robot_x, robot_y]
        
        return {
            "status": "success",
            "picking_order": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to determine picking order: {str(e)}")

@app.post("/pick_object")
async def pick_object(
    x_robot: float = Form(...),
    y_robot: float = Form(...)
):
    """Pick up an object at the specified robot coordinates and drop it in the box"""
    try:
        pipeline.pick_up_and_drop(x_robot, y_robot)
        return {"status": "success", "message": f"Object picked at ({x_robot}, {y_robot}) and dropped at box"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pick and drop operation failed: {str(e)}")

@app.post("/run_pipeline")
async def run_pipeline(
    background_tasks: BackgroundTasks,
    image_path: Optional[str] = Form(None),
    detection_prompt: Optional[str] = Form(None),
    picking_prompt: Optional[str] = Form(None)
):
    """Run the full pipeline to detect, order, and pick objects"""
    # Generate task ID
    task_id = str(uuid.uuid4())
    task_status[task_id] = {"status": "started", "progress": 0}
    
    def run_task():
        try:
            task_status[task_id]["progress"] = 10
            
            # If no image provided, capture one
            if not image_path:
                cap = cv2.VideoCapture(pipeline.camera_index)
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    task_status[task_id] = {"status": "failed", "error": "Failed to capture image"}
                    return
                
                # Save the image
                image_path_task = os.path.join("static", f"captured_{task_id}.jpg")
                cv2.imwrite(image_path_task, frame)
            else:
                image_path_task = image_path
                
            task_status[task_id]["progress"] = 30
            task_status[task_id]["image_path"] = image_path_task
            
            # Run the pipeline
            success = pipeline.process_and_pick_objects(
                image_path=image_path_task,
                detection_prompt=detection_prompt,
                picking_prompt=picking_prompt
            )
            
            if success:
                task_status[task_id] = {
                    "status": "completed",
                    "progress": 100,
                    "image_path": image_path_task,
                    "message": "Pipeline completed successfully"
                }
            else:
                task_status[task_id] = {
                    "status": "failed",
                    "progress": 100,
                    "image_path": image_path_task,
                    "error": "Pipeline failed to complete"
                }
        except Exception as e:
            task_status[task_id] = {
                "status": "failed",
                "error": str(e)
            }
    
    # Run the task in the background
    background_tasks.add_task(run_task)
    
    return {
        "status": "task_started",
        "task_id": task_id,
        "message": "Pipeline execution started in background"
    }

@app.get("/task_status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    if task_id not in task_status:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task_status[task_id]

@app.post("/home_robot")
async def home_robot():
    """Move robot to home position"""
    try:
        pipeline.arm.move_gohome()
        return {"status": "success", "message": "Robot moved to home position"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to home robot: {str(e)}")

@app.get("/images/{filename}")
async def get_image(filename: str):
    """Retrieve a saved image"""
    filepath = os.path.join("static", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Image not found")
    
    return FileResponse(filepath)

if __name__ == "__main__":
    import cv2  # Import here to avoid issues when importing this module elsewhere
    uvicorn.run(app, host="0.0.0.0", port=8000)