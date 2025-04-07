import cv2
import numpy as np
import time
import json
import os
import dotenv
from PIL import Image, ImageDraw, ImageFont, ImageColor
from xarm.wrapper import XArmAPI
from google import genai
from google.genai import types
from groq import Groq
import io

class RobotVisionPipeline:
    def __init__(self, robot_ip='192.168.1.155', camera_index=1):
        # Environment setup
        dotenv.load_dotenv()
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        
        # Configuration
        self.robot_ip = robot_ip
        self.camera_index = camera_index
        self.homography_matrix = None
        self.box_position = None
        
        # Initialize components
        self.setup_vision_models()
    
    def setup_vision_models(self):
        """Initialize vision models and clients"""
        # Google Gemini setup
        self.genai_client = genai.Client(api_key=self.gemini_api_key)
        self.bounding_box_system_instructions = """
        Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
        If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
        """
        
        # Groq setup
        self.groq_client = Groq(api_key=self.groq_api_key)
    
    def calibrate(self, pixel_coordinates, robot_coordinates):
        """Calculate the homography matrix for coordinate transformation"""
        pixel_points = np.array(pixel_coordinates, dtype=np.float32)
        robot_points = np.array(robot_coordinates, dtype=np.float32)
        
        self.homography_matrix, _ = cv2.findHomography(pixel_points, robot_points)
        print("Homography Matrix:", self.homography_matrix)
        return self.homography_matrix
    
    def pixel_to_robot(self, x, y):
        """Convert pixel coordinates to robot coordinates using the homography matrix"""
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not calibrated. Run calibrate() first.")
            
        pixel = np.array([x, y, 1]).reshape(3, 1)
        robot_coords = np.dot(self.homography_matrix, pixel)
        robot_coords /= robot_coords[2]  
        return robot_coords[0][0], robot_coords[1][0]
    
    def connect_robot(self):
        """Connect to the robot arm"""
        self.arm = XArmAPI(self.robot_ip)
        self.arm.motion_enable(enable=True)
        self.arm.set_mode(0)
        self.arm.set_state(0)
        self.arm.connect()
        self.arm.move_gohome()
        print("Robot connected and homed")
        return self.arm
    
    def set_box_position(self, pixel_x, pixel_y):
        """Set the destination box position"""
        self.box_position = self.pixel_to_robot(pixel_x, pixel_y)
        print(f"Box position set to: {self.box_position}")
        return self.box_position
    
    def pick_up_and_drop(self, x_robot, y_robot):
        """Pick up an object and drop it in the box"""
        if self.box_position is None:
            raise ValueError("Box position not set. Run set_box_position() first.")
            
        # Move to object position
        self.arm.set_position(x_robot, y_robot, 70)  
        self.arm.set_position(x_robot, y_robot, 18, wait=True)  
        
        # Turn on suction
        self.arm.set_suction_cup(False)
        time.sleep(1)
        
        # Lift object
        self.arm.set_position(x_robot, y_robot, 70, wait=True)  
        
        # Move to box
        box_x, box_y = self.box_position
        self.arm.set_position(box_x, box_y, 200)
        self.arm.set_position(box_x, box_y, 100)
        
        # Release object
        self.arm.set_suction_cup(True)
        
        # Return to safe height
        self.arm.set_position(box_x, box_y, 200)
        print(f"Picked up object at ({x_robot}, {y_robot}) and dropped at box")
    
    def capture_image(self):
        """Capture an image from the camera"""
        cap = cv2.VideoCapture(self.camera_index)
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Can't receive frame. Exiting...")
                break
            
            cv2.imshow('Webcam Preview - Press Q to capture', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite('captured_image.jpg', frame)
                print("Image saved as captured_image.jpg")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        return 'captured_image.jpg'
    
    def detect_objects(self, image_path, prompt):
        """Detect objects in the image using Gemini vision model"""
        # Load and resize image
        im = Image.open(io.BytesIO(open(image_path, "rb").read()))
        im.thumbnail([1024, 1024], Image.Resampling.LANCZOS)
        
        # Run model to find bounding boxes
        response = self.genai_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, im],
            config=types.GenerateContentConfig(
                system_instruction=self.bounding_box_system_instructions,
                temperature=0.5,
            )
        )
        
        print(f"Detection results: {response.text}")
        return response.text, im
    
    def parse_json(self, json_output):
        """Parse JSON from model output with markdown fencing"""
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```json":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```json" is found
        return json_output
    
    def parse_json_groq(self, json_output):
        """Parse JSON from Groq model output with markdown fencing"""
        lines = json_output.splitlines()
        for i, line in enumerate(lines):
            if line == "```":
                json_output = "\n".join(lines[i+1:])  # Remove everything before "```"
                json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
                break  # Exit the loop once "```" is found
        return json_output
    
    def plot_bounding_boxes(self, im, bounding_boxes_text):
        """Plot bounding boxes on the image"""
        img = im
        width, height = img.size
        print(f"Image size: {img.size}")
        
        # Create a drawing object
        draw = ImageDraw.Draw(img)
        
        # Define a list of colors
        colors = [
            'red', 'green', 'blue', 'yellow', 'orange', 'pink', 'purple',
            'brown', 'gray', 'beige', 'turquoise', 'cyan', 'magenta',
            'lime', 'navy', 'maroon', 'teal', 'olive', 'coral', 'lavender',
            'violet', 'gold', 'silver'
        ] + [colorname for (colorname, colorcode) in ImageColor.colormap.items()]
        
        # Parsing out the markdown fencing
        bounding_boxes = self.parse_json(bounding_boxes_text)
        
        font = ImageFont.load_default()
        
        # Iterate over the bounding boxes
        for i, bounding_box in enumerate(json.loads(bounding_boxes)):
            # Select a color from the list
            color = colors[i % len(colors)]
            
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
            abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
            abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
            abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
            
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            
            # Draw the bounding box
            draw.rectangle(
                ((abs_x1, abs_y1), (abs_x2, abs_y2)), outline=color, width=4
            )
            
            # Draw the text
            if "label" in bounding_box:
                draw.text((abs_x1 + 8, abs_y1 + 6), bounding_box["label"], fill=color, font=font)
        
        # Save and display the image
        img.save("output_image.jpg")
        print("Image saved as output_image.jpg")
        img.show()
        
        return img
    
    def get_centroids_from_bounding_boxes(self, bounding_boxes_text, im):
        """Extract centroids from bounding boxes"""
        bounding_boxes = json.loads(self.parse_json(bounding_boxes_text))
        centroids = []
        
        width, height = im.size
        for i, bounding_box in enumerate(bounding_boxes):
            # Convert normalized coordinates to absolute coordinates
            abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
            abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
            abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
            abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)
            
            if abs_x1 > abs_x2:
                abs_x1, abs_x2 = abs_x2, abs_x1
            
            if abs_y1 > abs_y2:
                abs_y1, abs_y2 = abs_y2, abs_y1
            
            c1 = (abs_x1 + abs_x2) / 2
            c2 = (abs_y1 + abs_y2) / 2
            print(f"Centroid for object {i}: ({c1}, {c2})")
            
            centroids.append({
                "label": bounding_box["label"],
                "centroid": [c1, c2]
            })
        
        return centroids
    
    def get_objects_to_pick(self, centroids, custom_prompt):
        """Get objects to pick in specified order using LLM"""
        response = self.groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system", 
                    "content": "You are an assistant tasked with processing object data and returning only the centroids that match the user's request in JSON format."
                },
                {
                    "role": "user", 
                    "content": f"""
                        Here is a list of objects with their labels and centroids:
                        {json.dumps(centroids, indent=2)}

                        The command is: "{custom_prompt}"

                        Please return only the centroids of objects that match this command in this exact JSON format:
                        {{
                            "centroids": [
                                {{
                                    "centroid": [x, y]
                                }},
                                ...
                            ]
                        }}
                        Do not include any explanation or thinking in the output, only the raw JSON.
                    """
                }
            ],
            model="llama-3.3-70b-versatile"
        )
        
        response_content = response.choices[0].message.content
        print(f"LLM Response: {response_content}")
        return response_content
    
    def process_and_pick_objects(self, image_path=None, detection_prompt=None, picking_prompt=None):
        """Run the full pipeline to detect, order, and pick objects"""
        # Use captured image or take a new one
        if image_path is None:
            image_path = self.capture_image()
        
        # Default detection prompt
        if detection_prompt is None:
            detection_prompt = "Detect the 2d bounding boxes of the small cubes (with “label” as color of the small cube) outside the brown box"
        
        # Default picking order prompt
        if picking_prompt is None:
            picking_prompt = "pick the blocks in the increasing order of their color intensity"
        
        # Detect objects
        bounding_boxes_text, im = self.detect_objects(image_path, detection_prompt)
        
        # Plot bounding boxes
        self.plot_bounding_boxes(im, bounding_boxes_text)
        
        # Get centroids
        centroids = self.get_centroids_from_bounding_boxes(bounding_boxes_text, im)
        
        # Get picking order
        response_content = self.get_objects_to_pick(centroids, picking_prompt)
        parsed_response = self.parse_json_groq(response_content)
        
        # Process picking order
        try:
            result = json.loads(parsed_response)
            centroids_to_pick = result["centroids"]
            
            # Pick each object
            for centroid in centroids_to_pick:
                c = centroid["centroid"]
                x_robot, y_robot = self.pixel_to_robot(c[0], c[1])
                self.pick_up_and_drop(x_robot, y_robot)
            
            # Return home when done
            self.arm.move_gohome()
            return True
        except Exception as e:
            print(f"Error processing picking order: {e}")
            return False

def main():
    # Example calibration data
    pixel_coordinates = [
        [63, 95], [174, 107], [115, 139], [234, 138], [92, 185], 
        [178, 198], [163, 211], [117, 258], [240, 255], [103, 291], [178, 288]
    ]
    robot_coordinates = [
        [393.6, -47.6], [245.8, -29.9], [325.3, 11.4], [164.9, 7.9], [356.9, 73.4], 
        [244.5, 92.9], [262.6, 107.8], [323.9, 170.4], [164.6, 168.5], [343.1, 215.1], [243.9, 212.2]
    ]
    
    # Initialize pipeline
    pipeline = RobotVisionPipeline()
    
    # Calibrate
    pipeline.calibrate(pixel_coordinates, robot_coordinates)
    
    # Connect to robot
    pipeline.connect_robot()
    
    # Set box position
    pipeline.set_box_position(145, 360)
    
    # Run the full pipeline
    pipeline.process_and_pick_objects()

if __name__ == "__main__":
    main()