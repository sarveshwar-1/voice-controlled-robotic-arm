import numpy as np
import cv2

# Clicked at: (63, 95)
# Clicked at: (174, 107)
# Clicked at: (115, 139)
# Clicked at: (234, 138)
# Clicked at: (92, 185)
# Clicked at: (178, 198)
# Clicked at: (163, 211)
# Clicked at: (117, 258)
# Clicked at: (240, 255)
# Clicked at: (103, 291)
# Clicked at: (178, 288)
# pixel_coordinates = [[202,63], [323,76], [256,112], [388,108], [232,161], [323,178], [308,190], [259,240], [388,239], [243,276], [323,274]]
# pixel_coordinates = [[196,92], [326,106], [255,146], [398,142], [229,199], [327,216], [310,230], [259,282], [395,281], [244,319], [327,317]]
pixel_coordinates = [[63, 95], [174, 107], [115, 139], [234, 138], [92, 185], [178, 198], [163, 211], [117, 258], [240, 255], [103, 291], [178, 288]]
robot_coordinates = [[393.6,-47.6], [245.8,-29.9], [325.3,11.4], [164.9,7.9], [356.9,73.4], [244.5,92.9], [262.6,107.8], [323.9,170.4], [164.6,168.5], [343.1,215.1], [243.9,212.2]]
pixel_points = np.array(pixel_coordinates, dtype=np.float32)
robot_points = np.array(robot_coordinates, dtype=np.float32)

homography_matrix, _ = cv2.findHomography(pixel_points, robot_points)
print("Homography Matrix:", homography_matrix)

def pixel_to_robot(x, y, matrix):
    pixel = np.array([x, y, 1]).reshape(3, 1)
    robot_coords = np.dot(matrix, pixel)
    robot_coords /= robot_coords[2]  
    return robot_coords[0][0], robot_coords[1][0]

import cv2
import numpy as np
from xarm.wrapper import XArmAPI


arm = XArmAPI('192.168.1.155')
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(0)
arm.connect()
arm.move_gohome()

box = pixel_to_robot(145, 360, homography_matrix)
box_x, box_y = box

import time
def pick_up_and_drop(x_robot, y_robot):
    arm.set_position(x_robot, y_robot, 70)  
    arm.set_position(x_robot, y_robot, 18, wait=True)  
    arm.set_suction_cup(False)
    time.sleep(1)
    # arm.set_position(x_robot, y_robot, 16.5, wait=True)  
    arm.set_position(x_robot, y_robot, 70, wait=True)  
    arm.set_position(box_x, box_y, 200)
    arm.set_position(box_x, box_y, 100)
    arm.set_suction_cup(True)
    arm.set_position(box_x, box_y, 200)

import cv2

cap = cv2.VideoCapture(1)

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

import dotenv
import os
from google import genai
from google.genai import types
from PIL import Image
import io
import os
import requests
from io import BytesIO

dotenv.load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY1")

model_name = "gemini-2.0-flash" 

bounding_box_system_instructions = """
    Return bounding boxes as a JSON array with labels. Never return masks or code fencing. Limit to 25 objects.
    If an object is present multiple times, name them according to their unique characteristic (colors, size, position, unique characteristics, etc..).
    """

client = genai.Client(api_key=GEMINI_API_KEY)

safety_settings = [
    types.SafetySetting(
        category="HARM_CATEGORY_DANGEROUS_CONTENT",
        threshold="BLOCK_ONLY_HIGH",
    ),
]

def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

def parse_json_groq(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```":
            json_output = "\n".join(lines[i+1:])  # Remove everything before "```json"
            json_output = json_output.split("```")[0]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output

import json
import random
import io
from PIL import Image, ImageDraw, ImageFont
from PIL import ImageColor

additional_colors = [colorname for (colorname, colorcode) in ImageColor.colormap.items()]

def plot_bounding_boxes(im, bounding_boxes):
    """
    Plots bounding boxes on an image with markers for each a name, using PIL, normalized coordinates, and different colors.

    Args:
        img_path: The path to the image file.
        bounding_boxes: A list of bounding boxes containing the name of the object
         and their positions in normalized [y1 x1 y2 x2] format.
    """

    # Load the image
    img = im
    width, height = img.size
    print(img.size)
    # Create a drawing object
    draw = ImageDraw.Draw(img)

    # Define a list of colors
    colors = [
    'red',
    'green',
    'blue',
    'yellow',
    'orange',
    'pink',
    'purple',
    'brown',
    'gray',
    'beige',
    'turquoise',
    'cyan',
    'magenta',
    'lime',
    'navy',
    'maroon',
    'teal',
    'olive',
    'coral',
    'lavender',
    'violet',
    'gold',
    'silver',
    ] + additional_colors

    # Parsing out the markdown fencing
    bounding_boxes = parse_json(bounding_boxes)

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

    # Display the image
    img.save("output_image.jpg")  # or "output_image.png"
    print("Image saved as output_image.jpg")

    img.show()

prompt = "Detect the 2d bounding boxes of the small cubes (with “label” as color of the small cube) outside the brown box"  # @param {type:"string"}

image = "captured_image.jpg"
# Load and resize image
im = Image.open(io.BytesIO(open(image, "rb").read()))
im.thumbnail([1024,1024], Image.Resampling.LANCZOS)

# Run model to find bounding boxes
response = client.models.generate_content(
    model=model_name,
    contents=[prompt, im],
    config = types.GenerateContentConfig(
        system_instruction=bounding_box_system_instructions,
        temperature=0.5,
    )
)

# Check output
print(response.text)

plot_bounding_boxes(im, response.text)

from groq import Groq

def groq(bounding_boxes):
    dotenv.load_dotenv()
    print(os.environ.get("GROQ_API_KEY"))

    # Your Groq API client setup
    client = Groq(
        api_key=os.environ.get("GROQ_API_KEY")  
    )
    centroids = []

    for i, bounding_box in enumerate(bounding_boxes):

        width, height = im.size
        # Convert normalized coordinates to absolute coordinates
        abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
        abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
        abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
        abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

        if abs_x1 > abs_x2:
            abs_x1, abs_x2 = abs_x2, abs_x1

        if abs_y1 > abs_y2:
            abs_y1, abs_y2 = abs_y2, abs_y
        
        c1 = (abs_x1 + abs_x2) / 2
        c2 = (abs_y1 + abs_y2) / 2
        print(c1, c2)

        centroids.append({
            "label": bounding_box["label"],
            "centroid": [c1, c2]})

    custom_prompt = "pick the blocks in the increasing order of their color intensity"

    response = client.chat.completions.create(
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
                    Do not include any explanation or thinking inthe output, only the raw JSON.
        
                """
            }
        ],
        model="llama-3.3-70b-versatile"
    )

    print("Raw response:", response)
    response_content = response.choices[0].message.content
    print(response_content)
    return response_content

print(response.text)
parse_json(response.text)

result = json.loads(parse_json_groq(response_content))
print(result)
centroids = result["centroids"]
print(centroids)


result = json.loads(parse_json_groq(response_content))

centroids_from_response = result["centroids"]
# print("Centroids from response:", type(centroids_from_respo|nse))

for centroid in centroids_from_response:
    c = centroid["centroid"]
    x, y = pixel_to_robot(c[0], c[1], homography_matrix)
    pick_up_and_drop(x, y)

bounding_boxes = parse_json(response.text)
for i, bounding_box in enumerate(json.loads(bounding_boxes)):

    width, height = im.size
    # Convert normalized coordinates to absolute coordinates
    abs_y1 = int(bounding_box["box_2d"][0]/1000 * height)
    abs_x1 = int(bounding_box["box_2d"][1]/1000 * width)
    abs_y2 = int(bounding_box["box_2d"][2]/1000 * height)
    abs_x2 = int(bounding_box["box_2d"][3]/1000 * width)

    if abs_x1 > abs_x2:
        abs_x1, abs_x2 = abs_x2, abs_x1

    if abs_y1 > abs_y2:
        abs_y1, abs_y2 = abs_y2, abs_y2
    
    c1 = (abs_x1 + abs_x2) / 2
    c2 = (abs_y1 + abs_y2) / 2

    c1_robot, c2_robot = pixel_to_robot(c1, c2, homography_matrix)
    pick_up_and_drop(c1_robot, c2_robot)
arm.move_gohome()
