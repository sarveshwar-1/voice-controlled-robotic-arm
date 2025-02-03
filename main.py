import pybullet as p
import pybullet_data
import time
import numpy as np
import logging
import random
import cv2
import threading
from object_detection import detect_objects

logging.basicConfig(level=logging.INFO)

# Add joint limits for Franka Panda
JOINT_LIMITS = [
    (-2.8973, 2.8973),  # joint 1
    (-1.7628, 1.7628),  # joint 2
    (-2.8973, 2.8973),  # joint 3
    (-3.0718, -0.0698), # joint 4
    (-2.8973, 2.8973),  # joint 5
    (-0.0175, 3.7525),  # joint 6
    (-2.8973, 2.8973)   # joint 7
]

def get_random_position():
    """Generate truly random position within robot's reach"""
    angle = random.uniform(0, 2 * np.pi)
    radius = random.uniform(0.3, 0.7)
    return [
        radius * np.cos(angle),  # x
        radius * np.sin(angle),  # y
        0.05                     # z
    ]

def capture_image(camera_position=[0.5, -1.0, 1.0], target_position=[0.5, 0, 0]):
    """Capture image from simulation with configurable camera"""
    width, height = 640, 480
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=camera_position,
        cameraTargetPosition=target_position,
        cameraUpVector=[0, 0, 1]
    )
    projMatrix = p.computeProjectionMatrixFOV(60, width/height, 0.1, 100.0)
    
    images = p.getCameraImage(
        width, height, viewMatrix, projMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    
    return np.reshape(images[2], (height, width, 4))[:,:,:3]

def initialize_simulation():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    
    # Load environment
    plane = p.loadURDF("plane.urdf")
    robot = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    
    # Place apple at random position
    random_pos = get_random_position()
    apple = p.createMultiBody(
        baseMass=0.1,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_SPHERE, radius=0.05),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.05, rgbaColor=[1, 0, 0, 1]),
        basePosition=random_pos
    )
    
    # Initialize robot joints to a good starting position
    initial_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
    for i, pos in enumerate(initial_positions):
        p.resetJointState(robot, i, pos)
    
    # Let simulation settle
    for _ in range(200):
        p.stepSimulation()
        time.sleep(1/240)
    
    return robot, apple

def detect_apple_position():
    """Detect apple position using YOLO"""
    image = capture_image(
        camera_position=[0.5, 0, 1.5],  # Top-down view for better detection
        target_position=[0.5, 0, 0]
    )
    detections = detect_objects(image)
    
    if detections and detections.get('apple'):
        bbox = detections['apple']
        # Convert bbox center to world coordinates
        center_x = (bbox[0] + bbox[2]) / 2 
        center_y = (bbox[1] + bbox[3]) / 2
        
        # Normalize and map to world coordinates
        norm_x = center_x / 640.0
        norm_y = center_y / 480.0
        
        # Map to world coordinates with adjusted ranges
        world_x = -0.3 + norm_x * 1.2  # Map [0,1] to [-0.3,0.9]
        world_y = 0.4 - norm_y * 0.8   # Map [0,1] to [0.4,-0.4]
        
        logging.info(f"Detection mapped from ({center_x}, {center_y}) to ({world_x}, {world_y})")
        return [world_x, world_y, 0.05]
    
    return None

def visualization_thread(stop_event):
    """Thread for real-time visualization"""
    cv2.namedWindow('Top View', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Top View', 800, 600)
    
    while not stop_event.is_set():
        top_view = capture_image(
            camera_position=[0.5, 0, 1.5],
            target_position=[0.5, 0, 0]
        )
        cv2.imshow('Top View', top_view)
        cv2.waitKey(1)

def execute_pick_place(robot, apple_id):
    def check_joint_limits(positions):
        """Verify joint positions are within limits"""
        return all(JOINT_LIMITS[i][0] <= pos <= JOINT_LIMITS[i][1] 
                  for i, pos in enumerate(positions[:7]))
    
    def generate_trajectory(start_pos, end_pos, steps):
        """Generate smooth trajectory between positions"""
        trajectory = []
        for t in range(steps):
            fraction = t / float(steps - 1)
            # Use cubic interpolation for smoother acceleration/deceleration
            smooth_frac = (3 * fraction ** 2 - 2 * fraction ** 3)
            pos = [
                start_pos[i] + (end_pos[i] - start_pos[i]) * smooth_frac 
                for i in range(3)
            ]
            trajectory.append(pos)
        return trajectory

    def move_arm(target_pos, steps=100):  # Even faster movement
        # Get current joint positions for all joints (including fixed joints)
        current_joints = []
        for i in range(p.getNumJoints(robot)):
            joint_info = p.getJointInfo(robot, i)
            if joint_info[2] != p.JOINT_FIXED:
                current_joints.append(p.getJointState(robot, i)[0])
        
        # Calculate IK with current positions
        joint_positions = p.calculateInverseKinematics(
            robot, 
            11,  # end effector link index
            target_pos,
            maxNumIterations=100,
            residualThreshold=.01,
            currentPositions=current_joints
        )
        
        # Move to target position with increased speed
        for step in range(steps):
            fraction = step / float(steps)
            for i in range(len(current_joints)):
                current_pos = p.getJointState(robot, i)[0]
                target_pos = joint_positions[i]
                intermediate_pos = current_pos + (target_pos - current_pos) * fraction
                
                p.setJointMotorControl2(
                    robot, i,
                    p.POSITION_CONTROL,
                    targetPosition=intermediate_pos,
                    force=300,    # More force
                    maxVelocity=3.0  # Faster velocity
                )
            
            p.stepSimulation()
            time.sleep(1/480)  # Reduced sleep time
        
        return True

    try:
        while True:
            # Get current apple position from world coordinates
            apple_pos = p.getBasePositionAndOrientation(apple_id)[0]
            logging.info(f"Actual apple position: {apple_pos}")
            
            # Detect apple using vision
            detected_pos = detect_apple_position()
            logging.info(f"Detected apple position: {detected_pos}")
            
            if not detected_pos:
                logging.error("Could not detect apple")
                return False
            
            # Use detected position for movement
            positions = [
                ([detected_pos[0], detected_pos[1], 0.4], "pre-grasp"),
                (detected_pos, "grasp"),
                ([detected_pos[0], detected_pos[1], 0.4], "lift"),
                ([0.8, 0, 0.4], "transport"),
                ([0.8, 0, 0.05], "place")
            ]
            
            constraint = None
            for pos, name in positions:
                logging.info(f"Moving to {name} position")
                if not move_arm(pos):
                    if constraint:
                        p.removeConstraint(constraint)
                    return False
                
                # Create/remove constraint at appropriate steps
                if name == "grasp":
                    constraint = p.createConstraint(
                        robot, 11, apple_id, -1,
                        p.JOINT_FIXED, [0, 0, 0],
                        [0, 0, 0.05], [0, 0, 0]
                    )
                elif name == "place":
                    p.removeConstraint(constraint)
                    
                # Add small delay between movements
                time.sleep(0.5)
            
            # Move to truly random position
            new_pos = get_random_position()
            logging.info(f"Moving apple to new position: {new_pos}")
            p.resetBasePositionAndOrientation(apple_id, new_pos, [0, 0, 0, 1])
            
            # Let physics settle
            for _ in range(50):
                p.stepSimulation()
                time.sleep(1/240)
            
    except Exception as e:
        logging.error(f"Error during pick and place: {e}")
        if constraint:
            try:
                p.removeConstraint(constraint)
            except:
                pass
        return False

def main():
    try:
        robot, apple = initialize_simulation()
        
        # Start visualization thread
        stop_event = threading.Event()
        vis_thread = threading.Thread(target=visualization_thread, args=(stop_event,))
        vis_thread.start()
        
        while True:  # Continuous operation
            success = execute_pick_place(robot, apple)
            if not success:
                logging.error("Pick and place operation failed")
            time.sleep(1)  # Brief pause between attempts
            
    except KeyboardInterrupt:
        logging.info("Simulation stopped by user")
        stop_event.set()
        vis_thread.join()
    except Exception as e:
        logging.error(f"Simulation error: {e}")
        stop_event.set()
        vis_thread.join()
    finally:
        cv2.destroyAllWindows()
        p.disconnect()

if __name__ == "__main__":
    main()