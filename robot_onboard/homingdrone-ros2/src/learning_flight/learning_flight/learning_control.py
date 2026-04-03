import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleLocalPosition, VehicleStatus, VehicleOdometry
from geometry_msgs.msg import PoseStamped

from sensor_msgs.msg import Range 
import numpy as np
import time
import requests
import os
import random

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('learning_control_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)

        # Create subscribers
        self.vehicle_local_position_subscriber = self.create_subscription(
            VehicleLocalPosition, '/fmu/out/vehicle_local_position', self.vehicle_local_position_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_callback, qos_profile)
        self.vehicle_odometry_subscriber = self.create_subscription(
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_visual_odometry_callback, qos_profile)
        self.motion_capture_subscriber = self.create_subscription(
            VehicleOdometry, '/dumb', self.motion_capture_callback, 10)
        
        self.lidar_front_sub = self.create_subscription(
            Range, '/tf_nova/dist_front', self.lidar_front_callback, 10)
        self.lidar_left_sub = self.create_subscription(
            Range, '/tf_nova/dist_left', self.lidar_left_callback, 10)
        self.lidar_right_sub = self.create_subscription(
            Range, '/tf_nova/dist_right', self.lidar_right_callback, 10)

        # Declare parameters
        self.declare_parameter('takeoff_height', -0.5)
        self.declare_parameter('takeoff_yaw', 0.0)
        self.declare_parameter('position_clamp', 0.2)
        self.declare_parameter('yaw_clamp', 0.2)
        self.declare_parameter('learning_pattern', 'spiral') # 'square' or 'spiral' or 'grid' or 'bee' or 'test'
        self.declare_parameter('grid_side_size', 7)
        self.declare_parameter('grid_diff', 0.5)
        self.declare_parameter('reach_point_margin', 0.08)    
        self.declare_parameter('reach_yaw_margin', 0.174)
        self.declare_parameter('facing', 'next') # 'home' or 'next' or '0' or '2' or 'omni'
        self.declare_parameter('omni_step', 10)
        self.declare_parameter('quadrant', 1)
        self.declare_parameter('learning_time', 0.0)
        self.declare_parameter('obstacle_avoidance', False) # True or False
        self.declare_parameter('init_heading', 0.0)
        
        self.declare_parameter('oa_stop_distance', 2.0) # Stop distance in meters
        self.declare_parameter('oa_strafe_distance', 1.0) # Strafe distance in meters

        self.update_parameters()
        self.add_on_set_parameters_callback(self.parameters_callback)

        # Initialize variables
        self.offboard_setpoint_counter = 0
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.vehicle_odometry = VehicleOdometry()
        self.motion_capture = VehicleOdometry()
        
        self.learning_path = [(0.0, 0.0)]
        self.path_yaw = [0.0]
        self.path_idx = 0
        self.learning_point = [0.0, 0.0, self.takeoff_height] #Initial poition
        self.learning_velocity = [float("NaN"), float("NaN"), float("NaN")]
        self.before_counter = 0
        self.after_counter = 1000
        self.learning_yaw = self.init_heading

        self.lidar_dist_front = 99.0  # Default to clear
        self.lidar_dist_left = 99.0   # Default to clear
        self.lidar_dist_right = 99.0  # Default to clear
        
        self.oa_target_yaw = 0.0
        self.oa_original_yaw = 0.0
        self.oa_target_position = [0.0, 0.0, 0.0]

        # State machine
        self.yaw_lock = True
        self.position_lock = True
        self.snapshot_lock = True
        self.state = 0 
        
        # 0: takeoff
        # 1: snapshot
        # 2: yaw
        # 3: move
        # 4: land (implicit)
        # 10: OA_DECIDE, 11: OA_YAW, 12: OA_MOVE, 13: OA_YAW_BACK, 14: Panic Snapshot, 15: Panic Yaw, 16: Panic Move

        # Create a timer to publish control commands
        self.timer = self.create_timer(0.1, self.timer_callback)

    def update_parameters(self):
        """Update parameters from the parameter server."""
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.takeoff_yaw = self.get_parameter('takeoff_yaw').value
        self.position_clamp = self.get_parameter('position_clamp').value
        self.yaw_clamp = self.get_parameter('yaw_clamp').value
        self.learning_pattern = self.get_parameter('learning_pattern').value
        self.reach_point_margin = self.get_parameter('reach_point_margin').value
        self.reach_yaw_margin = self.get_parameter('reach_yaw_margin').value
        self.facing = self.get_parameter('facing').value
        self.grid_side_size = self.get_parameter('grid_side_size').value
        self.grid_diff = self.get_parameter('grid_diff').value
        self.quadrant = self.get_parameter('quadrant').value
        self.learning_time = self.get_parameter('learning_time').value 
        self.omni_step = self.get_parameter('omni_step').value
        self.obstacle_avoidance = self.get_parameter('obstacle_avoidance').value
        self.init_heading = self.get_parameter('init_heading').value
        self.init_heading = np.deg2rad(self.init_heading)

        self.oa_stop_distance = self.get_parameter('oa_stop_distance').value
        self.oa_strafe_distance = self.get_parameter('oa_strafe_distance').value

    def parameters_callback(self, params):
        for param in params:
            self.get_logger().info(f'Parameter {param.name} has been updated to {param.value}')
        self.update_parameters()
        return SetParametersResult(successful=True)

    def vehicle_local_position_callback(self, vehicle_local_position):
        self.vehicle_local_position = vehicle_local_position

    def vehicle_status_callback(self, vehicle_status):
        self.vehicle_status = vehicle_status

    def vehicle_visual_odometry_callback(self, vehicle_odometry):
        self.vehicle_odometry = vehicle_odometry

    def motion_capture_callback(self, pose):
        self.motion_capture = pose
    
    def lidar_front_callback(self, msg):
        if msg.range > msg.min_range:
            self.lidar_dist_front = msg.range
        else:
            self.lidar_dist_front = 99.0 # Treat out-of-range/blind-zone as clear

    def lidar_left_callback(self, msg):
        if msg.range > msg.min_range:
            self.lidar_dist_left = msg.range
        else:
            self.lidar_dist_left = 99.0

    def lidar_right_callback(self, msg):
        if msg.range > msg.min_range:
            self.lidar_dist_right = msg.range
        else:
            self.lidar_dist_right = 99.0

    def arm(self):
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        """Publish the offboard control mode."""
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def normalize_angle(self, angle_diff: float) -> float:
        """Normalize the angle to be within -pi to pi."""
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi
    
    def clamp_position(self, x: float, y: float) -> float:
        """Clamp the position to not moving too fast."""
        current_angle = np.arctan2(y - self.vehicle_odometry.position[1], x - self.vehicle_odometry.position[0])
        distance = np.sqrt((x - self.vehicle_odometry.position[0])**2 + (y - self.vehicle_odometry.position[1])**2)
        if distance <= self.position_clamp:
            return x, y
        else:
            max_distance = self.position_clamp
            new_x = self.vehicle_odometry.position[0] + max_distance*np.cos(current_angle)
            new_y = self.vehicle_odometry.position[1] + max_distance*np.sin(current_angle)
            return new_x, new_y
    
    def clamp_yaw(self, yaw: float) -> float:
        """Clamp the yaw to not yawing too fast."""
        angle_diff = self.normalize_angle(yaw - self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3]))
        max_angle_diff = self.yaw_clamp
        if np.abs(angle_diff) <= max_angle_diff:
            new_yaw = yaw
        else:
            new_yaw = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3]) + max_angle_diff*np.sign(angle_diff)
        return self.normalize_angle(new_yaw)

    def publish_position_setpoint(self, x, y, z, yaw, vx, vy, vz) -> None:
        """Publish the trajectory setpoint."""
        msg = TrajectorySetpoint()

        # Clamp the position and yaw
        x, y = self.clamp_position(x, y)
        yaw = self.clamp_yaw(yaw)

        x = float(x)
        y = float(y)
        z = float(z)
        yaw = float(yaw)

        msg.yaw = yaw
        msg.position = [x, y, z]
        msg.velocity = [vx, vy, vz]

        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z], yaw, self.path_idx}")

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def next_learning_point(self):
        if self.path_idx < len(self.learning_path):
            self.learning_point = [self.learning_path[self.path_idx][0], self.learning_path[self.path_idx][1],\
                                    self.takeoff_height]
            
        else:
            self.get_logger().info('Reached the end of the learning path')
            self.land()
            exit(0)

    def generate_square_path(self, n):
        path = []
        side_length = 4.0
        limit = side_length / 2  # This sets the bounds to -3.0 and 3.0

        # start at 0,0
        x, y = 0, 0
        path.append((x, y))
        
        # Start at the bottom-left corner (-3, -3)
        x, y = -limit, -limit
        path.append((x, y))
        
        # Directions: Right, Up, Left, Down
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
        dir_idx = 0
        
        current_dist = 0
        
        while current_dist < n:
            dx, dy = directions[dir_idx]
            
            # Move by the full side length (4m)
            x += dx * side_length
            y += dy * side_length
            
            path.append((float(x), float(y)))
            
            current_dist += side_length
            dir_idx = (dir_idx + 1) % 4 

        # Return to home position
        path.append((0.0, 0.0))
            
        return path
    
    def generate_spiral_path(self, n, m, b):
        a = 0

        theta = np.linspace(0, 2*n*np.pi, m)

        r = a - b*theta
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        path = []
        for i in range(3,m):
            path.append((float(x[i]), float(y[i])))
        path.append((0.0, 0.0))  # Add the home position

        return path

    def generate_grid_path(self, side, diff):
        sign = [[1,-1], [-1,-1], [-1,1], [1,1]]
        quadrant = int(self.quadrant)
        path = []
        for i in range(side):
            for j in range(side):
                path.append((i*diff*sign[quadrant-1][0], j*diff*sign[quadrant-1][1]))
        return path
    
    def generate_bee_path(self, n, m, b):
        a = 0

        theta = np.linspace(0, 2*n*np.pi, m)

        r = a + b*theta
        x = r*np.cos(theta)
        y = r*np.sin(theta)

        path = []

        # add the landing learning
        path.extend([(0.2, 0.0), (0.15, 0.15), (0.0, 0.2), (-0.15, 0.15), (-0.2, 0.0), (-0.15, -0.15), (0.0, -0.2), (0.15, -0.15)])

        sign = 1
        for i in range(3,m):
            x_i, y_i = float(x[i]), float(y[i])
            
            # every time there is a change in the sign of x, change the sign of x and laters
            if np.sign(x_i) != np.sign(x[i-1])and i != 3 and y_i < 0:
                sign *= -1

            path.append((sign*x_i, y_i))


        path.append((0, 0))

        return path
    
    def get_current_heading(self, z, y, x, w) -> float:

        t1 = 2.0 * (w * z + x * y)
        t2 = 1.0 - 2.0 * (z * z + y * y)
        heading = -np.arctan2(t1, t2) + np.pi
        if heading > np.pi:
            heading -= 2.0 * np.pi
        elif heading < -np.pi:
            heading += 2.0 * np.pi

        return heading
    
    def get_current_pitch(self, z, y, x, w) -> float:

        sinp = 2 * (w * y - z * x)
        pitch = 0
        if abs(sinp) >= 1:
            if sinp >= 0:
                pitch = np.pi / 2
            else:
                pitch = np.pi / -2
        else:
            pitch = np.arcsin(sinp)

        return -1 * pitch
    
    def get_current_roll(self, z, y, x, w) -> float:
        t1 = 2 * (w * x + y * z)
        t2 = 1 - 2 * (x * x + y * y)

        return np.arctan2(t1, t2)
    
    def generate_yaw_path(self):
        yaw_path = []

        for i in range(0, len(self.learning_path)):
            x = [point[0] for point in self.learning_path]
            y = [point[1] for point in self.learning_path]
            if self.facing == 'home':
                # yaw to be towards 0,0, in radian
                if x[i]<0:
                    yaw_path.append(np.arctan(y[i]/(x[i]+1e-6)))
                else:
                    yaw_path.append(min(np.pi + np.arctan(y[i]/(x[i]+1e-6)), -np.pi + np.arctan(y[i]/(x[i]+1e-6)), key=abs))

            elif self.facing == 'next':
                # yaw to be towards the next point
                yaw_path.append(np.arctan2((y[i]-y[i-1]),(x[i]-x[i-1]+1e-6)))
            elif self.facing == '0':
                yaw_path.append(self.takeoff_yaw)

            elif self.facing == '2':
                yaw_path.append(np.arctan2((y[i]-y[i-1]+1e-6),(x[i]-x[i-1]+1e-6)))
                if i != 0: #no look back to home fot the first point to avoid 180 degree yaw which causes crazy drift
                    if x[i]<0:
                        yaw_path.append(np.arctan(y[i]/(x[i]+1e-6)))
                    else:
                        yaw_path.append(min(np.pi + np.arctan(y[i]/(x[i]+1e-6)), -np.pi + np.arctan(y[i]/(x[i]+1e-6)), key=abs))
            
            elif self.facing == 'omni':
                # or each point, yaw to every omni_step degrees
                omni_point = int(360/self.omni_step)
                for j in range(omni_point):
                    yaw_path.append(j*self.omni_step*np.pi/180)
                # for each point , duplicate the point omni_point times
                

        if self.facing == '2':
            self.learning_path = [elem for elem in self.learning_path for _ in (0, 1)][1:]

        if self.facing == 'omni':
            self.learning_path = [elem for elem in self.learning_path for _ in range(omni_point)]

        # Check the length of yaw_path is the same as the length of learning_path
        if len(yaw_path) != len(self.learning_path):
            self.get_logger().info(f'Length of yaw path: {len(yaw_path)} is not the same as the length of learning path: {len(self.learning_path)}')
            self.get_logger().info('Error in generating yaw path')
            self.land()
            exit(0)

        # Ensure the yaw is within -pi to pi
        for i in range(len(yaw_path)):
            yaw_path[i] = self.normalize_angle(yaw_path[i])
        
        return yaw_path
    
    def trigger_camera(self):
        path_idx, pos_x, pos_y, heading = self.path_idx, self.vehicle_odometry.position[0], self.vehicle_odometry.position[1], self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
        pos_x_mocap, pos_y_mocap, heading_mocap = self.motion_capture.position[0], self.motion_capture.position[1], self.get_current_heading(self.motion_capture.q[0], self.motion_capture.q[1], self.motion_capture.q[2], self.motion_capture.q[3])
        pitch = self.get_current_pitch(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
        roll = self.get_current_roll(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])

        try:
            requests.post('http://localhost:5000/trigger', data={'path_idx': path_idx, 'pos_x': pos_x, 'pos_y': pos_y, 'heading': heading, 'pitch': pitch, 'roll': roll, 'pos_x_mocap': pos_x_mocap, 'pos_y_mocap': pos_y_mocap, 'heading_mocap': heading_mocap})
        except requests.exceptions.RequestException as e:
            print(e)

    def calculate_next_point_yaw(self, current_position, next_position):
        x1, y1 = current_position
        x2, y2 = next_position
        return np.arctan2((y2-y1),(x2-x1+1e-6))
    
    def compare_yaw(self, yaw1, yaw2):
        yaw1 = yaw1 % (2*np.pi)
        yaw2 = yaw2 % (2*np.pi)
        
        diff =  np.abs(yaw1 - yaw2)
        diff = diff % (2*np.pi)
        
        if diff > np.pi:
            self.get_logger().info(f'Yaw difference: {diff}')
            return 2*np.pi - diff
        else:
            self.get_logger().info(f'Yaw difference: {diff}')
            return diff

    # Helper to trigger panic
    def start_panic_sequence(self):
        self.get_logger().info('OA: ALL BLOCKED. Panic Mode: Snapshot -> Random.')
        self.trigger_camera()
        self.panic_counter = 0
        self.state = 14 

    def timer_callback(self) -> None:
        """Callback function for the timer."""
        self.publish_offboard_control_heartbeat_signal()

        self.get_logger().info(f'Current state: {self.state}')
        self.get_logger().info(f'Learning index: {self.path_idx}')
        self.get_logger().info(f'Learning point: {self.learning_path[self.path_idx]}')
        current_position = (self.vehicle_odometry.position[0], self.vehicle_odometry.position[1])
        self.get_logger().info(f'Current position: {current_position}')

        # Generate the learning path
        if self.offboard_setpoint_counter == 1:
            self.get_logger().info('Generating learning path...')
            if self.learning_pattern == 'square':
                self.learning_path = self.generate_square_path(n=self.grid_side_size)
            elif self.learning_pattern == 'spiral':
                self.learning_path= self.generate_spiral_path(n=4, m=35, b=0.1)
            elif self.learning_pattern == 'bee':
                self.learning_path = self.generate_bee_path(n=4, m=36, b=0.1)
            elif self.learning_pattern == 'bee_outdoor3_op':
                self.learning_path = self.generate_bee_path(n=5, m=56, b=0.2)
                self.learning_path = self.learning_path[5:]
            elif self.learning_pattern == 'hover':
                self.learning_path = [(0.0, 0.0)]
            else:
                self.get_logger().info('Learning pattern not allowed')
                self.land()
                exit(0)

            for i in range(len(self.learning_path)):
                
                x = self.learning_path[i][0]*np.cos(self.init_heading) - self.learning_path[i][1]*np.sin(self.init_heading)
                y = -self.learning_path[i][0]*np.sin(self.init_heading) + self.learning_path[i][1]*np.cos(self.init_heading)
                self.learning_path[i] = (x, y)

            self.path_yaw = self.generate_yaw_path()
                
            self.get_logger().info(f'Learning path generated:{self.learning_path}')
            self.get_logger().info(f'Learning yaw generated:{self.path_yaw}')

            if self.obstacle_avoidance:
                self.get_logger().info('Obstacle avoidance enabled')
                
        # Taking off
        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.get_logger().info('Armed and offboard mode engaged')

        if np.abs(self.vehicle_odometry.position[2] - self.takeoff_height) <= 0.1 and self.state == 0:
            self.get_logger().info('Takeoff completed')
            self.state = 1
            self.before_counter = 0

        # Hover & Snapshot
        if self.state == 1:
            self.nudge_attempt_counter = 0
            if 1000 > self.before_counter > self.learning_time*10:
                self.get_logger().info(f'Has reached learning point for {self.learning_time}s, snapshot taking now')
                self.snapshot_lock = False
                self.before_counter = 1000

        if self.state == 1 and self.snapshot_lock == False:
            self.get_logger().info('Taking snapshot...')
            self.get_logger().info(f'mocap: {self.motion_capture}')
            self.trigger_camera()
            self.snapshot_lock = True  
            self.after_counter = 0

        if self.state == 1 and 1000 > self.after_counter > self.learning_time*10:
            self.get_logger().info(f'After snapshot taken for {self.learning_time}s, now yawing')
            self.state = 2
            self.yaw_lock = False
            self.after_counter = 1000

        # Yaw Control
        # Calculate yaw
        if self.state == 2 and self.yaw_lock == False:
            if self.facing == 'next':
                if self.path_idx < 8 and self.learning_pattern == 'bee':
                    self.learning_yaw = 0.0
                else:
                    self.learning_yaw = self.calculate_next_point_yaw((self.vehicle_odometry.position[0], self.vehicle_odometry.position[1]), self.learning_path[self.path_idx])
            elif self.facing == '0':
                self.learning_yaw = self.takeoff_yaw
            self.get_logger().info(f'Yawing to learning yaw: {self.learning_yaw}')
            self.yaw_lock = True
        
        # Execute yaw and check obstacles
        if self.state == 2:
            current_yaw = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
            
            # Check for panic retry
            if hasattr(self, 'panic_retry_flag') and self.panic_retry_flag:
                 if self.obstacle_avoidance and self.lidar_dist_front < self.oa_stop_distance:
                     self.get_logger().warn('OA: Panic maneuver failed. SKIPPING POINT.')
                     self.panic_retry_flag = False 
                     self.path_idx += 1
                     self.yaw_lock = False
                     return 

            if self.compare_yaw(current_yaw, self.learning_yaw) < self.reach_yaw_margin and np.abs(self.vehicle_odometry.angular_velocity[2]) < 0.1:
                
                if self.obstacle_avoidance and self.lidar_dist_front < self.oa_stop_distance:
                    # Obstacle detected logic
                    if not hasattr(self, 'nudge_attempt_counter'):
                        self.nudge_attempt_counter = 0

                    if self.nudge_attempt_counter >= 3:
                        self.get_logger().warn(f'OA: Nudged {self.nudge_attempt_counter} times and still blocked. Forcing PANIC.')
                        self.start_panic_sequence()
                    else:
                        self.get_logger().warn(f'Obstacle detected. Attempt {self.nudge_attempt_counter+1}/3. Entering OA state 10.')
                        self.state = 10 
                else:
                    # Path clear
                    self.nudge_attempt_counter = 0 # Success!
                    self.panic_retry_flag = False
                    self.state = 3
                    self.position_lock = False

        # Move to target
        if self.state == 3 and self.position_lock == False:
            self.learning_point = [self.learning_path[self.path_idx][0], self.learning_path[self.path_idx][1], self.takeoff_height]
            self.learning_velocity = [float("NaN"), float("NaN"), float("NaN")]
            self.get_logger().info(f'Moving to learning point: {self.learning_point}')
            self.position_lock = True
            
        if self.state == 3:
            if np.abs(self.vehicle_odometry.position[0]-self.learning_point[0]) < self.reach_point_margin and\
               np.abs(self.vehicle_odometry.position[1]-self.learning_point[1]) < self.reach_point_margin and \
               np.abs(self.vehicle_odometry.position[2]-self.learning_point[2]) < self.reach_point_margin:
                self.state = "3a"
                
        # Reached point -> Loop back to State 1
        if self.state == "3a" and np.sqrt((self.vehicle_odometry.velocity[0])**2 + (self.vehicle_odometry.velocity[1])**2) < 0.1:
            self.get_logger().info(f'Reached learning point: {self.learning_point}')
            self.path_idx += 1
            self.before_counter = 0
            self.state = 1 # Loops back to snapshot
            self.learning_velocity = [0.0, 0.0, 0.0]
            self.learning_point = [float("NaN"), float("NaN"), float("NaN")]

        # [State 10] OA_DECIDE
        if self.state == 10: 
            self.get_logger().info(f'OA: Deciding action. Left: {self.lidar_dist_left:.2f}, Right: {self.lidar_dist_right:.2f}')
            self.nudge_attempt_counter += 1

            if self.lidar_dist_left > self.lidar_dist_right and self.lidar_dist_left > self.oa_stop_distance:
                self.get_logger().info('OA: Trying left (18 deg).')
                self.oa_target_yaw = self.normalize_angle(self.learning_yaw - np.deg2rad(18.0))
                self.learning_yaw = self.oa_target_yaw 
                self.state = 11 
            elif self.lidar_dist_right > self.oa_stop_distance:
                self.get_logger().info('OA: Trying right (18 deg).')
                self.oa_target_yaw = self.normalize_angle(self.learning_yaw + np.deg2rad(18.0))
                self.learning_yaw = self.oa_target_yaw 
                self.state = 11
            else:
                self.start_panic_sequence()

        # [State 11] OA_NUDGE_YAW_CHECK
        if self.state == 11: 
            current_yaw = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
            self.learning_yaw = self.oa_target_yaw 
            self.learning_point = [self.vehicle_odometry.position[0], self.vehicle_odometry.position[1], self.takeoff_height]

            if self.compare_yaw(current_yaw, self.oa_target_yaw) < self.reach_yaw_margin and np.abs(self.vehicle_odometry.angular_velocity[2]) < 0.1:
                # Check side safety margin
                side_safety_margin = 0.5 
                
                is_front_clear = self.lidar_dist_front > self.oa_stop_distance
                is_sides_safe = (self.lidar_dist_left > side_safety_margin) and (self.lidar_dist_right > side_safety_margin)

                if is_front_clear and is_sides_safe:
                    self.get_logger().info(f'OA: Path and Sides clear. Moving sideways.')
                    current_pos = self.vehicle_odometry.position
                    self.oa_target_position = [
                        current_pos[0] + self.oa_strafe_distance * np.cos(self.oa_target_yaw),
                        current_pos[1] + self.oa_strafe_distance * np.sin(self.oa_target_yaw),
                        self.takeoff_height
                    ]
                    self.learning_point = self.oa_target_position
                    self.state = 12 
                else:
                    self.get_logger().warn(f'OA: Nudge path unsafe! Front:{is_front_clear}, Sides:{is_sides_safe}. PANIC.')
                    self.start_panic_sequence()

        # [State 12] OA_MOVE
        if self.state == 12:
            self.learning_point = self.oa_target_position
            self.learning_yaw = self.oa_target_yaw
            if np.abs(self.vehicle_odometry.position[0]-self.oa_target_position[0]) < self.reach_point_margin and \
               np.abs(self.vehicle_odometry.position[1]-self.oa_target_position[1]) < self.reach_point_margin:
                self.get_logger().info('OA: Sideways move done. Re-aligning.')
                self.state = 13 
        
        # [State 13] OA_REALIGNMENT
        if self.state == 13: 
            # Re-calculate yaw to original target from NEW position
            new_target_yaw = self.calculate_next_point_yaw(
                (self.vehicle_odometry.position[0], self.vehicle_odometry.position[1]), 
                self.learning_path[self.path_idx]
            )
            self.learning_yaw = new_target_yaw
            self.learning_point = [self.vehicle_odometry.position[0], self.vehicle_odometry.position[1], self.takeoff_height]

            current_yaw = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
            if self.compare_yaw(current_yaw, new_target_yaw) < self.reach_yaw_margin and np.abs(self.vehicle_odometry.angular_velocity[2]) < 0.1:
                self.get_logger().info('OA: Realigned. Back to State 2.')
                self.state = 2 
                self.yaw_lock = True 

        # [State 14] PANIC_SNAPSHOT_WAIT
        if self.state == 14:
            self.panic_counter += 1
            if self.panic_counter > 50: # Wait for snapshot
                self.get_logger().info('Snapshot taken. Randomizing Yaw.')
                direction = random.choice([-1, 1])
                self.oa_target_yaw = self.normalize_angle(self.learning_yaw + (np.deg2rad(36.0) * direction))
                self.learning_yaw = self.oa_target_yaw
                self.state = 15

        # [State 15] PANIC_YAW
        if self.state == 15:
            current_yaw = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
            self.learning_point = [self.vehicle_odometry.position[0], self.vehicle_odometry.position[1], self.takeoff_height]
            
            if self.compare_yaw(current_yaw, self.oa_target_yaw) < self.reach_yaw_margin and np.abs(self.vehicle_odometry.angular_velocity[2]) < 0.1:
                if self.lidar_dist_front > self.oa_stop_distance:
                    self.get_logger().info('OA Panic: Path clear. Moving 0.5m.')
                    current_pos = self.vehicle_odometry.position
                    self.oa_target_position = [
                        current_pos[0] + 0.5 * np.cos(self.oa_target_yaw),
                        current_pos[1] + 0.5 * np.sin(self.oa_target_yaw),
                        self.takeoff_height
                    ]
                    self.learning_point = self.oa_target_position
                    self.state = 16 
                else:
                    self.get_logger().warn('OA Panic: Random direction blocked! Skipping.')
                    self.path_idx += 1
                    self.state = 2
                    self.yaw_lock = False
                    self.panic_retry_flag = False

        # [State 16] PANIC_MOVE
        if self.state == 16:
            self.learning_point = self.oa_target_position
            if np.abs(self.vehicle_odometry.position[0]-self.oa_target_position[0]) < self.reach_point_margin and \
               np.abs(self.vehicle_odometry.position[1]-self.oa_target_position[1]) < self.reach_point_margin:
                
                self.get_logger().info('OA Panic: Move done. Re-aligning to target.')
                # Calc yaw to target
                new_target_yaw = self.calculate_next_point_yaw(
                    (self.vehicle_odometry.position[0], self.vehicle_odometry.position[1]), 
                    self.learning_path[self.path_idx]
                )
                self.learning_yaw = new_target_yaw
                self.panic_retry_flag = True # Mark that we just panic-moved
                self.state = 2
                self.yaw_lock = True
        
        if self.learning_pattern == 'hover':
            self.learning_point = [0.0, 0.0, self.takeoff_height]
            self.learning_yaw = self.takeoff_yaw
            self.learning_velocity = [0.0, 0.0, 0.0]
            self.path_idx = 0

        self.publish_position_setpoint(self.learning_point[0], self.learning_point[1], self.learning_point[2], self.learning_yaw, self.learning_velocity[0], self.learning_velocity[1], self.learning_velocity[2])

        self.offboard_setpoint_counter += 1
        self.before_counter += 1
        self.after_counter += 1

        if self.path_idx == len(self.learning_path) and self.state != 0 and self.learning_pattern != 'hover':
            self.get_logger().info('Learning completed')
            self.land()
            exit(0)
            

def main(args=None) -> None:
    print('Starting offboard control node...')
    rclpy.init(args=args)
    offboard_control = OffboardControl()
    rclpy.spin(offboard_control)
    offboard_control.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)