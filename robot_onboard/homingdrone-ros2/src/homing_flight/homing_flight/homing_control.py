import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult

from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from px4_msgs.msg import OffboardControlMode, TrajectorySetpoint, VehicleCommand, \
    VehicleLocalPosition, VehicleStatus, VehicleOdometry, VehicleAttitudeSetpoint
import numpy as np

from sensor_msgs.msg import Range

import time
import os

import requests

import threading
from flask import Flask, request, jsonify

import random

app = Flask(__name__)

class OffboardControl(Node):
    """Node for controlling a vehicle in offboard mode."""

    def __init__(self) -> None:
        super().__init__('homing_control_node')

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
        self.attitude_setpoint_publisher = self.create_publisher(
            VehicleAttitudeSetpoint, '/fmu/in/vehicle_attitude_setpoint', qos_profile)

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
        self.declare_parameter('takeoff_height', -1.0)
        self.declare_parameter('position_clamp', 0.5)
        self.declare_parameter('yaw_clamp', 0.8)
        self.declare_parameter('reach_point_margin', 0.1)
        self.declare_parameter('reach_yaw_margin', 0.174) # 10 degrees
        self.declare_parameter('moving_step', 0.3)      
        self.declare_parameter('initial_mode', 'homing')  # 'homing' or 'outbound' or 'inbound'
        self.declare_parameter('outbound_shape', 'insect') # 'search' or 'straight line' or 'insect'
        self.declare_parameter('outbound_times', 1) 
        self.declare_parameter('env', 'cyberzoo') # 'cyberzoo' or 'indoor' or 'outdoor'
        self.declare_parameter('obstacle_avoidance', False) 
        self.declare_parameter('init_heading', 0.0) # offset to current heading is 0
        self.declare_parameter('smallest_step', 0.1)
        
        # OA parameters
        self.declare_parameter('oa_stop_distance', 2.0) # Stop distance for outbound/inbound
        self.declare_parameter('oa_homing_stop_distance', 0.5) # Stop distance for homing
        self.declare_parameter('oa_evade_distance', 1.5) # Strafe distance
        self.declare_parameter('oa_homing_evade_distance', 1.0) # Homing strafe distance

        self.update_parameters()
        self.add_on_set_parameters_callback(self.parameters_callback)
 
        self.offboard_setpoint_counter = 0
        self.reach_0_counter = 1000
        self.vehicle_local_position = VehicleLocalPosition()
        self.vehicle_status = VehicleStatus()
        self.vehicle_odometry = VehicleOdometry()
        self.motion_capture = VehicleOdometry()

        self.pathidx = 0
        self.output = 0.0
        self.distance = 0.0
        self.snapshot_heading = 0.0
        self.snapshot_x = 0.0
        self.snapshot_y = 0.0
        self.output_received = False
        self.next_heading = 0.0
        self.next_waypoint_x = 0.0
        self.next_waypoint_y = 0.0
        self.desired_velocity = [float("NaN"), float("NaN"), float("NaN")]

        self.outbound_path = []
        self.outbound_path_yaw = []
        self.outbound_path_idx = 0

        self.mode = self.initial_mode

        # state machine
        self.yaw_lock = True
        self.position_lock = True
        self.snapshot_lock = True
        
        # 0: takeoff
        # 1: snapshot (homing)
        # 2: yaw
        # 3: move
        # 4: land
        # 10: OA_DECIDE, 11: OA_YAW_LEFT, 12: OA_YAW_RIGHT, 13: OA_MOVE, 14: OA_YAW_BACK, 15: random rotate, 16: random check
        self.state = 0 
        
        self.oa_target_yaw = 0.0
        self.oa_original_heading = 0.0
        self.oa_evade_position = [0.0, 0.0, 0.0]
        self.current_evade_distance = 0.0  # Will be updated by mode
        self.current_stop_distance = 0.0   # Will be updated by mode

        self.flask_thread = threading.Thread(target=self.run_flask_app)
        self.flask_thread.start()

        self.lidar_dist_front = 99.0
        self.lidar_dist_left = 99.0
        self.lidar_dist_right = 99.0

        self.timer = self.create_timer(0.1, self.timer_callback)

    def update_parameters(self):
        """Update parameters from the parameter server."""
        self.takeoff_height = self.get_parameter('takeoff_height').value
        self.position_clamp = self.get_parameter('position_clamp').value
        self.preset_position_clamp = self.get_parameter('position_clamp').value
        self.yaw_clamp = self.get_parameter('yaw_clamp').value
        self.reach_point_margin = self.get_parameter('reach_point_margin').value
        self.reach_yaw_margin = self.get_parameter('reach_yaw_margin').value
        self.moving_step = self.get_parameter('moving_step').value
        self.initial_mode = self.get_parameter('initial_mode').value
        self.outbound_shape = self.get_parameter('outbound_shape').value
        self.outbound_times = self.get_parameter('outbound_times').value
        self.env = self.get_parameter('env').value
        self.obstacle_avoidance = self.get_parameter('obstacle_avoidance').value
        self.init_heading = self.get_parameter('init_heading').value
        self.smallest_step = self.get_parameter('smallest_step').value
        
        self.oa_stop_distance = self.get_parameter('oa_stop_distance').value
        self.oa_homing_stop_distance = self.get_parameter('oa_homing_stop_distance').value
        self.oa_evade_distance = self.get_parameter('oa_evade_distance').value
        self.oa_homing_evade_distance = self.get_parameter('oa_homing_evade_distance').value
        
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

    def motion_capture_callback(self, motion_capture):
        self.motion_capture = motion_capture

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
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self):
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self):
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal(self):
        msg = OffboardControlMode()
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_control_mode_publisher.publish(msg)

    def normalize_angle(self, angle_diff: float) -> float:
        return (angle_diff + np.pi) % (2 * np.pi) - np.pi

    def clamp_position(self, x: float, y: float) -> float:
        """Clamp the position to avoid moving too fast."""
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
        angle_diff = self.normalize_angle(yaw - self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3]))
        max_angle_diff = self.yaw_clamp
        if np.abs(angle_diff) <= max_angle_diff:
            new_yaw = yaw
        else:
            new_yaw = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3]) + max_angle_diff*np.sign(angle_diff)
        return self.normalize_angle(new_yaw)
    
    def publish_position_setpoint(self, x, y, z, yaw, vx, vy, vz) -> None:
        msg = TrajectorySetpoint()

        # Clamp the position and yaw
        x, y = self.clamp_position(x, y)
        yaw = self.clamp_yaw(yaw)

        # make sure the position and yaw are float
        x = float(x)
        y = float(y)
        z = float(z)
        yaw = float(yaw)

        msg.yaw = yaw
        msg.position = [x, y, z]
        msg.velocity = [vx, vy, vz]
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)

    def generate_setpoint(self, current_position) -> TrajectorySetpoint:
        pass

    def publish_attitude_setpoint(self, yaw: float):
        msg = VehicleAttitudeSetpoint()
        msg.roll_body = 0.0
        msg.pitch_body = 0.0
        msg.yaw_body = yaw
        msg.yaw_sp_move_rate = 5.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.attitude_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing attitude setpoints {[yaw]}")

    def publish_vehicle_command(self, command, **params) -> None:
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

    def trigger_camera(self):
        path_idx, pos_x, pos_y = self.pathidx, self.vehicle_odometry.position[0], self.vehicle_odometry.position[1]
        heading = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
        pos_x_mocap, pos_y_mocap = self.motion_capture.position[0], self.motion_capture.position[1]
        heading_mocap = self.get_current_heading(self.motion_capture.q[0], self.motion_capture.q[1], self.motion_capture.q[2], self.motion_capture.q[3])
        pitch = self.get_current_pitch(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
        roll = self.get_current_roll(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])

        try:
            requests.post('http://localhost:5000/trigger', data={'path_idx': path_idx, 'pos_x': pos_x, 'pos_y': pos_y, 'heading': heading, 'pitch': pitch, 'roll': roll,'pos_x_mocap': pos_x_mocap, 'pos_y_mocap': pos_y_mocap, 'heading_mocap': heading_mocap})
        except requests.exceptions.RequestException as e:
            print(e)

    def run_flask_app(self):
        @app.route('/predict', methods=['POST'])
        def handle_predict():
            data = request.form 
            self.output = float(data.get('output', 0)) 
            self.distance = float(data.get('distance', 0)) 
            self.output_received = True 
            return jsonify({"message": "Data received"}), 200

        app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)

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

    def generate_outbound_path_yaw(self):
        yaw_path = []
        for i in range(len(self.outbound_path)):
            x = [point[0] for point in self.outbound_path]
            y = [point[1] for point in self.outbound_path]
            yaw_path.append(self.calculate_next_point_yaw([x[i-1], y[i-1]], [x[i], y[i]]))
        return yaw_path

    def generate_outbound_path(self):
        if self.outbound_shape == 'straight_line' and self.env  == 'indoor':
            self.outbound_path = [[0.0, 0.0],[15.0, 0.0],[15.0, -20.0]]
        elif self.outbound_shape == 'insect' and self.env  == 'indoor':
            self.outbound_path = [[0.0, 0.0], [5.0, -5.0], [10.0, 0.0], [15.0, 0.0], [10.0, -5.0], [15.0, -5.0], [2.5, -20.0], [5.0, -20.0], [10.0, -15.0], [5.0, -15.0], [5.0, -20.0], [15.0, -10.0], [15.0, -20.0]]
        elif self.outbound_shape == 'search' and self.env  == 'indoor':
            self.outbound_path = [[0.0, 0.0], [0.0, -10.0], [20.0, -10.0], [20.0, -15.0], [0.0, -15.0], [0.0, -20.0], [20.0, -20.0], [20.0, -25.0], [0.0, -25.0], [0.0, -30.0], [20.0, -30.0]]
        elif self.outbound_shape == 'search2' and self.env  == 'indoor':
            self.outbound_path= [[0.0, 0.0], [-5.0, -3.0], [-10.0, -5.0], [0.0, -5.0], [0.0, -10.0], [-10.0, -10.0], [-10.0, -15.0], [-0.0, -15.0], [-0.0, -20.0], [-10.0, -20.0], [-10.0, -25.0]]
        elif self.outbound_shape == 'star' and self.env  == 'indoor':
            self.outbound_path= [[0.0, 0.0], [0.0, -10.0], [20.0, -25.0], [0.0, -25.0], [20.0, -10.0], [10.0, -30.0], [0.0, -10.0], [0.0, -0.0]]
        elif self.outbound_shape == 'center' and self.env == 'indoor':
            self.outbound_path = [[0.0, 0.0], [-5.0, -10.0], [-5.0, -15.0], [-10.0, -15.0], [-10.0, -10.0], [5.0, -10.0], [10.0, -10.0], [10.0, -15.0], [5.0, -15.0], [5.0, 10.0], [10.0, 10.0], [10.0, 15.0], [5.0, 15.0], [-5.0, 15.0], [-10.0, 15.0], [-10.0, 10.0], [-5.0, 10.0]]
        elif self.outbound_shape == 'straight_line' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [5.0, -5.0], [-5.0, -50.0]]
        elif self.outbound_shape == 'straight_line2' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [5.0, -5.0], [-20.0, -50.0], [3.0, -100.0]]
        elif self.outbound_shape == 'straight_line2_mirror' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [-5.0, -5.0], [20.0, -50.0], [-3.0, -100.0]]
        elif self.outbound_shape == 'straight_line3' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [-30.0, 40.0], [3.0, -100.0]]
        elif self.outbound_shape == 'insect' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [5.0, -5.0], [8.0, -9.0], [-5.0, -7.0], [-3.0, -11.0], [0.0, -13.0], [3.0, -17.0], [6.0, -19.0], [-4.0, -25.0], [-2.0, -31.0], [2.0, -26.0], [5.0, -35.0], [3.0, -28.0], [-4.0, -37.0], [0.0, -42.0], [5.0, -44.0], [-2.0, -50.0]]
        elif self.outbound_shape == 'insect2' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [5.0, -10.0], [-43.0, -65.0], [-30.0, -100.0]]
        elif self.outbound_shape == 'insect3' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [5.0, -10.0], [0.0, -18.0], [-12.0, -25.0], [-24.0, -30.0], [-34.0, -40.0], [-40.0, -53.0], [-42.0, -60.0], [-43.0, -65.0], [-41.0, -75.0], [-40.0, -80.0], [-36.0, -90.0], [-30.0, -100.0]]
        elif self.outbound_shape == '50m_1' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [15.0, -8.0], [14.0, -15.0], [0.0, -17.0], [-20.0, -27.0], [-16.0, -35.0], [-2.0, -25.0], [9.0, -35.0], [23.0, -40.0], [27.0, -50.0], [3.0, -45.0], [-12.0, -55.0]]
        elif self.outbound_shape == '50m_2' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [15.0, -7.0], [25.0, -17.0], [5.0, -17.0], [-7.0, -11.0], [-17.0, -15.0], [-7.0, -30.0], [2.0, -25.0], [8.0, -35.0], [23.0, -35.0], [27.0, -50.0], [-22.0, -35.0], [-13.0, -55.0], [12.0, -56.0]]
        elif self.outbound_shape == '50m_3' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [-7.0, -11.0], [-17.0, -15.0], [-7.0, -30.0], [2.0, -25.0], [25.0, -20.0], [23.0, -35.0], [27.0, -50.0], [-22.0, -35.0], [-13.0, -55.0], [12.0, -56.0]]
        elif self.outbound_shape == '50m_4' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [70.0, -40.0], [-10.0, -80.0]]
        elif self.outbound_shape == '50m_5' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [-20.0, -20.0],[-15.0, -50.0], [25.0, -90.0], [30.0, -120.0], [5.0, -150.0]]
        elif self.outbound_shape == '100m_1' and self.env == 'outdoor':
            self.outbound_path = [[0.0, -4.0], [50.0, -34.0], [-34.0, -30.0], [-14.0, -60.0],[46.0, -70.0], [54.0, -100.0], [-34.0, -100.0], [18.0, -120.0]]
        elif self.outbound_shape == '100m_2' and self.env == 'outdoor':
            self.outbound_path = [[0.0, -4.0], [100.0, -15.0], [100.0, -20.0], [-34.0, -30.0], [-30.0, -50.0], [-14.0, -60.0],[120.0, -70.0], [130.0, -80.0], [120.0, -95.0], [-34.0, -100.0], [-25.0, -130.0],[25.0, -110.0], [75.0, -150.0], [25.0, -160.0], [-10.0, -140.0]]
        elif self.outbound_shape == 'cc_1' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [-6.0, 2.0],[2.0, -2.0], [-5.0, -6.0], [-4.0, -3.0], [2.0, -8.0], [-5.0, -9.0], [0.0, -10.0], [-4.0, -12.0], [3.0, -16.0]]
        elif self.outbound_shape == 'cc_2' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [2.0, -2.0], [-5.0, -6.0], [-4.0, -3.0], [-2.0, -2.0], [2.0, -8.0], [0.0, -6.0], [-5.0, -9.0], [0.0, -10.0], [3.0, -12.0], [-5.0, -16.0]]
        elif self.outbound_shape == 'cc_3' and self.env == 'outdoor':
            self.outbound_path = [[0.0, 0.0], [2.0, 0.0,], [2.0, -4.0], [-5.0, -4.0], [-5.0, -8.0], [2.0, -8.0], [2.0, -12.0], [-5.0, -12.0], [-5.0, -16.0], [2.0, -16.0]]
        elif self.outbound_shape == 'ddi_6' and self.env == 'indoor':
            self.outbound_path = [[0.0, 0.0], [3.5, -2.5], [-2.0, -5.0], [1.0, -8.0], [-3.8, -11.5], [2.5, -14.8]]
        elif self.outbound_shape == 'ddi_7' and self.env == 'indoor':
            self.outbound_path = [[0.0, 0.0], [-3.9, -3.0], [4.0, -6.5], [-1.5, -9.0], [1.0, -11.0], [-3.0, -15.0]]
        elif self.outbound_shape == 'ddi_8' and self.env == 'indoor':
            self.outbound_path = [[0.0, 0.0], [2.1, -1.8], [-1.1, -3.5], [3.0, -5.5], [0.5, -7.8], [-3.2, -9.9], [1.8, -12.5], [-0.5, -14.5]]
        elif self.outbound_shape == 'ddi_9' and self.env == 'indoor':
            self.outbound_path = [[0.0, 0.0], [3.8, -4.0], [1.0, -3.0], [-2.5, -7.0], [2.0, -10.0], [-4.0, -12.0], [0.0, -15.0]]
        elif self.outbound_shape == 'ddi_10' and self.env == 'indoor':
            self.outbound_path = [[0.0, 0.0], [-3.5, -4.5], [-1.0, -6.0], [-4.0, -9.0], [3.0, -12.0], [-2.0, -14.9]]
        else:
            return None
        
        self.outbound_path = self.outbound_times * self.outbound_path
        self.init_heading = np.radians(self.init_heading)
        for i in range(len(self.outbound_path)):
            self.outbound_path[i][0] = self.outbound_path[i][0] * np.cos(self.init_heading) - self.outbound_path[i][1] * np.sin(self.init_heading)
            self.outbound_path[i][1] = -self.outbound_path[i][0] * np.sin(self.init_heading) + self.outbound_path[i][1] * np.cos(self.init_heading)
        
        self.outbound_path_yaw = self.generate_outbound_path_yaw()  

    def get_moving_step(self):
        return self.smallest_step + 0.4 * (self.distance / 3.0)
        
    def timer_callback(self) -> None:
        self.publish_offboard_control_heartbeat_signal()
        
        self.current_heading = self.get_current_heading(
            self.vehicle_odometry.q[0], 
            self.vehicle_odometry.q[1], 
            self.vehicle_odometry.q[2], 
            self.vehicle_odometry.q[3]
        )

        self.get_logger().info(f'Mode: {self.mode}, State: {self.state}, Idx: {self.pathidx}')
        current_position = (self.vehicle_odometry.position[0], self.vehicle_odometry.position[1])
        self.get_logger().info(f'Current pos: {current_position}, Target pos: ({self.next_waypoint_x:.2f}, {self.next_waypoint_y:.2f})')
        self.get_logger().info(f'LIDAR (F/L/R): {self.lidar_dist_front:.1f}m / {self.lidar_dist_left:.1f}m / {self.lidar_dist_right:.1f}m')
        
        if self.mode == 'homing':
            self.current_evade_distance = self.oa_homing_evade_distance
            self.current_stop_distance = self.oa_homing_stop_distance
        else:
            self.current_evade_distance = self.oa_evade_distance
            self.current_stop_distance = self.oa_stop_distance


        if self.offboard_setpoint_counter == 10:
            self.engage_offboard_mode()
            self.arm()
            self.lift_heading = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
            self.lift_position = (self.vehicle_odometry.position[0], self.vehicle_odometry.position[1])
            if self.obstacle_avoidance:
                self.get_logger().info('Obstacle avoidance enabled')

        # once liftoff is done, take snapshot is allowed (State 0 -> 1 or 2)
        if np.abs(self.vehicle_odometry.position[2] - self.takeoff_height) <= 0.1 and self.state == 0:
            if self.mode == 'homing':
                self.get_logger().info('Lift off done, taking the first snapshot')
                self.snapshot_lock = False
                self.state = 1
            elif self.mode == 'outbound':
                self.get_logger().info('Lift off done, performing outbound path')
                self.reach_point_margin = 1.0
                self.yaw_lock = False
                self.state = 2
                self.generate_outbound_path()
                self.next_heading = self.calculate_next_point_yaw(current_position, self.outbound_path[self.outbound_path_idx])

        # taking off
        if self.state == 0 and self.offboard_setpoint_counter >= 10:
            self.get_logger().info('Taking off')
            if self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD:
                self.next_waypoint_x = self.lift_position[0]
                self.next_waypoint_y = self.lift_position[1]
                self.next_heading = self.lift_heading

        # take snapshot (State 1)
        if self.snapshot_lock == False and self.state == 1 and self.mode == 'homing':
            self.get_logger().info("Snapshot request sent, waiting for output")
            self.snapshot_heading = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
            self.snapshot_x = self.vehicle_odometry.position[0]
            self.snapshot_y = self.vehicle_odometry.position[1]
            self.trigger_camera()
            self.pathidx += 1
            self.snapshot_lock = True
                
        # output received (State 1 -> 2)
        if self.output_received == True and self.state == 1 and self.mode == 'homing':
            self.get_logger().info(f'Output received: {self.output}, {self.distance}')
            self.output_received = False
            self.yaw_lock = False
            self.state = 2

        # landing logic
        if self.state == 4 and np.sqrt((self.vehicle_odometry.position[0]-self.next_waypoint_x)**2 + (self.vehicle_odometry.position[1]-self.next_waypoint_y)**2) < 0.05:
            self.get_logger().info('Reached landing point, descending')
            self.next_waypoint_x = self.vehicle_odometry.position[0]
            self.next_waypoint_y = self.vehicle_odometry.position[1]
            self.takeoff_height = 0.0

        if self.state == 4 and self.vehicle_odometry.position[2] >= -0.1:
            self.land()
            exit(0)

        # rotate (State 2)
        if self.yaw_lock == False and self.state == 2:
            if self.mode == 'homing':
                self.next_heading = self.output + self.snapshot_heading
            elif self.mode == 'outbound':
                self.trigger_camera()
                self.next_heading = self.calculate_next_point_yaw([self.vehicle_odometry.position[0], self.vehicle_odometry.position[1]], self.outbound_path[self.outbound_path_idx])
            elif self.mode == 'inbound':
                self.next_heading = self.calculate_next_point_yaw([self.vehicle_odometry.position[0], self.vehicle_odometry.position[1]], [0.0, 0.0])
            self.get_logger().info(f"Yawing to next heading: {self.next_heading}")

            self.next_heading = self.normalize_angle(self.next_heading)
            self.moving_step = self.get_moving_step()
            self.yaw_lock = True
        
        # Check yaw completion and OA path check (State 2 -> 3)
        if self.state == 2:
            self.current_heading = self.get_current_heading(self.vehicle_odometry.q[0], self.vehicle_odometry.q[1], self.vehicle_odometry.q[2], self.vehicle_odometry.q[3])
            if self.compare_yaw(self.next_heading, self.current_heading) < self.reach_yaw_margin and np.abs(self.vehicle_odometry.angular_velocity[2]) < 0.1:
                
                if self.obstacle_avoidance:
                    self.get_logger().info(f'Checking path. Front: {self.lidar_dist_front:.2f}m')
                    
                    if self.lidar_dist_front < self.current_stop_distance:
                        # Blocked! Start evasion state machine.
                        self.get_logger().warn('Path blocked! Entering OA state 10.')
                        self.state = 10 
                        self.oa_original_heading = self.next_heading
                    else:
                        # Path is clear
                        self.oa_attempt_counter = 0
                        self.get_logger().info('Path clear. Proceeding.')
                        self.position_lock = False
                        self.state = 3
                
                else: 
                    self.get_logger().info('Reached next heading, moving to the next waypoint (OA disabled)')
                    if self.mode == 'outbound':
                        self.trigger_camera()
                    self.position_lock = False
                    self.state = 3

        # move to next waypoint (State 3)
        if self.position_lock == False and self.state == 3:
            self.desired_velocity = [float("NaN"), float("NaN"), float("NaN")]
            if self.mode == 'homing':
                self.next_waypoint_x = self.snapshot_x+self.moving_step*np.cos(self.next_heading)
                self.next_waypoint_y = self.snapshot_y+self.moving_step*np.sin(self.next_heading)
                self.position_lock = True
                self.get_logger().info(f"Moving to next waypoint: {self.next_waypoint_x, self.next_waypoint_y}")
            elif self.mode == 'outbound':
                if self.outbound_path_idx >= len(self.outbound_path):
                    self.get_logger().info('Outbound path completed, performing inbound path')
                    self.mode = 'inbound'
                    self.yaw_lock = False
                    self.state = 2
                    self.position_clamp = self.preset_position_clamp
                else:
                    self.next_waypoint_x = self.outbound_path[self.outbound_path_idx][0]
                    self.next_waypoint_y = self.outbound_path[self.outbound_path_idx][1] 
                    self.outbound_path_idx += 1
                    self.position_lock = True
                    self.get_logger().info(f"Moving to next waypoint: {self.next_waypoint_x, self.next_waypoint_y}")
            elif self.mode == 'inbound':
                self.next_waypoint_x = 0.0
                self.next_waypoint_y = 0.0
                self.position_lock = True

        # Mid-flight OA check
        if self.state == 3 and self.obstacle_avoidance:
            if self.lidar_dist_front < self.current_stop_distance:
                self.get_logger().warn('Obstacle detected mid-flight! Emergency brake.')
                self.next_waypoint_x = self.vehicle_odometry.position[0]
                self.next_waypoint_y = self.vehicle_odometry.position[1]
                self.state = 10 
                self.oa_original_heading = self.next_heading
        
        # Check waypoint completion (State 3 -> 3a or 2)
        if np.sqrt((self.vehicle_odometry.position[0] - self.next_waypoint_x)**2 + (self.vehicle_odometry.position[1] - self.next_waypoint_y)**2) < self.reach_point_margin and self.state == 3:
            if self.mode != 'outbound':
                self.state = "3a"
            else:
                if self.outbound_path_idx < len(self.outbound_path):
                    self.yaw_lock = False
                    self.state = 2
                    self.position_clamp = self.preset_position_clamp
                else:
                    self.get_logger().info('Outbound path completed, performing inbound path')
                    self.mode = 'inbound'
                    self.yaw_lock = False
                    self.state = 2
                    self.position_clamp = self.preset_position_clamp
        
        if np.sqrt((self.vehicle_odometry.velocity[0])**2 + (self.vehicle_odometry.velocity[1])**2) < 0.1 and self.state == "3a":
            self.desired_velocity = [0.0, 0.0, 0.0]
            self.next_waypoint_x = float("NaN") 
            self.next_waypoint_y = float("NaN")
            
            if self.mode == 'homing':
                self.get_logger().info('Reached next waypoint, taking next snapshot')
                self.snapshot_lock = False
                self.state = 1
            
            elif self.mode == 'inbound':
                self.get_logger().info('Reached odometry home, performing homing')
                self.mode = 'homing'
                self.reach_0_counter = 0
                self.reach_point_margin = 0.10
                self.position_clamp = 1.0
                self.yaw_clamp = 0.4
                
        # Wait a bit after reaching home
        if self.state == "3a" and 100 > self.reach_0_counter > 60 and self.mode == 'homing':
            self.state = 1
            self.snapshot_lock = False
            self.output_received = False
            self.reach_point_margin = 0.1
            self.reach_0_counter = 1000

        # Incremental/gradual decrease in speed when approaching goal
        speed_limits = {1.0: 0.2, 3.0: 0.4, 5.0: 1.0, 10.0: 1.5}

        if self.state == 3 and self.mode != 'homing':
            x, y = self.vehicle_odometry.position[0], self.vehicle_odometry.position[1]
            goal_x, goal_y = self.next_waypoint_x, self.next_waypoint_y
            
            distance_to_goal = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)
            
            self.position_clamp = self.preset_position_clamp
            for threshold, speed in sorted(speed_limits.items()):
                if distance_to_goal < threshold:
                    self.position_clamp = speed
                    break

        # [State 10] OA_DECIDE
        if self.state == 10: 
            self.get_logger().info(f'OA: Deciding. L/R: {self.lidar_dist_left:.1f}m / {self.lidar_dist_right:.1f}m')
            # Hover
            self.next_waypoint_x = self.vehicle_odometry.position[0]
            self.next_waypoint_y = self.vehicle_odometry.position[1]
            self.next_heading = self.oa_original_heading

            left_clear = self.lidar_dist_left > self.current_stop_distance
            right_clear = self.lidar_dist_right > self.current_stop_distance

            if left_clear and (self.lidar_dist_left > self.lidar_dist_right):
                self.get_logger().info('OA: Evading left.')
                self.oa_target_yaw = self.normalize_angle(self.oa_original_heading - np.deg2rad(36.0))
                self.state = 11 
            elif right_clear:
                self.get_logger().info('OA: Evading right.')
                self.oa_target_yaw = self.normalize_angle(self.oa_original_heading + np.deg2rad(36.0))
                self.state = 12 
            else:
                # Completely blocked
                self.get_logger().warn(f'OA: Fully blocked in mode {self.mode}.')

                if self.mode == 'homing':
                    self.get_logger().info('OA: Homing blocked. Random evade.')
                    self.state = 15 
                
                elif self.mode == 'outbound':
                    if self.oa_attempt_counter < 1:
                        self.get_logger().info(f'OA: Outbound blocked. Attempting Random Evade (Attempt {self.oa_attempt_counter + 1}).')
                        self.oa_attempt_counter += 1
                        self.state = 15 
                    else:
                        self.get_logger().warn('OA: Outbound blocked. Retries exhausted. Skipping Waypoint.')
                        self.oa_attempt_counter = 0
                        if self.outbound_path_idx < len(self.outbound_path):
                            self.outbound_path_idx += 1 
                            self.yaw_lock = False
                            self.state = 2
                        else:
                            self.mode = 'inbound'
                            self.state = 2

                elif self.mode == 'inbound':
                    dist_to_home = np.sqrt(self.vehicle_odometry.position[0]**2 + self.vehicle_odometry.position[1]**2)
                    
                    if dist_to_home < 2.0:
                        self.get_logger().warn('OA: Inbound blocked near home (<2m). Switching to Visual Homing.')
                        self.mode = 'homing'
                        self.state = 1 
                        self.snapshot_lock = False
                    else:
                        self.get_logger().info('OA: Inbound blocked. Random evade.')
                        self.state = 15 

        # [State 11 & 12] DIRECTED EVADE YAW
        if self.state == 11 or self.state == 12: 
            self.next_heading = self.oa_target_yaw
            self.next_waypoint_x = self.vehicle_odometry.position[0]
            self.next_waypoint_y = self.vehicle_odometry.position[1]
            
            if self.compare_yaw(self.current_heading, self.oa_target_yaw) < self.reach_yaw_margin:
                self.get_logger().info('OA: Yaw reached. Calculating evade path.')
                
                current_pos = self.vehicle_odometry.position
                self.oa_evade_position = [
                    current_pos[0] + self.current_evade_distance * np.cos(self.oa_target_yaw),
                    current_pos[1] + self.current_evade_distance * np.sin(self.oa_target_yaw)
                ]
                self.state = 13 

        # [State 15] OA_RANDOM_ROTATE
        if self.state == 15: 
            self.get_logger().info('OA: Picking random direction.')
            self.next_waypoint_x = self.vehicle_odometry.position[0]
            self.next_waypoint_y = self.vehicle_odometry.position[1]
            
            rotation_angle = random.choice([-np.deg2rad(36.0), np.deg2rad(36.0)])
            self.oa_target_yaw = self.normalize_angle(self.current_heading + rotation_angle)
            
            self.state = 16

        # [State 16] OA_RANDOM_YAW_CHECK
        if self.state == 16:
            self.next_heading = self.oa_target_yaw
            self.next_waypoint_x = self.vehicle_odometry.position[0]
            self.next_waypoint_y = self.vehicle_odometry.position[1]

            if self.compare_yaw(self.current_heading, self.oa_target_yaw) < self.reach_yaw_margin:
                # Check sensor after turn
                if self.lidar_dist_front > self.current_stop_distance:
                    self.get_logger().info('OA: Random direction is clear. Moving.')
                    
                    current_pos = self.vehicle_odometry.position
                    self.oa_evade_position = [
                        current_pos[0] + self.current_evade_distance * np.cos(self.oa_target_yaw),
                        current_pos[1] + self.current_evade_distance * np.sin(self.oa_target_yaw)
                    ]
                    
                    self.state = 13 
                else:
                    self.get_logger().warn('OA: Random direction also blocked! Retrying.')
                    self.state = 15

        # [State 13] OA_MOVE
        if self.state == 13: 
            if self.lidar_dist_front < self.current_stop_distance:
                self.get_logger().warn('OA: Blocked during evade move! Stopping.')
                self.state = 10 
                self.next_waypoint_x = self.vehicle_odometry.position[0]
                self.next_waypoint_y = self.vehicle_odometry.position[1]
            else:
                self.next_heading = self.oa_target_yaw
                self.next_waypoint_x = self.oa_evade_position[0]
                self.next_waypoint_y = self.oa_evade_position[1]
                
                dist_to_evade = np.sqrt((self.vehicle_odometry.position[0] - self.oa_evade_position[0])**2 + 
                                        (self.vehicle_odometry.position[1] - self.oa_evade_position[1])**2)
                
                if dist_to_evade < self.reach_point_margin:
                    self.get_logger().info('OA: Evade move done. Re-aligning to goal.')
                    self.state = 14

        # [State 14] OA_YAW_BACK
        if self.state == 14:
            self.next_heading = self.oa_original_heading 
            self.next_waypoint_x = self.vehicle_odometry.position[0]
            self.next_waypoint_y = self.vehicle_odometry.position[1]

            if self.compare_yaw(self.current_heading, self.oa_original_heading) < self.reach_yaw_margin:
                if self.mode == 'homing':
                    self.get_logger().info('OA: Re-aligned. Taking new snapshot.')
                    self.state = 1 
                    self.snapshot_lock = False
                else:
                    self.get_logger().info('OA: Re-aligned. Resuming path.')
                    self.state = 2

        self.publish_position_setpoint(
            self.next_waypoint_x, 
            self.next_waypoint_y, 
            self.takeoff_height, 
            self.next_heading, 
            self.desired_velocity[0], 
            self.desired_velocity[1], 
            self.desired_velocity[2]
        )
        
        self.offboard_setpoint_counter += 1
        self.reach_0_counter += 1
            

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
