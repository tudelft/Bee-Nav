from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import ExecuteProcess
from datetime import datetime

# Note: /home/data/ is the data directory on the drone's Raspberry Pi
os.makedirs(datetime.now().strftime('/home/data/%Y%m%d'), exist_ok=True)


def generate_launch_description():
    config = os.path.join(
    get_package_share_directory('learning_flight'),
    'config',
    'params.yaml'
    )

    bag_dir_name = datetime.now().strftime('/home/data/%Y%m%d/rosbag_%Y%m%d-%H%M%S')

    topics_to_record = [
        '/fmu/out/vehicle_odometry', 
        '/fmu/in/vehicle_motion_capture'
    ]

    rosbag_record = ExecuteProcess(
        cmd=['ros2', 'bag', 'record', '-o', bag_dir_name] + topics_to_record,
        output='screen',
        emulate_tty=True, # Necessary for ros2 bag to run correctly
        # This tag ensures the rosbag process is killed cleanly when you stop the launch file (Ctrl+C)
        name='rosbag_recorder',
    )

    mocap_relay_node = Node(
        package='mocap_relay',
        executable='mocap_relay_node',
        name='mocap_relay',
        output='screen',
    )

    learning_control_node = Node(
        package='learning_flight',  # Replace with your package name
        executable='learning',  # Replace with your node's executable name
        name='learning_control_node',  # Optional: Replace with a custom node name if you wish
        output='screen',  # Output logs to the screen
        parameters=[config],  # Pass the path to the YAML file
    )

    return LaunchDescription([
        rosbag_record,
        mocap_relay_node,
        learning_control_node,
    ])
