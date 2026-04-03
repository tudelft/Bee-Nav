import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

from px4_msgs.msg import TimesyncStatus, VehicleOdometry

class MocapRelayNode(Node):
    """
    A ROS 2 node to relay motion capture data to PX4, ensuring timestamp synchronization.
    """
    def __init__(self):
        super().__init__('mocap_relay_node')

        # Configure a QoS profile for BEST_EFFORT sensor data from PX4
        sensor_qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # --- Subscribers ---
        self.timesync_sub = self.create_subscription(
            TimesyncStatus,
            '/fmu/out/timesync_status',
            self.timesync_callback,
            sensor_qos_profile  # Use the corrected QoS profile
        )
        self.mocap_sub = self.create_subscription(
            VehicleOdometry,
            '/dummy',  # Subscribing to your external mocap topic
            self.mocap_callback,
            10  # Standard QoS for high-rate data
        )

        # --- Publisher ---
        self.odometry_pub = self.create_publisher(
            VehicleOdometry,
            # '/fmu/in/vehicle_visual_odometry', # to fly with mocap
            '/vehicle_motion_capture',        # to log only
            10 # Standard QoS
        )

        # --- State Variables ---
        self.timesync_offset_ns = 0
        self.get_logger().info('Mocap Relay Node started. Waiting for timesync and mocap data...')

    def timesync_callback(self, msg: TimesyncStatus):
        """
        Callback for the timesync topic. Calculates the offset between ROS time and PX4 time.
        """
        ros_time_ns = self.get_clock().now().nanoseconds
        px4_time_us = msg.timestamp
        
        # The offset is the difference between the computer's time and the flight controller's time
        self.timesync_offset_ns = ros_time_ns - (px4_time_us * 1000)
        self.get_logger().info(f"Timesync received. Offset: {self.timesync_offset_ns} ns", once=True)


    def mocap_callback(self, msg: VehicleOdometry):
        """
        Callback for the motion capture data. Corrects the timestamp and republishes.
        """
        if self.timesync_offset_ns == 0:
            # rclpy logger doesn't have a built-in throttle method like rclcpp.
            # A simple warning is sufficient here as it only appears at the start.
            self.get_logger().warn('No timesync received yet. Skipping mocap message.')
            return

        # Get the current ROS time in nanoseconds
        ros_time_ns = self.get_clock().now().nanoseconds

        # Calculate the synchronized PX4 timestamp in microseconds
        px4_timestamp_us = (ros_time_ns - self.timesync_offset_ns) // 1000

        # Create a new message and populate it with the incoming data
        republished_msg = VehicleOdometry()
        republished_msg.timestamp = px4_timestamp_us
        republished_msg.timestamp_sample = px4_timestamp_us

        # Copy all the data from the incoming message
        republished_msg.pose_frame = msg.pose_frame
        republished_msg.position = msg.position
        republished_msg.q = msg.q
        republished_msg.velocity_frame = msg.velocity_frame
        republished_msg.velocity = msg.velocity
        republished_msg.angular_velocity = msg.angular_velocity
        republished_msg.position_variance = msg.position_variance
        republished_msg.orientation_variance = msg.orientation_variance
        republished_msg.velocity_variance = msg.velocity_variance

        # Publish the message with the corrected timestamp
        self.odometry_pub.publish(republished_msg)
        self.get_logger().debug(f"Relayed mocap data with synced timestamp: {px4_timestamp_us}")


def main(args=None):
    rclpy.init(args=args)
    node = MocapRelayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # This check prevents the error on shutdown
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
