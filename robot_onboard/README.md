# Onboard Robot Code

This folder contains the software stack for the paper *"Efficient robot navigation inspired by honeybee learning flights"*, designed to run onboard the robot hardware (Raspberry Pi 4 with PX4 flight controller).

The system architecture is split into two main components handling distinct responsibilities: high-level flight control and visual navigation inference.

## System Architecture

The codebase is organized into two primary subsystems that communicate via HTTP requests:

1.  **`homingdrone-ros2` (Flight Control)**: A ROS 2 workspace that interfaces with the PX4 flight controller to execute flight patterns for data collection and perform homing maneuvers based on navigation vector.
2.  **`picamera` (Vision & Learning)**: A service managing the omnidirectional camera, logging sensor data, and running the neural network for online inference or data collection.

Both components are designed to run in separate Docker containers for environment isolation and ease of deployment.

## Repository Structure

### 1. `homingdrone-ros2`
This folder contains the ROS 2 packages responsible for vehicle control. It utilizes the `micro_ros_agent` to bridge with the PX4 uORB middleware.

**Key Nodes:**
*   **`learning_control_node`**: Automates flight paths for data collection and online learning.
    *   **Flight Patterns**: Defines the shape and size of the learning flight. We mainly used `'bee'` (wasp-inspired looping paths) for small learning areas and `'bee_outdoor_3_op'` for larger outdoor learning areas.
    *   **State Machine**: Manages the flight phases: Takeoff $\to$ Hover/Snapshot $\to$ Yaw $\to$ Move to Target $\to$ Land.
    *   **Obstacle Avoidance**: Integrates with LiDAR sensors (`/tf_nova/dist_*`) to perform "nudge" or "panic" maneuvers if the path is blocked.
    *   **Camera Triggering**: Sends HTTP requests to the `picamera` service to capture synchronized images and pose data at specific waypoints.
*   **`homing_control_node`**: Executes homing flights using the trained model for inference.
    *   **Homing**: Uses the predicted homing vector from the `picamera` service to navigate back to the home location.
    *   **State Machine**: Manages the flight phases: Takeoff $\to$ Outbound Flight $\to$ Inbound Flight $\to$ Homing Flight $\to$ Land.
    *   **Full flight/Homing only**: Configures parameters for different environments (indoor/outdoor) with predefined outbound shapes, then the robot performs outbound flight followed by inbound and then homing. Or only homing flight from a random location.
    *   **Camera Triggering**: Similar to learning node, triggers image capture and inference during homing.


### 2. `picamera`

This subsystem acts as the vision server for the robot.

*   **Service**: Runs a local HTTP server (port 5000) listening for `/trigger` events.
*   **Functionality**:
    *   Captures images from the omnidirectional lens.
    *   Records synchronized odometry (heading, pitch, roll) from the motion capture system or VIO.
    *   **Training Mode**: Activated during the learning phase. Performs online learning using the captured image and data. Also saves image-pose pairs for offline training when required.
    *   **Inference Mode**: Activated during the homing phase. Feeds the current view into the navigation network to predict the homing vector, and sends the predicted direction and distance back to `homingdrone-ros2` for flight control.

## Getting Started

### Prerequisites
*   Raspberry Pi 4 (Ubuntu Server 20.04/22.04 recommended)
*   Docker & Docker Compose
*   Flight controller with PX4 Autopilot (connected via UART/Serial)
*   Raspberry Pi Camera Module (with omnidirectional lens)


### Installation
For use of the PX4 with ROS 2, please refer to the [PX4 ROS 2 User Guide](https://docs.px4.io/main/en/ros2/user_guide).

Pi-camera can be used with the standard Raspberry Pi Camera Module drivers. Ensure that the camera is properly connected and enabled.

