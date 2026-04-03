import numpy as np
from matplotlib import pyplot as plt
from output_utils import _save_or_show

yaw_rate = 0.2  # rad/s
speed = 1.0 # m/s

def simulate(yaw_rate, speed, time=10):
    """
    Simulate the trajectory of a vehicle given yaw rate and speed.
    """
    dt = 0.01  # time step
    time_steps = int(time / dt)
    psi = 0  # initial orientation
    x, y = 0, 0  # initial position
    trajectory = []
    angles_home = []
    headings = []

    for _ in range(time_steps):
        # store state:
        headings.append(psi)
        trajectory.append((x, y))
        angle_home = np.arctan2(y, x)  -psi - np.pi # angle to home position
        angles_home.append(angle_home)
        # update state:
        psi += yaw_rate * dt
        x += speed * np.cos(psi) * dt
        y += speed * np.sin(psi) * dt


    angles_home = np.array(angles_home)
    # if angles are < 2pi or > 2pi, adjust them to be within [-pi, pi]
    angles_home = np.mod(angles_home + np.pi, 2 * np.pi) - np.pi
    angles_home = np.rad2deg(angles_home)  # convert to degrees

    return np.array(trajectory), angles_home, np.array(headings)

def calculate(yaw_rate, speed, times=[]):
    """
    Calculate the trajectory of a vehicle given yaw rate, speed, and times, algebraically.
    """

    if len(times) == 0:
        times = np.arange(0, 10, 0.1)

    x = np.zeros(len(times))
    y = np.zeros(len(times))
    angles_home = np.zeros(len(times))
    headings = np.zeros(len(times))

    for i, time in enumerate(times):
        x[i] = (speed / yaw_rate) * np.sin(yaw_rate * time)
        y[i] = (speed / yaw_rate) * (1 - np.cos(yaw_rate * time))
        angles_home[i] = np.arctan2(y[i], x[i]) - (yaw_rate * time) - np.pi
        headings[i] = yaw_rate * time

    trajectory = np.column_stack((x, y))

    # if angles are < 2pi or > 2pi, adjust them to be within [-pi, pi]
    angles_home = np.mod(angles_home + np.pi, 2 * np.pi) - np.pi
    angles_home = np.rad2deg(angles_home)  # convert to degrees

    return trajectory, angles_home, headings

def run_tortuosity_analysis(output_dir = None):
    T = 30
    dt_sim = 0.01  # simulation time step
    dt_calc = 1.0  # calculation time step
    simulated_trajectory, simulated_angles, simulated_headings = simulate(yaw_rate, speed, time = T)
    calculated_trajectory, calculated_angles, calculated_headings = calculate(yaw_rate, speed, times=np.arange(0, T, dt_calc))

    yaw_rate_bias = np.deg2rad(0.5)  # rad/s
    simulated_real_trajectory, simulated_real_angles, simulated_real_headings = simulate(yaw_rate + yaw_rate_bias, speed, time=T)
    calculated_real_trajectory, calculated_real_angles, calculated_real_headings = calculate(yaw_rate + yaw_rate_bias, speed, times=np.arange(0, T, dt_calc))

    plt.figure()
    plt.plot(simulated_trajectory[:, 0], simulated_trajectory[:, 1], label='Simulated Presumed Trajectory')
    plt.plot(calculated_trajectory[:, 0], calculated_trajectory[:, 1], 'bx', label='Calculated Presumed Trajectory')
    plt.plot(simulated_real_trajectory[:, 0], simulated_real_trajectory[:, 1], 'g', label='Simulated Real Trajectory', linestyle='--')
    plt.plot(calculated_real_trajectory[:, 0], calculated_real_trajectory[:, 1], 'gx', label='Calculated Real Trajectory', linestyle='--')
    plt.legend()
    plt.title('Vehicle Trajectory Simulation')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    _save_or_show(output_dir, 'vehicle_trajectory')

    plt.figure()
    plt.plot(np.arange(0, T, dt_sim), simulated_angles, label='Simulated Angles to Home')
    plt.plot(np.arange(0, T, dt_calc), calculated_angles, 'bx', label='Calculated Angles to Home')
    plt.plot(np.arange(0, T, dt_sim), simulated_real_angles, 'g', label='Simulated Real Angles to Home', linestyle='--')
    plt.plot(np.arange(0, T, dt_calc), calculated_real_angles, 'gx', label='Calculated Real Angles to Home')
    plt.legend()
    plt.title('Angles to Home Position')
    plt.xlabel('Time (s)')
    plt.ylabel('Angle (rad)')
    _save_or_show(output_dir, 'angles_to_home')

    angular_error = np.deg2rad(simulated_angles) - np.deg2rad(simulated_real_angles)
    heading_based_error = simulated_real_headings - simulated_headings
    position_based_error = np.arctan2(simulated_trajectory[:, 1], simulated_trajectory[:, 0]) \
                                  - np.arctan2(simulated_real_trajectory[:, 1], simulated_real_trajectory[:, 0])
    angular_error = np.rad2deg(angular_error)
    heading_based_error = np.rad2deg(heading_based_error)
    position_based_error = np.rad2deg(position_based_error)

    calculated_angular_error = np.deg2rad(calculated_angles) - np.deg2rad(calculated_real_angles)
    calculated_heading_based_error = yaw_rate_bias * np.arange(0, T, dt_calc)
    calculated_position_based_error = np.arctan2(calculated_trajectory[:, 1], calculated_trajectory[:, 0]) \
                                  - np.arctan2(calculated_real_trajectory[:, 1], calculated_real_trajectory[:, 0])
    calculated_angular_error = np.rad2deg(calculated_angular_error)
    calculated_heading_based_error = np.rad2deg(calculated_heading_based_error)
    calculated_position_based_error = np.rad2deg(calculated_position_based_error)

    plt.figure()
    plt.plot(np.arange(0, T, dt_sim), angular_error, 'b-', label='Angular Error')
    plt.plot(np.arange(0, T, dt_sim), heading_based_error, 'r--', label='Heading Based Error')
    plt.plot(np.arange(0, T, dt_sim), position_based_error, 'g--', label='Position Based Error')
    plt.plot(np.arange(0, T, dt_calc), calculated_angular_error, 'bx', label='Calculated Angular Error')
    plt.plot(np.arange(0, T, dt_calc), calculated_heading_based_error, 'rx', label='Calculated Heading Based Error')
    plt.plot(np.arange(0, T, dt_calc), calculated_position_based_error, 'gx', label='Calculated Position Based Error')
    plt.legend()
    plt.title('Error Analysis')
    plt.xlabel('Time (s)')
    plt.ylabel('Error (deg)')
    _save_or_show(output_dir, 'error_analysis')

    print("Done")

if __name__ == "__main__":
    run_tortuosity_analysis()
