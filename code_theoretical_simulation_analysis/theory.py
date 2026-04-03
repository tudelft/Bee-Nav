import numpy as np
import matplotlib.pyplot as plt
from output_utils import _save_or_show

def find_nest_ID_AD(observation, heading_robot, landmarks, ground_truth_location, region_limits, verbose = False):
    """ Manual algorithm to find the nest location from the observation of landmarks without heading information."""

    # Algorithm to find the nest location from the observation of landmarks without heading information
    n_landmarks = landmarks.shape[0]
    if(n_landmarks >= 2):
        # The landmark orientation forms a heading reference frame, so we can find the nest.
        # Without loss of generality, we can use the first two landmarks:
        xb1 = observation[0]
        yb1 = observation[1]
        xb2 = observation[2]
        yb2 = observation[3]
        delta_xb = xb2 - xb1
        delta_yb = yb2 - yb1
        delta_psi = np.arctan2(delta_yb, delta_xb) - np.pi/2
        delta_xw = landmarks[1,0] - landmarks[0,0]
        delta_yw = landmarks[1,1] - landmarks[0,1]
        psi_landmarks = np.arctan2(delta_yw, delta_xw) - np.pi/2
        psi_robot = delta_psi - psi_landmarks
        if verbose:
            print(f'Distance landmarks: true = {np.linalg.norm(landmarks[1]-landmarks[0])}, calculated = {np.linalg.norm([delta_xb, delta_yb])}')
            print(f'Heading landmarks: world = {np.rad2deg(psi_landmarks)} relative to robot = {np.rad2deg(delta_psi)}')
            print(f'Psi_robot: true = {np.rad2deg(heading_robot)}, calculated = {np.rad2deg(psi_robot)}')

        # find the relative location of the first landmark in the world frame:
        c_psi = np.cos(psi_robot)
        s_psi = np.sin(psi_robot)        
        if(c_psi > 1e-5):    
            dy1 = (yb1 / c_psi - (s_psi * xb1) / c_psi**2) / (1 + (s_psi / c_psi)**2)
            dx1 = (s_psi * dy1 + xb1) / c_psi
        else:
            dx1 = (yb1 / s_psi + (c_psi * xb1) / s_psi**2) / (1 + (c_psi / s_psi)**2)
            dy1 = (c_psi * dx1 - xb1) / s_psi
        
        # find the location of the robot:
        xw1 = landmarks[0,0] - dx1
        yw1 = landmarks[0,1] - dy1

        if(verbose):
            print(f'Robot location: true = {ground_truth_location}, calculated = [{xw1}, {yw1}]')

        if(np.sqrt((xw1-ground_truth_location[0])**2 + (yw1-ground_truth_location[1])**2) > 0.5):
            print('Error in the calculation of the nest location')
            print(f'ground_truth_location = {ground_truth_location}, calculated = [{xw1}, {yw1}]')
        
        dir_nest_world = np.asarray([-xw1, -yw1])
        # Rotate with the heading:
        R = np.array([[np.cos(heading_robot), -np.sin(heading_robot)], [np.sin(heading_robot), np.cos(heading_robot)]])
        dir_nest_robot = np.dot(R, dir_nest_world)

        return dir_nest_robot
        #return np.asarray([-xw1, -yw1])
    
    elif(n_landmarks == 1):
        # This is only optimal when the landmark is in location (10, 10) 
        xb1 = observation[0]
        yb1 = observation[1]
        x_norm = xb1 / np.linalg.norm([xb1, yb1])
        y_norm = yb1 / np.linalg.norm([xb1, yb1])
        dist_obs = np.linalg.norm([xb1, yb1])
        if(landmarks[0,0] == 10 and landmarks[0,1] == 10):
            r_opt = 40 / np.pi 
            delta = r_opt - dist_obs
        else:
            delta = dist_obs
        return np.asarray([-x_norm * delta, -y_norm * delta])

def plot_error_radii(landmarks, region_limits, nest_location=[0,0], new_figure = True, output_dir = None):
    """ Plot the error for different radii around a single landmark.
        If the landmark is at (10, 10), it will also plot the theoretical optimal radius."""

    landmarks = np.asarray(landmarks)

    # The robot can only pick a distance (0 or otherwise, as direction cannot be discerned)
    rs = np.linspace(0, 30, 20)
    rs = rs[-1::-1]
    min_error = 1e6
    best_r = 0
    xs = []
    ys = []
    zs = []
    errors = []
    # gather error data for different radii around the landmark:
    for r in rs:
        error = 0
        n_points = 0
        x = []
        y = []
        z = []
        for t in np.linspace(0, 2*np.pi, 100):
            xw = r * np.cos(t) + landmarks[0,0]
            yw = r * np.sin(t) + landmarks[0,1]
            if(xw > region_limits[0] and xw < region_limits[1] and yw > region_limits[0] and yw < region_limits[1]):
                x.append(xw)
                y.append(yw)
                err = xw**2 + yw**2 # Squared Error with (0,0) nest location
                z.append(err)
                error += err
                n_points += 1
        if(n_points > 0):
            error /= n_points # Mean Squared Error
            if(error < min_error):
                min_error = error
                best_r = r
            xs.append(x)
            ys.append(y)
            zs.append(z)

    errors = []
    for i in range(len(xs)):
        errors.append(np.mean(zs[i]))
    norm_errors = errors - np.min(errors)
    norm_errors /= np.max(norm_errors)

    if(landmarks[0][0] == 10 and landmarks[0][1] == 10):    
        # Make the line that theoretically minimizes the error
        r_opt = 40 / np.pi
        x_opt = []
        y_opt = []
        z_opt = []
        error = 0
        n_points = 0
        for t in np.linspace(0, 2*np.pi, 100):
            xw = r_opt * np.cos(t) + landmarks[0,0]
            yw = r_opt * np.sin(t) + landmarks[0,1]
            if(xw > region_limits[0] and xw < region_limits[1] and yw > region_limits[0] and yw < region_limits[1]):
                x_opt.append(xw)
                y_opt.append(yw)
                err = xw**2 + yw**2 # Squared Error with (0,0) nest location
                z_opt.append(err)
                error += err
                n_points += 1
        if(n_points > 0):
            error /= n_points
        error_opt = error

    # plot the circles:
    red = np.asarray([1, 0, 0])
    green = np.asarray([0, 1, 0])
    if(new_figure):
        plt.figure()
    # plot the nest location:
    plt.plot(0, 0, 'g*')
    plt.plot(landmarks[0,0], landmarks[0,1], 'bo')
    if(landmarks[0][0] == 10 and landmarks[0][1] == 10): 
        plt.plot(x_opt, y_opt, 'k--')
        n = len(x_opt)
        # take a random number between 0 and n-1:
        j = int(0.9 * n) 
        # plot text with a red font color:
        plt.text(x_opt[j], y_opt[j], f'$\\mathcal{{L}}^{{*}} = {error_opt:.2f}$', color = 'green', weight='bold')

    for i in range(len(xs)):
        plt.plot(xs[i], ys[i], color = norm_errors[i] * red + (1-norm_errors[i]) * green)
        # add a text label to the circle:
        n = len(xs[i])
        # take a random number between 0 and n-1:
        j = np.random.randint(0, n)
        plt.text(xs[i][j], ys[i][j], f'$\\mathcal{{L}} = {errors[i]:.2f}$', weight='bold')

    if(new_figure):
        _save_or_show(output_dir, 'error_radii')

    if(landmarks[0][0] == 10 and landmarks[0][1] == 10):
        print(f'Best tried r = {best_r}, min_error = {min_error}, best theoretical r = {r_opt}, min_theoretical_error = {error_opt}')       

