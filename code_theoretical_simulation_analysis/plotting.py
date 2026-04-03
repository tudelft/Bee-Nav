import matplotlib.pyplot as plt
import numpy as np
import torch
from theory import *
from ANN import *
from matplotlib.patches import FancyArrow
from output_utils import _save_or_show


def plot_N(out, n_grid_points, nest_location, x, y, parameters = None, output_dir = None):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    z = out.data.numpy().reshape(n_grid_points, n_grid_points, 2)
    # show a plot of the prediction with arrows:
    fig, ax = plt.subplots()
    ax.quiver(x, y, z[:,:,0], z[:,:,1])
    ax.set_xlabel('$x$ [m]')
    ax.set_ylabel('$y$ [m]')
    # plot the nest location
    ax.plot(nest_location[0], nest_location[1], 'g*')
    # plot the landmarks
    landmarks = np.asarray(parameters['landmarks'])
    if(not parameters['ID']):
        ax.plot(landmarks[:,0], landmarks[:,1], 'bo')
    else:
        color_values = np.linspace(0, 1, parameters['n_landmarks'])
        ax.scatter(landmarks[:,0], landmarks[:,1], c=color_values, cmap='brg')
        ax.set_title('Predicted nest location')
    if(parameters['analyze_generalization']):
        #  show the learning region with a dashed red box:
        dx, dy = parameters['offset_learning_region']
        ax.plot([-parameters['size_learning_region'] + dx, parameters['size_learning_region'] + dx, parameters['size_learning_region'] + dx, -parameters['size_learning_region'] + dx, -parameters['size_learning_region'] + dx],
                [-parameters['size_learning_region'] + dy, -parameters['size_learning_region'] + dy, parameters['size_learning_region'] + dy, parameters['size_learning_region'] + dy, -parameters['size_learning_region'] + dy], 'r--')
    _save_or_show(output_dir, 'prediction_vectors_N')

def plot_not_N(n_headings, out, n_samples_grid, headings, n_grid_points, x, y, parameters = None, output_dir = None):

    FS = 11
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # create a plot with n_headings subplots:
    fig, ax = plt.subplots(1, n_headings, figsize=(n_headings * 5, 5))

    landmarks = np.asarray(parameters['landmarks'])

    if(n_headings == 1):
        ax = [ax]

    if parameters['plot_error_radii']:
        if(parameters['ID'] and parameters['AD'] and not parameters['N'] and parameters['n_landmarks'] == 1 and not parameters['OCC']):
            plot_error_radii(landmarks, [-parameters['size_region'], parameters['size_region']], nest_location=[0,0], new_figure = False)

    for h in range(n_headings):
        out_heading = out[h*n_samples_grid:(h+1)*n_samples_grid, :]
        out_heading = out_heading.data.numpy()
        # transform back to the world frame:
        R = np.array([[np.cos(-headings[h]), -np.sin(-headings[h])], [np.sin(-headings[h]), np.cos(-headings[h])]])
        out_heading = np.dot(R, out_heading.T).T
        z = out_heading.reshape(n_grid_points, n_grid_points, 2)
        # show a plot of the prediction with arrows:
        ax[h].quiver(x, y, z[:,:,0], z[:,:,1])
        ax[h].set_xlabel('$x$ [m]', fontsize=FS)
        ax[h].set_ylabel('$y$ [m]', fontsize=FS)
        ax[h].set_title('Heading: ' + str(np.rad2deg(headings[h])), fontsize=FS)
        # plot the nest location
        ax[h].plot(parameters['nest_location'][0], parameters['nest_location'][1], 'g*')
        # plot the landmarks
        if(not parameters['ID']):
            # all blue
            ax[h].plot(landmarks[:,0], landmarks[:,1], 'bo')
        else:
            # all with different colors:
            color_values = np.linspace(0, 1, parameters['n_landmarks'])
            ax[h].scatter(landmarks[:,0], landmarks[:,1], c=color_values, cmap='brg')
        if(parameters['OCC']):
            # plot circles of radius radius around the landmarks:
            for j in range(parameters['n_landmarks']):
                circle = plt.Circle(landmarks[j], parameters['landmark_radius'], linestyle = ':', color='k', fill=False)
                ax[h].add_artist(circle)
        if(parameters['analyze_generalization']):
            #  show the learning region with a dashed red box:
            dx, dy = parameters['offset_learning_region']
            if not parameters['circular_learning_region']:
                ax[h].plot([-parameters['size_learning_region'] + dx, parameters['size_learning_region'] + dx, parameters['size_learning_region'] + dx, -parameters['size_learning_region'] + dx, -parameters['size_learning_region'] + dx],
                    [-parameters['size_learning_region'] + dy, -parameters['size_learning_region'] + dy, parameters['size_learning_region'] + dy, parameters['size_learning_region'] + dy, -parameters['size_learning_region'] + dy], 'r--')
            else:
                # plot a circle around the landmarks:
                offset_landmarks = np.mean(landmarks, axis=0)
                circle = plt.Circle(offset_landmarks + np.asarray(parameters['offset_learning_region']), parameters['size_learning_region'], linestyle = '--', color='r', fill=False)
                ax[h].add_artist(circle)

    if(n_headings == 1):
        # set axis equal:
        ax[0].set_aspect('equal')

    _save_or_show(output_dir, 'prediction_vectors')

def plot_theory_ID_AD(inputs, heading, xy, n_headings, n_samples_grid, headings, n_grid_points, x, y, parameters = None, output_dir = None):

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    landmarks = np.asarray(parameters['landmarks'])
    n_samples = len(heading)
    out_theory = np.zeros((n_samples, 2))
    for i in range(n_samples):
        out_theory[i,:] = find_nest_ID_AD(inputs[i,:].detach().numpy(), heading[i], landmarks, xy[i,:], [-parameters['size_region'], parameters['size_region']])
    # create a plot for the theoretical predictions  for all headings:
    fig, ax = plt.subplots(1, n_headings, figsize=(15, 5))
    if(n_headings == 1):
        ax = [ax]

    if(parameters['n_landmarks'] == 1):
        plot_error_radii(landmarks, [-parameters['size_region'], parameters['size_region']], nest_location=[0,0], new_figure = False)

    for h in range(n_headings):
        out_heading = out_theory[h*n_samples_grid:(h+1)*n_samples_grid, :]
        R = np.array([[np.cos(-headings[h]), -np.sin(-headings[h])], [np.sin(-headings[h]), np.cos(-headings[h])]])
        out_heading = np.dot(R, out_heading.T).T
        z = out_heading.reshape(n_grid_points, n_grid_points, 2)

        # show a plot of the prediction with arrows:
        ax[h].quiver(x, y, z[:,:,0], z[:,:,1])
        ax[h].set_xlabel('x')
        ax[h].set_ylabel('y')
        ax[h].set_title('Theoretical prediction for heading: ' + str(np.rad2deg(headings[h])))
        # plot the nest location
        ax[h].plot(parameters['nest_location'][0], parameters['nest_location'][1], 'g*')
        # plot the landmarks
        # all with different colors:
        color_values = np.linspace(0, 1, parameters['n_landmarks'])
        ax[h].scatter(landmarks[:,0], landmarks[:,1], c=color_values, cmap='brg')
    if(n_headings == 1):
        # set axis equal:
        ax[0].set_aspect('equal')
    _save_or_show(output_dir, 'theory_prediction')

def plot_end_points_not_N(n_headings, n_samples_grid, out, headings, xy, show = True, col = 'grey', out_2 = None, col_2 = 'black', sup_title_postfix = '', parameters = None, output_dir = None):
    fig, ax = plt.subplots(1, n_headings)
    landmarks = np.asarray(parameters['landmarks'])
    for h in range(n_headings):
        out_heading = out[h*n_samples_grid:(h+1)*n_samples_grid, :]
        out_heading = out_heading.data.numpy()
        # transform back to the world frame:
        R = np.array([[np.cos(-headings[h]), -np.sin(-headings[h])], [np.sin(-headings[h]), np.cos(-headings[h])]])
        out_heading = np.dot(R, out_heading.T).T
        # take the right input positions:
        xy_heading = xy[h*n_samples_grid:(h+1)*n_samples_grid]
        predicted_location = xy_heading + out_heading
        if n_headings == 1:
            ax = [ax]
        ax[h].plot(predicted_location[:,0], predicted_location[:,1], 'x', color=col)

        if out_2 is not None:
            out_heading_2 = out_2[h*n_samples_grid:(h+1)*n_samples_grid, :]
            out_heading_2 = out_heading_2.data.numpy()
            out_heading_2 = np.dot(R, out_heading_2.T).T
            predicted_location_2 = xy_heading + out_heading_2
            ax[h].plot(predicted_location_2[:,0], predicted_location_2[:,1], 'x', color=col_2)

        # plot the landmark:
        if not parameters['ID']:
            ax[h].plot(landmarks[:,0], landmarks[:,1], 'bo')
        else:
            color_values = np.linspace(0, 1, parameters['n_landmarks'])
            ax[h].scatter(landmarks[:,0], landmarks[:,1], c=color_values, cmap='brg')
        ax[h].set_xlabel('x')
        ax[h].set_ylabel('y')
        # set the axis range to the minimum and maximum of the grid:
        ax[h].set_xlim([np.min(xy[:,0]), np.max(xy[:,0])])
        ax[h].set_ylim([np.min(xy[:,1]), np.max(xy[:,1])])
        ax[h].set_title(f'Heading: {np.rad2deg(headings[h])} $^\circ$')
        # set axes equal:
        ax[h].set_aspect('equal')
    # set the title of the window of the figure:
    plt.suptitle('Predicted end points' + sup_title_postfix)
    if show:
        safe_postfix = sup_title_postfix.replace(' ', '_').replace(',', '').replace('[', '').replace(']', '') if sup_title_postfix else ''
        _save_or_show(output_dir, f'end_points{safe_postfix}')

def plot_hidden_activations(hiddens, xy, headings, n_samples_grid, n_headings, order = None, act_min = None, act_max = None, output_dir = None):
    # show a heatmap for a hidden unit h:
    n_show = 5

    # Use Times New Roman font for the plot:
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']

    # make a plot with n_show x n_headings subplots:
    fig, ax = plt.subplots(n_show, n_headings, figsize=(15, 5))
    # plot the general title for the figure:
    if order is not None:
        plt.suptitle(f'Order {order}')
    for hidden in range(n_show):
        for h in range(n_headings):
            # select the right hidden activations:
            hiddens_heading = hiddens[h*n_samples_grid:(h+1)*n_samples_grid, :]
            hidden_heading = hiddens_heading[:,hidden]
            # take the right input positions:
            xy_heading = xy[h*n_samples_grid:(h+1)*n_samples_grid]
            # show a heatmap of the hidden unit:
            if act_min is not None and act_max is not None:
                ax[hidden, h].scatter(xy_heading[:,0], xy_heading[:,1], c=hidden_heading, cmap='coolwarm', vmin=act_min, vmax=act_max)
            else:
                ax[hidden, h].scatter(xy_heading[:,0], xy_heading[:,1], c=hidden_heading, cmap='coolwarm')
            if h == 0:
                ax[hidden, h].set_ylabel('$y$ [m]', fontsize=11)
            if hidden == 0:
                ax[hidden, h].set_title(f'$h$ = {hidden}, {np.rad2deg(headings[h])} $^\circ$', fontsize=11)
            if hidden == n_show - 1:
                ax[hidden, h].set_xlabel('$x$ [m]', fontsize=11)
            ax[hidden, h].set_aspect('equal')
    order_str = f'_order_{"_".join(map(str, order))}' if order is not None else ''
    _save_or_show(output_dir, f'hidden_activations{order_str}')

def plot_inputs(inputs, xy, headings, n_samples_grid, n_headings, output_dir = None):

    vmin = np.min(inputs)
    vmax = np.max(inputs)

    # plot all dimensions of the input in an equal number of subplots:
    n_dimensions = inputs.shape[1]
    # make a plot with n_dimensions x n_headings subplots:
    fig, ax = plt.subplots(n_dimensions, n_headings, figsize=(15, 5))
    for d in range(n_dimensions):
        for h in range(n_headings):
            # select the right input dimension:
            inputs_heading = inputs[h*n_samples_grid:(h+1)*n_samples_grid, d]
            # take the right input positions:
            xy_heading = xy[h*n_samples_grid:(h+1)*n_samples_grid]
            # show a heatmap of the input dimension:
            #  force the same color scale for all subplots:

            ax[d, h].scatter(xy_heading[:,0], xy_heading[:,1], c=inputs_heading, cmap='coolwarm', vmin=vmin, vmax=vmax)
            if h == 0:
                ax[d, h].set_ylabel('y')
            if d == 0:
                ax[d, h].set_title(f'd = {d}, {np.rad2deg(headings[h])} $^\circ$', fontsize=6)
            if d == n_dimensions - 1:
                ax[d, h].set_xlabel('x')
            ax[d, h].set_aspect('equal')
    _save_or_show(output_dir, 'inputs_heatmap')

def plot_arrows_for_weights(hiddens, model, plot_output_weights = False, parameters = None, output_dir = None):
    n_hiddens = hiddens.shape[0]

    W = np.zeros((n_hiddens, parameters['n_landmarks']*2))
    W_out = np.zeros((n_hiddens, 2))
    # get statistics for limits plots:
    for idx, h in enumerate(hiddens):
        weights, bias, weights_out = get_weights_hidden_unit(model, h)
        W[idx, :] = weights
        W_out[idx, :] = weights_out

    # get the minimum and maximum values for the weights:
    min_x = np.min(W[0::2])
    max_x = np.max(W[0::2])
    min_y = np.min(W[1::2])
    max_y = np.max(W[1::2])

    # create a plot with n_hiddens subplots:
    n_plotting = 16
    if n_hiddens < n_plotting:
        raise ValueError(f'Number of hidden units {n_hiddens} is less than the number of plotting units {n_plotting}. Please reduce n_plotting or increase n_hiddens.')
    n_rows = np.ceil(np.sqrt(n_plotting)).astype(int)
    n_cols = np.ceil(n_plotting / n_rows).astype(int)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    angles = np.zeros((n_hiddens,1))
    angles_out = np.zeros((n_hiddens,1))
    longest_inds = np.zeros((n_hiddens,1))
    for idx, h in enumerate(hiddens):
        weights, bias, weights_out = get_weights_hidden_unit(model, h)
        n_arrows =(int)(weights.shape[0] / 2)
        arrows = np.zeros((n_arrows, 2))
        for a in range(n_arrows):
            arrows[a, 0] = weights[2*a]
            arrows[a, 1] = weights[2*a + 1]
        # sort the arrows by their length:
        arrow_sizes = np.linalg.norm(arrows, axis=1)
        inds_arrows = np.argsort(arrow_sizes)
        longest_inds[idx] = inds_arrows[-1]
        arrows = arrows[inds_arrows]
        print(f'Hidden unit {h}: weights: {weights}, bias: {bias}, weights_out: {weights_out}')

        # get the angle of the longest arrow:
        angles[idx] = np.arctan2(arrows[-1, 1], arrows[-1, 0])
        angles[idx] = np.rad2deg(angles[idx])

        if idx < n_plotting:
            # plot the arrows:
            colors = plt.get_cmap('Dark2')(np.linspace(0, 1, n_arrows))
            row = idx // n_cols
            col = idx % n_cols
            for a in range(n_arrows):
                arrow = FancyArrow(0, 0, arrows[a, 0], arrows[a, 1], color=colors[a],
                                width=0.01, length_includes_head=True)
                ax[row, col].add_patch(arrow)

            if(plot_output_weights):
                arrow = FancyArrow(0, 0, weights_out[0], weights_out[1], color='red',
                                width=0.01, length_includes_head=True)
                ax[row, col].add_patch(arrow)
                angles_out[idx] = np.arctan2(weights_out[1], weights_out[0])
                angles_out[idx] = np.rad2deg(angles_out[idx])

            ax[row, col].set_xlim([min_x, max_x])
            ax[row, col].set_ylim([min_y, max_y])
            ax[idx // n_cols, idx % n_cols].set_title(f'Hidden unit {h}')
            ax[idx // n_cols, idx % n_cols].set_aspect('equal')
            ax[idx // n_cols, idx % n_cols].set_xlabel('$w_x$')
            ax[idx // n_cols, idx % n_cols].set_ylabel('$w_y$')

    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    suffix = '_with_output' if plot_output_weights else ''
    _save_or_show(output_dir, f'weight_arrows{suffix}')

    plt.figure()
    plt.hist(angles, bins=20)
    plt.xlabel('Angle of longest arrow')
    plt.ylabel('Frequency')
    plt.title('Histogram of angles of longest arrow')
    _save_or_show(output_dir, f'weight_angle_histogram{suffix}')

    plt.figure()
    cmap = plt.get_cmap('viridis')
    colors = cmap(np.linspace(0, 1, 3))
    n, bins, patches = plt.hist(longest_inds, bins=3)
    for patch, color in zip(patches, colors):
        patch.set_facecolor(color)
    plt.xlabel('Index of longest arrow')
    plt.ylabel('Frequency')
    plt.title('Histogram of index of longest arrow')
    _save_or_show(output_dir, f'weight_longest_index_histogram{suffix}')

    if(plot_output_weights):
        plt.figure()
        plt.hist(angles_out, bins=20)
        plt.xlabel('Angle of output arrow')
        plt.ylabel('Frequency')
        plt.title('Histogram of angles of output arrow')
        _save_or_show(output_dir, 'output_weight_angle_histogram')
