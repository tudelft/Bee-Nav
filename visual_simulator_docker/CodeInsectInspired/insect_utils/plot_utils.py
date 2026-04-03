import numpy as np
import matplotlib.pyplot as plt
# import wandb
import torch
import json
from insect_utils.flight_path_functions import rotate_vector_by_yaw
import csv
import os

def get_landmark_filename(dataset_folder):
    folder_elements = dataset_folder.split('/')
    if folder_elements[-1] == '':
        folder_elements.pop()  # Remove the last empty element if it exists
    experiment_name = folder_elements[-1]
    name_elements = experiment_name.split('_')
    landmark_name = name_elements[0]
    if landmark_name == 'simple':
        return './maps/' + experiment_name + '_locations.csv'
    else:
        for i in range(1, len(name_elements) - 1):
            landmark_name += '_'
            landmark_name += name_elements[i]
        landmark_name += '_locations_'
        landmark_name += name_elements[-1]
        landmark_file = './maps/' + landmark_name + '.csv'
        return landmark_file

def make_nice_plot(predicted_labels, position_map, rotation_map = None, ground_truths = None, noisy_position_map = None, \
                   config = None, append_name = '', plot_quiver = True, homing_positions = None, successes = None, area_limit=20, \
                    landmark_filename = None, show_plot = True):
    
    if config == None:
        json.load(open('config/config_training.json'))

    if type(position_map) is dict:
        positions = np.array(list(position_map.values()))
    else:
        positions = position_map
    
    if rotation_map is not None:
        if type(rotation_map) is dict:
            rotations = np.array(list(rotation_map.values()))
        else:
            rotations = rotation_map

        # rotate all predictions:
        rotate = False
        if np.sum(abs(rotations[:,2])) > 0.1:
            rotate = True
        if rotate:
            for i in range(len(predicted_labels)):
                predicted_labels[i] = rotate_vector_by_yaw(predicted_labels[i], rotations[i,2])
                ground_truths[i] = rotate_vector_by_yaw(ground_truths[i], rotations[i,2])

    if noisy_position_map is not None:
        if type(noisy_position_map) is dict:
            noisy_positions = np.array(list(noisy_position_map.values()))
        else:
            noisy_positions = noisy_position_map
    else:
        noisy_positions = None

    #positions = positions[:25]
    pos_min_x, pos_max_x = positions[:, 0].min(), positions[:, 0].max()
    pos_min_y, pos_max_y = positions[:, 1].min(), positions[:, 1].max()
    pos_min_z, pos_max_z = positions[:, 2].min(), positions[:, 2].max()

    if abs(pos_min_x+10.0) < 1e-5:
        SIMPLE_ENVIRONMENT = True
    else:
        SIMPLE_ENVIRONMENT = False

    # Virtual home position
    config_render = json.load(open("config/config_render.json"))
    virtual_home_position = config_render['home_position'] 

    if not plot_quiver:
        # Setup figure for plotting
        fig = plt.figure()
        ax = fig.add_subplot(111)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch


        legend_elements = [
            Line2D([0], [0], color='green', lw=4, label='Ground Truth'),
            Line2D([0], [0], color='blue', lw=4, label='Prediction', alpha=0.5),
            Line2D([0], [0], marker='x', color='red', markersize=10, lw=0, label='Virtual Home'),
            Line2D([0], [0], marker='o', color='black', markersize=10, lw=0, label='Pic position noisy')
        ]

        arrow_scale = 0.5  
        head_width = 0.15    
        head_length = 0.15 

        # Plot the vectors at their positions
        for i, (position, gt_vector, pred_vector) in enumerate(zip(positions, ground_truths, predicted_labels)):
            start_x, start_z = position[0], position[1]  # Extract the X and Z positions
            

            # Ground truth vector
            gt_end_x = start_x + arrow_scale * gt_vector[0]
            gt_end_z = start_z + arrow_scale * gt_vector[1]
            ax.arrow(start_x, start_z, gt_end_x - start_x, gt_end_z - start_z,
                    head_width=head_width, head_length=head_length, fc='green', ec='green')

            # Predicted vector
            pred_end_x = start_x + arrow_scale * pred_vector[0]
            pred_end_z = start_z + arrow_scale * pred_vector[1]
            ax.arrow(start_x, start_z, pred_end_x - start_x, pred_end_z - start_z,
                    head_width=head_width, head_length=head_length, fc='blue', ec='blue', alpha=0.5, label='Prediction' if i == 1 else None)


        home_x, home_z = virtual_home_position[0], virtual_home_position[1]
        ax.scatter(home_x, home_z, c='red', marker='x', s=100, label='Virtual Home')
        if noisy_positions is not None:
            ax.scatter(noisy_positions[:,0], noisy_positions[:,1], c ='black', marker='o')
        ax.set_xlim([pos_min_x - 2, pos_max_x + 2])
        ax.set_ylim([pos_min_z - 2, pos_max_z + 2])

        ax.set_xlabel('X position')
        ax.set_ylabel('Z position')
        ax.set_title('Comparison of Ground Truths and Predictions')
        ax.legend(handles=legend_elements, loc='upper right')
        ax.grid(True)
        plt.show()
        figure_filename = './plot_predictions_evaluation_set' + append_name + '.png'
        fig.savefig(figure_filename)
        if wandb.run is not None:
            wandb.log({"plot": wandb.Image(figure_filename)})
    else:
        if landmark_filename == None:
            landmark_filename = get_landmark_filename(config['dataset_folder'])

        if os.path.exists(landmark_filename):
            with open(landmark_filename, mode='r') as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # Skip header row
                X = []
                Y = []
                Z = []
                for row in reader:
                    x, y, z = row
                    X.append(float(x))
                    Y.append(float(y))
                    Z.append(float(z))
                x = np.asarray(X)
                y = np.asarray(Y)
                z = np.asarray(Z)
                n_landmarks = len(X)
                landmark_positions = np.zeros([n_landmarks, 3])
                landmark_positions[:, 0] = x
                landmark_positions[:, 1] = y
                landmark_positions[:, 2] = z
        else:
            landmark_positions = None

        # Calculate the MSE:
        if ground_truths is not None:
            mse = np.mean(np.sum((ground_truths - predicted_labels) ** 2, axis=1))
            print(f"Mean Squared Error: {mse}")

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.quiver(positions[:, 0], positions[:, 1], predicted_labels[:, 0], predicted_labels[:, 1], color='blue', alpha=0.5)

        # plot the home position with a red cross:
        ax.scatter(virtual_home_position[0], virtual_home_position[1], c='red', marker='*', s=100)
        if ground_truths is not None:
            # add the MSE as text on the top right:
            mse = np.mean(np.sum((ground_truths - predicted_labels) ** 2, axis=1))
            # ax.text(0.95, 0.95, f"MSE: {mse:.2f}", ha='right', va='top', transform=ax.transAxes)
        # plot the landmarks:
        if landmark_positions is not None:

            for l in range(len(landmark_positions)):
                # create a circle artist:
                # make the color 50% transparent:
                if SIMPLE_ENVIRONMENT:
                    colors = ['lightblue', 'lightgreen', 'lightcoral']
                    # if config['model_suffix'] contains the word non_unique, plot all in lightblue:
                    if 'non_unique' in config['model_suffix']:
                        circle = plt.Circle((landmark_positions[l, 0], landmark_positions[l, 1]), 1.0, color='lightblue', alpha=0.5, fill=True)
                    else:
                        circle = plt.Circle((landmark_positions[l, 0], landmark_positions[l, 1]), 1.0, color=colors[l % len(colors)], alpha=0.5, fill=True)
                else:
                    # make the radius 2.0
                    circle = plt.Circle((landmark_positions[l, 0], landmark_positions[l, 1]), 2.0, color='lightgreen', alpha=0.5, fill=True)
                ax.add_artist(circle)

        # plot the different trajectories in homing_positions:
        if homing_positions is not None:
            for i in range(len(homing_positions)):
                hp = np.asarray(homing_positions[i])
                if successes is not None:
                    if successes[i]:
                        ax.plot(hp[:,0], hp[:,1], '-', color='green', linewidth=2)
                        # plot an arrow from the first to the second position:
                        # ax.arrow(hp[0,0], hp[0,1], hp[1,0] - hp[0,0], hp[1,1] - hp[0,1], head_width=0.5, head_length=0.5, fc='green', ec='green')
                        # plot an arrow from the penultimate to the last position:
                        ax.arrow(hp[-2,0], hp[-2,1], hp[-1,0] - hp[-2,0], hp[-1,1] - hp[-2,1], head_width=0.5, head_length=0.5, fc='green', ec='green')
                    else:
                        ax.plot(hp[:,0], hp[:,1], '--', color='red', linewidth=2)
                        # plot an arrow from the first to the second position:
                        # ax.arrow(hp[0,0], hp[0,1], hp[1,0] - hp[0,0], hp[1,1] - hp[0,1], head_width=0.5, head_length=0.5, fc='red', ec='red')
                        # plot an arrow from the penultimate to the last position:
                        ax.arrow(hp[-2,0], hp[-2,1], hp[-1,0] - hp[-2,0], hp[-1,1] - hp[-2,1], head_width=0.5, head_length=0.5, fc='red', ec='red')
                else:
                    ax.plot(hp[:,0], hp[:,1], color='black', linewidth=2)
        
        #if config_render['shape_flight'] == 'home_circle':
        if not SIMPLE_ENVIRONMENT:
            learning_radius = 10 # config_render['shape_params'][0]
            # plot a circle with the learning radius:
            # make it linewidth 2 and with linestyle ':':
            circle = plt.Circle((virtual_home_position[0], virtual_home_position[1]), learning_radius, color='orange', fill=False, linestyle=':', linewidth=2)
            ax.add_artist(circle)
        # set the axis limits to the area limit:
        ax.set_xlim([pos_min_x, pos_max_x])
        ax.set_ylim([pos_min_y, pos_max_y])
        ax.set_xlabel('X position')
        ax.set_ylabel('Y position')
        ax.set_title('Predictions')
        # make the axes equal size:
        ax.set_aspect('equal', adjustable='box')

        if show_plot:
            plt.show()
        
        figure_filename = './plot_predictions_quiver' + append_name + '.png'
        fig.savefig(figure_filename)

        if wandb.run is not None:
            wandb.log({"plot simple": wandb.Image(figure_filename)})

        # figure_filename = './plot_predictions_quiver' + append_name + '.pdf'
        # fig.savefig(figure_filename)

        figure_filename = './plot_predictions_quiver' + append_name + '.svg'
        fig.savefig(figure_filename)


