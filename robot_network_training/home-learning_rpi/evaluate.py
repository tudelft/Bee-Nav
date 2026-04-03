import os
import yaml
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
import logging
from generate_label import get_home_direction, deg_to_unit_vector, get_relative_home_direction

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)

def read_data(data_path):
    return pd.read_csv(data_path)

def calculate_angular_error(prediction, ground_truth):
    delta = prediction - ground_truth
    error = np.arctan2(np.sin(delta), np.cos(delta))
    return np.abs(np.rad2deg(error))

def get_output_ground_truth(dataset, pos_x_col, pos_y_col, heading_col, output_col):
    pos_x = dataset[pos_x_col]
    pos_y = dataset[pos_y_col]
    heading = dataset[heading_col]


    for i in range(len(dataset)):
        home_direction = get_home_direction(pos_x[i], pos_y[i])
        home_vector = deg_to_unit_vector(home_direction)
        relative_home_direction = get_relative_home_direction(home_vector, np.rad2deg(heading[i])%360)
        dataset.at[i, output_col] = np.arctan2(relative_home_direction[1], relative_home_direction[0])
    return dataset


def get_distance_ground_truth(dataset, pos_x_col, pos_y_col, output_col):
    pos_x = dataset[pos_x_col]
    pos_y = dataset[pos_y_col]
    distance = np.sqrt(pos_x**2 + pos_y**2)
    dataset[output_col] = distance
    return dataset

def get_error(dataset, prediction_col, ground_truth_col, error_col, error_func):
    dataset[error_col] = error_func(dataset[prediction_col], dataset[ground_truth_col])
    return dataset

def plot_vectors(dataset, pos_x_col, pos_y_col,heading_col,  ground_truth_col, prediction_col, title, output_file):
    fig, ax = plt.subplots()
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    for i in range(len(dataset)):
        pos_x = dataset[pos_x_col][i]
        pos_y = dataset[pos_y_col][i]
        heading = dataset[heading_col][i]
        ground_truth = dataset[ground_truth_col][i]
        prediction = dataset[prediction_col][i]
        ground_truth_plot = ground_truth + heading
        prediction_plot = prediction + heading
        ax.quiver(pos_x, pos_y, np.cos(ground_truth_plot), np.sin(ground_truth_plot), angles='xy', scale_units='xy', scale=5, color='C0', width=0.005)
        ax.quiver(pos_x, pos_y, np.cos(prediction_plot), np.sin(prediction_plot), angles='xy', scale_units='xy', scale=5, color='C1', width=0.005)
    ax.plot(0, 0, 'o', markersize=5)
    ax.invert_yaxis()
    ax.legend(['Ground', 'Prediction'])
    ax.set_title(title)
    plt.savefig(output_file)
    plt.close(fig)

def plot_distance(dataset, pos_x_col, pos_y_col, distance_col, title, output_file):
    colors = ["#E3EFFB", "#5C8DC7"]
    cmap_name = 'distance_cmap'
    dist_cmap = LinearSegmentedColormap.from_list(cmap_name, colors)
    fig, ax = plt.subplots()
    norm_dist = Normalize(vmin=dataset[distance_col].min(), vmax=dataset[distance_col].max())
    distance_values = dataset[distance_col]
    colors = [custom_cmap(val, dist_cmap, norm_dist) for val in distance_values]
    ax.scatter(dataset[pos_x_col], dataset[pos_y_col], c=colors)
    sm = plt.cm.ScalarMappable(cmap=dist_cmap, norm=norm_dist)
    sm.set_array([])
    sm.set_clim(0, 10)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical')
    cbar.set_label('Distance (m)')
    ax.set_title(title)
    ax.set_xlabel(f'Position x ({pos_x_col})')
    ax.set_ylabel(f'Position y ({pos_y_col})')
    ax.grid(True)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-8, 8)
    ax.set_aspect('equal', adjustable='box')
    ax.invert_yaxis()
    plt.savefig(output_file)
    plt.close(fig)

def plot_error_histogram(dataset, error_col, title, output_file):
    fig, ax = plt.subplots()
    ax.hist(dataset[error_col], bins=20)
    ax.set_title(title)
    ax.set_xlabel('Error')
    ax.set_ylabel('Frequency')
    mean_error = np.mean(dataset[error_col])
    std_error = np.std(dataset[error_col])
    ax.axvline(mean_error, color='k', linestyle='dashed', linewidth=1)
    ax.axvline(mean_error + std_error, color='r', linestyle='dashed', linewidth=1)
    ax.axvline(mean_error - std_error, color='r', linestyle='dashed', linewidth=1)
    ax.legend([f'Mean={round(mean_error,2)}', f'Mean+Std({round(std_error,2)})', f'Mean-Std({round(std_error,2)})'])
    plt.savefig(output_file)
    plt.close(fig)
    with open(output_file.replace('.png', '.txt'), 'w') as f:
        f.write(f'Mean Error: {mean_error}\n')
        f.write(f'Std Error: {std_error}\n')

def custom_cmap(value, cmap, norm, low_val_color=(1.0, 0.0, 0.0, 1.0), threshold=0.05):
    return low_val_color if value < threshold else cmap(norm(value))

def annotate_image(dataset, prediction_col, ground_truth_col, output_prefix):
    for i in range(len(dataset)):
        filename = dataset['path_idx'][i] + '_preprocessed.jpg'
        img_path = os.path.join(data_path, filename)
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            angle_prediction = dataset[prediction_col][i]
            angle_ground_truth = dataset[ground_truth_col][i]
            line_position_prediction = 900 + int(angle_prediction * 900 / np.pi)
            img = cv2.line(img, (line_position_prediction, 0), (line_position_prediction, 1800), (255, 0, 0), 2)
            line_position_gt = 900 + int(angle_ground_truth * 900 / np.pi)
            img = cv2.line(img, (line_position_gt, 0), (line_position_gt, 1800), (0, 255, 0), 2)

            # legend
            cv2.putText(img, 'Prediction', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(img, 'Ground Truth', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imwrite(os.path.join(output_path, f'{output_prefix}{filename}'), img)

def main():
    global neural_network, output_path, data_path
    config = load_config("./config.yaml")
    data_path = config['evaluating_data_path']
    data_name = os.path.basename(data_path)
    dataset = read_data(os.path.join(data_path, 'data.csv'))
    neural_network = config['network']
    output_path = os.path.join("./evaluation_results", neural_network, data_name)
    os.makedirs(os.path.join(output_path, 'plots'), exist_ok=True)

    if config['ground_truth'] == 'odometry':
        dataset = get_output_ground_truth(dataset, 'pos_x', 'pos_y', 'heading', 'output_ground_truth_odometry')
        dataset = get_distance_ground_truth(dataset, 'pos_x', 'pos_y', 'distance_ground_truth_odometry')
        dataset = get_error(dataset, f'{neural_network}_output', 'output_ground_truth_odometry', 'angular_error', calculate_angular_error)
        dataset = get_error(dataset, f'{neural_network}_distance', 'distance_ground_truth_odometry', 'distance_error', lambda x, y: np.abs(x - y))
    elif config['ground_truth'] == 'mocap':
        dataset = get_output_ground_truth(dataset, 'pos_x_mocap', 'pos_y_mocap', 'heading_mocap', 'output_ground_truth_mocap')
        dataset = get_distance_ground_truth(dataset, 'pos_x_mocap', 'pos_y_mocap', 'distance_ground_truth_mocap')
        dataset = get_error(dataset, f'{neural_network}_output', 'output_ground_truth_mocap', 'angular_error', calculate_angular_error)
        dataset = get_error(dataset, f'{neural_network}_distance', 'distance_ground_truth_mocap', 'distance_error', lambda x, y: np.abs(x - y))
    elif config['ground_truth'] == 'manual':


        dataset = get_error(dataset, f'{neural_network}_output', 'ground_truth', 'angular_error', calculate_angular_error)

    if config['ground_truth'] == 'odometry':
        plot_vectors(dataset, 'pos_x', 'pos_y','heading','output_ground_truth_odometry', f'{neural_network}_output', 'Vector output versus ground truth', os.path.join(output_path, 'plots', 'output_odometry.png'))
        plot_distance(dataset, 'pos_x', 'pos_y', f'{neural_network}_distance', 'Predicted distance', os.path.join(output_path, 'plots', 'distance_odometry.png'))
        plot_error_histogram(dataset, 'angular_error', 'Angular Error Histogram', os.path.join(output_path, 'plots', 'angular_error_hist.png'))
        plot_error_histogram(dataset, 'distance_error', 'Distance Error Histogram', os.path.join(output_path, 'plots', 'distance_error_hist.png'))
    elif config['ground_truth'] == 'mocap':
        plot_vectors(dataset, 'pos_x_mocap', 'pos_y_mocap', 'output_ground_truth_mocap', f'{neural_network}_output', 'Vector output versus ground truth', os.path.join(output_path, 'plots', 'output_mocap.png'))
        plot_distance(dataset, 'pos_x_mocap', 'pos_y_mocap', f'{neural_network}_distance', 'Predicted distance', os.path.join(output_path, 'plots', 'distance_mocap.png'))
        plot_error_histogram(dataset, 'angular_error', 'Angular Error Histogram', os.path.join(output_path, 'plots', 'angular_error_hist.png'))
        plot_error_histogram(dataset, 'distance_error', 'Distance Error Histogram', os.path.join(output_path, 'plots', 'distance_error_hist.png'))
    elif config['ground_truth'] == 'manual':
        plot_error_histogram(dataset, 'angular_error', 'Angular Error Histogram', os.path.join(output_path, 'plots', 'angular_error_hist.png'))


    if config['annotate_image']:
        if config['ground_truth'] == 'odometry':
            os.makedirs(os.path.join(output_path, 'annotated_images_oodometry'), exist_ok=True)
            annotate_image(dataset, f'{neural_network}_output', 'output_ground_truth_odometry', 'annotated_images_oodometry/')
        elif config['ground_truth'] == 'mocap':
            os.makedirs(os.path.join(output_path, 'annotated_images_mocap'), exist_ok=True)
            annotate_image(dataset, f'{neural_network}_output', 'output_ground_truth_mocap', 'annotated_images_mocap/')
        elif config['ground_truth'] == 'manual':
            os.makedirs(os.path.join(output_path, 'annotated_images_manual'), exist_ok=True)
            annotate_image(dataset, f'{neural_network}_output', 'ground_truth', 'annotated_images_manual/')

if __name__ == "__main__":
    main()
