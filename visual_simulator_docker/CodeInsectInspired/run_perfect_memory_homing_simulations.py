# run multiple simulation scenarios in sequence
import json

import virtual_homing_perfect_memory
with open("config/server.json", 'r') as file:
    options = json.load(file)
server = options['server']

from omni.isaac.kit import SimulationApp
# Create the simulation app
simulation_app = SimulationApp({"headless": server})
import generate_forest_environment
import render_dataset
import os
import numpy as np
import torch
import random
import pdb

generate_environments = False # whether to generate the environments or not

seed_number = 10
np.random.seed(seed_number) # set the random seed for reproducibility
torch.random.manual_seed(seed_number) # set the random seed for reproducibility
random.seed(seed_number)

# number of experiments:
n_experiments = 10
# number of trees:
num_locations = 15

# learning flight pattern:
shape_flight = "bee" # "home_circle", "spiral", "bee", "grid", "area"
# home circle: n_positions and shape_params = [radius]
# spiral: shape_params = [n_spirals, n_points_on_spiral] + use_noisy_spiral
# bee: shape_params = [n_traverses, n_points_per_traverse, resolution] + use_noisy_spiral
# n = 5 # how many circles
# m = 150 # resolution of the circle
# b = 0.35 # magnitude away from the nest
# for a small area: n=4, m=36, b=0.1. For a bigger area: n=5, m=56, b=0.2
# grid: shape_params = [x_min, x_max, y_min, y_max, n_steps]
# area: shape_params = [x_min, x_max, y_min, y_max], + n_positions
n_positions = 1000 # only relevant for home_circle, area
use_noisy_spiral = True # only relevant for spiral and bee

# epochs should be smaller for dense datasets (50), like home_circle or area (if there are like 1000 points)
# should be larger for sparse datasets, like bee or spiral
if shape_flight == 'bee':
    learning_rate = 5e-4
    batch_size = 8
    split_ratio = 0.95
    epochs = 150 # number of epochs for training
    shape_params = [5, 150, 0.35] # bee
elif shape_flight == 'home_circle':
    epochs = 25 # number of epochs for training
    shape_params = [10] # home circle

# 1. create a new environment
# 2. generate a dataset
# 3. train a model
# 4. perform virtual homing

my_world = None

successful = []
total_distances = []
straight_line = []
angular_errors = []
distance_errors = []
radii = []
rates = []
success_per_run = []

for exp in range(n_experiments):

    # 1. create a new environment
    config_generate_file = "config/config_forest_environment.json"
    with open(config_generate_file, 'r') as file:
        config_generate = json.load(file)
    area_bounds = config_generate['area_bounds'] # area bounds as in [-area_bounds, area_bounds]
    min_separation = config_generate['min_separation'] # minimal separation between trees.
    envparams = (num_locations, area_bounds, min_separation)
    ground_plane = config_generate['ground_plane']
    props = config_generate['props'] # path to the props folder

    # The map path name is automatically generated based on the parameters and is returned by the main function:
    if generate_environments:
        map_path, my_world = generate_forest_environment.main(envparams, ground_plane, props, simulation_app=simulation_app, my_world = my_world)
    else: 
        # check if the map already exists:
        my_world = None
        map_path = os.getcwd() + "/maps/forest_" + str(num_locations) + "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area_" + str(exp) + '.usd'
        if not os.path.exists(map_path):
            # raise an error if the map does not exist: 
            raise FileNotFoundError(f"Map {map_path} does not exist. Please generate the environment first. Set generate_environments = True.")

    # 2. generate a dataset
    print("Generating dataset...")
    config_render_file = "config/config_render.json"
    config_render = render_dataset.load_config(config_render_file)
    # make sure to set the map path to the generated map:
    config_render['map'] = map_path
    output_name = os.getcwd() + "/forest/forest_" + str(num_locations) + "_trees_" + str(area_bounds) + "x" + str(area_bounds) + \
                    "_area_" + str(exp) + '_' + shape_flight
    if use_noisy_spiral:
        output_name += '_noisy'
    config_render['output_folder'] = output_name
    config_render['shape_flight'] = shape_flight
    config_render['shape_params'] = shape_params
    config_render['n_positions'] = n_positions
    config_render['only_point_north'] = False
    config_render['use_noisy_spiral'] = use_noisy_spiral
    if not os.path.exists(output_name):
        # if the output folder does not exist, create it:
        os.makedirs(output_name, exist_ok=True)
        my_world = render_dataset.generate_dataset(config_render, simulation_app=simulation_app, my_world = my_world)


    # 3. perform virtual homing
    print("Performing virtual homing...")
    
    # Automatically determine the dataset and image folder based on the render config:
    dataset_folder = config_render["output_folder"]
    image_folder = config_render["output_folder"] + "/Replicator/rgb/"

    config_virtual_homing_file = "config/config_virt_homing.json"
    config_homing = virtual_homing_perfect_memory.load_config(config_virtual_homing_file)
    config_homing['map'] = map_path
    if my_world is not None:
        my_world.reset()
        my_world.clear()
    successful_runs, total_distances_traveled, straight_line_distances, all_angular_errors, radius_scales, success_rates, trajectories, successes = \
        virtual_homing_perfect_memory.perfect_memory_navigation(config_homing, verbose = True, graphics = False, simulation_app = simulation_app, \
                                            dataset_folder = dataset_folder, image_folder = image_folder, usd_path= map_path) # pass my_world?
    
    # store the results in lists:
    successful.append(successful_runs)
    total_distances.append(list(total_distances_traveled))
    straight_line.append(list(straight_line_distances))
    angular_errors.append(all_angular_errors)
    radii.append(radius_scales)
    rates.append(success_rates)
    success_per_run.append(list(successes))

    results = {
        "successful": successful,
        "total_distances": total_distances,
        "straight_line": straight_line,
        "angular_errors": angular_errors,
        "radii": radii,
        "rates": rates,
        "success_per_run": success_per_run
    }
    # save the results to a json file:
    # check if the results folder exists:
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    results_filename = "results/results" + str(num_locations) + "_trees_" + shape_flight
    if use_noisy_spiral:
        results_filename += '_noisy'
    results_filename += "_perfect_memory"

    with open(results_filename + ".json", "w") as f:
        json.dump(results, f, indent=4)

simulation_app.close()


