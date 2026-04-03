# run multiple simulation scenarios in sequence
import json
with open("config/server.json", 'r') as file:
    options = json.load(file)
server = options['server']

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": server})
import json
import generate_forest_environment
import render_dataset
import train
import virtual_homing_ANN
import os
import numpy as np
import torch
import random
import virtual_homing_snapshot

generate_environments = False # whether to generate the environments or not

seed_number = 1
np.random.seed(seed_number) # set the random seed for reproducibility
torch.random.manual_seed(seed_number) # set the random seed for reproducibility
random.seed(seed_number)

# number of experiments:
n_experiments = 10
# number of trees:
num_locations = 15

# 1. create a new environment
# 2. perform virtual homing
my_world = None

successful = []
total_distances = []
straight_line = []
radii = []
rates = []
success_per_run = []
angular_errors = []

for exp in range(0, n_experiments):

    print(f'Running experiment {exp + 1} of {n_experiments}')

    # 1. create a new environment
    config_generate_file = "config/config_forest_environment.json"
    with open(config_generate_file, 'r') as file:
        config_generate = json.load(file)
    area_bounds = config_generate['area_bounds'] # area bounds as in [-area_bounds, area_bounds]
    min_separation = config_generate['min_separation'] # minimal separation between trees.
    envparams = (num_locations, area_bounds, min_separation)
    ground_plane = config_generate['ground_plane'] # path to the grass plane USD file
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

    # 2. perform virtual homing
    config_virtual_homing_file = "config/config_virt_homing.json"
    config_homing = virtual_homing_ANN.load_config(config_virtual_homing_file)
    config_homing['map'] = map_path

    if my_world is not None:
        my_world.reset()
        my_world.clear()

    successful_runs, total_distances_traveled, straight_line_distances, all_angular_errors, radius_scales, success_rates, trajectories, successes = \
        virtual_homing_snapshot.snapshot_navigation(verbose = True, graphics = False, save_images = False, simulation_app = simulation_app, my_world = my_world, usd_path = map_path)
    
    # store the results in lists:
    successful.append(successful_runs)
    total_distances.append(list(total_distances_traveled))
    straight_line.append(list(straight_line_distances))
    radii.append(radius_scales)
    rates.append(success_rates)
    success_per_run.append(list(successes))
    angular_errors.append(list(all_angular_errors))

    results = {
        "successful": successful,
        "total_distances": total_distances,
        "straight_line": straight_line,
        "radii": radii,
        "rates": rates,
        "success_per_run": success_per_run,
        "angular_errors": angular_errors
    }
    # save the results to a json file:
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    results_filename = "results/results" + str(num_locations) + "_trees_" + "snapshot"
    with open(results_filename + ".json", "w") as f:
        json.dump(results, f, indent=4)

simulation_app.close()


