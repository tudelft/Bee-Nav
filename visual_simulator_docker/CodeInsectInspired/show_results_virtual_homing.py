# show results of virtual homing experiments
import json
import os
from matplotlib import pyplot as plt
import numpy as np
import insect_utils.plot_utils

def on_resize(event):
    fig.tight_layout()  # or fig.set_constrained_layout(True)
    fig.canvas.draw()

results_file = 'results40_trees_bee_noisy_attention.json'
suffix = 'bee_noisy'
network_type = 'attention'
suffix_model = suffix + network_type

show_aggregate_results = False
show_runs_specific_experiment = True

# The number of tree locations will be used to retrieve the right map name, etc.
num_locations = 40

if show_runs_specific_experiment:
    import virtual_homing_ANN
    # specify which experiment you want to run:
    exp = 8

# Whether to show success rates as a percentage:
percentage = True
# Whether to show histograms with frequency or as a probability distribution:
probability = False

# import the results.json file:
def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# load the results file:
results = load_json_file(results_file)
successful = results['successful']
total_distances = results['total_distances']
straight_line = results['straight_line']
angular_errors = results['angular_errors']
if "distance_errors" in results:
    distance_errors = results['distance_errors']
else:
    distance_errors = None
radii = results['radii']
rates = results['rates']
if "success_per_run" in results:
    success_per_run = results["success_per_run"]
    process_success = True
else:
    process_success = False

if show_aggregate_results:
    # Result 1: Success rates and radii
    config_virt_homing = load_json_file('config/config_virt_homing.json')
    base_radius = config_virt_homing['circle_params']['radius']
    # Make a figure to plot the success rates and radii
    success_rates = np.asarray(rates)
    initial_radii = np.asarray(radii)
    # determine the mean:
    mean_success_rate = np.mean(success_rates, axis=0)
    median_success_rate = np.median(success_rates, axis=0)
    mean_radius = np.mean(initial_radii, axis=0)
    lower_bound = np.percentile(success_rates, 25, axis=0)
    upper_bound = np.percentile(success_rates, 75, axis=0)

    if percentage:
        factor = 100
    else:
        factor = 1
        
    fig = plt.figure()
    # plot the 25th and 75th percentiles as a shaded area:
    plt.fill_between(mean_radius, factor * lower_bound, factor * upper_bound, color='lightblue', alpha=0.5)
    # plot all individual lines for rates and radii in light gray:
    for i in range(len(success_rates)):
        plt.plot(initial_radii[i], factor * success_rates[i], color='lightgray')
    # plot the mean line:
    plt.plot(mean_radius, factor * median_success_rate, color='darkblue')
    # Indicate where the learning area ends:
    plt.axvline(x=1.0, color='black', linestyle='--')
    # set the y limits:
    plt.ylim(0, factor * 1.05)
    plt.xlabel('Radius factor')
    if percentage:
        plt.ylabel('Success rate [%]')
    else:
        plt.ylabel('Success rate')
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', on_resize)
    plt.show()

    # Print all median success rates with the corresponding radius:
    print("Radius factor\tMedian success rate")
    for i in range(len(mean_radius)):
        print(f"{mean_radius[i]:.2f}\t\t{median_success_rate[i]*factor:.2f}")
    
    # Print all mean success rates with the corresponding radius:
    print("\nRadius factor\tMean success rate")
    for i in range(len(mean_radius)):
        print(f"{mean_radius[i]:.2f}\t\t{mean_success_rate[i]*factor:.2f}")

    # Histogram of total homing distances vs. straight line distances
    total_distances = np.asarray(total_distances)
    straight_line = np.asarray(straight_line)
    # Select those runs with <= 1.0 radius factor:
    n_radii_inside = len(np.where(mean_radius <= 1.0)[0])
    n_radii = len(mean_radius)
    n_inds = int(n_radii_inside * (total_distances.shape[1] / n_radii))
    total_distances = total_distances[:n_inds]
    straight_line = straight_line[:n_inds]

    if process_success:
        tot_dist = []
        str_line = []
        for j in range(len(success_per_run)):
            for k in range(n_inds):
                if success_per_run[j][k] == 1:
                    tot_dist.append(total_distances[j][k])
                    str_line.append(straight_line[j][k])
        total_distances = np.asarray(tot_dist)
        straight_line = np.asarray(str_line)

    # flatten the arrays:
    total_distances = total_distances.flatten()
    straight_line = straight_line.flatten()
    total_runs = len(total_distances)
    # divide the arrays by each other:
    travel_ratios = total_distances / straight_line

    # plot the histogram:
    fig = plt.figure()
    # plot the histogram of the travel ratios, with as face color (230, 159, 0)
    face_color = (230, 159, 0)
    face_color = [x / 255 for x in face_color]
    plt.hist(travel_ratios, bins=50, facecolor=face_color, edgecolor='black', density=probability)
    plt.xlabel('Total distance / straight line distance')
    if probability:
        plt.ylabel('Probability density')
    else:
        plt.ylabel('Frequency')
    plt.tight_layout()
    fig.canvas.mpl_connect('resize_event', on_resize)
    plt.show()

if show_runs_specific_experiment:

    with open("config/server.json", 'r') as file:
        options = json.load(file)
    server = options['server']

    from omni.isaac.kit import SimulationApp
    # Create the simulation app
    simulation_app = SimulationApp({"headless": server})
    import render_dataset
    import train

    # 1. generate a grid dataset
    shape_flight = "grid" # "home_circle", "spiral", "bee", "grid", "area"
    # home circle: n_positions and shape_params = [radius]
    # spiral: shape_params = [n_spirals, n_points_on_spiral] + use_noisy_spiral
    # bee: shape_params = [n_traverses, n_points_per_traverse, resolution] + use_noisy_spiral
    # for a small area: n=4, m=36, b=0.1. For a bigger area: n=5, m=56, b=0.2
    # grid: shape_params = [x_min, x_max, y_min, y_max, n_steps]
    # area: shape_params = [x_min, x_max, y_min, y_max], + n_positions
    shape_params = [-20,20,-20,20,40]
    n_positions = shape_params[-1] * shape_params[-1] 
    use_noisy_spiral = False

    config_render_file = "config/config_render.json"
    config_render = render_dataset.load_config(config_render_file)
    # make sure to set the map path to the generated map:
    config_generate_file = "config/config_forest_environment.json"
    with open(config_generate_file, 'r') as file:
        config_generate = json.load(file)
    area_bounds = config_generate['area_bounds'] # area bounds as in [-area_bounds, area_bounds]
    map_path = os.getcwd() + "/maps/forest_" + str(num_locations) + "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area_" + str(exp) + '.usd'
    config_render['map'] = map_path
    output_name = os.getcwd() + "/forest/forest_" + str(num_locations) + "_trees_grid_" + str(exp)
    config_render['output_folder'] = output_name
    config_render['shape_flight'] = shape_flight
    config_render['shape_params'] = shape_params
    config_render['n_positions'] = n_positions
    config_render['only_point_north'] = True # We investigate the performance for a single direction
    config_render['use_noisy_spiral'] = use_noisy_spiral

    # Does the output folder already exist?
    if os.path.exists(output_name):
        # check if the folder contains the Replicator folder:
        print("Output folder already exists. Skipping dataset generation.")
        generate_dataset = False
    else:
        print("Output folder does not exist. Generating dataset.")
        generate_dataset = True

    # if the output folder does not exist, create it:
    if generate_dataset:
        os.makedirs(output_name, exist_ok=True)
        my_world = render_dataset.generate_dataset(config_render, simulation_app=simulation_app)

    # 2. test the neural network on the grid dataset:
    config_training_file = "config/config_training.json"
    config_training  = train.load_config(config_training_file)
    # Automatically determine the dataset and image folder based on the render config:
    original_dataset_folder = output_name = os.getcwd() + "/forest/forest_" + str(num_locations) + "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area_" + str(exp)
    if suffix != '':
        original_dataset_folder += '_' + suffix
    config_training["dataset_folder"] = original_dataset_folder
    config_training["image_folder"] = original_dataset_folder + "/Replicator/rgb/"
    config_training["evaluation_dataset_folder"] = config_render["output_folder"]
    config_training["evaluation_image_folder"] = config_render["output_folder"] + "/Replicator/rgb/"
    # model suffix, so all networks are saved and retrievable:
    config_training["model_suffix"] = "_" + str(num_locations) + "trees"
    if exp > 0:
        model_filename = os.path.join("models", "model" + config_training["model_suffix"] + "_" + str(exp) + '.pth')
    else:
        model_filename = os.path.join("models", "model" + config_training["model_suffix"] + '.pth')
    if suffix_model != '':
        end_name = "_" + str(exp) + '.pth'
        model_filename = model_filename.replace(end_name, '_' + suffix_model + end_name)
    config_training['model_type'] = network_type

    # Test the model:
    predictions, ground_truths, position_map, rotation_map, noisy_position_map = train.test_model(model_filename, config_training)

    # 3. Perform virtual homing:
    config_virtual_homing_file = "config/config_virt_homing.json"
    config_homing = virtual_homing_ANN.load_config(config_virtual_homing_file)
    config_homing['map'] = map_path
    config_homing['model'] = model_filename
    config_homing['circle_params'] = {
        "radius_scales": [0.5],
        "radius": 10,
        "num_points": 16
    }
    if generate_dataset:
        my_world.reset()
        my_world.clear()
    print("Starting virtual homing")
    successful_runs, total_distances_traveled, straight_line_distances, all_angular_errors, all_distance_errors, radius_scales, success_rates, trajectories, successes = \
        virtual_homing_ANN.main(config_homing, verbose = True, graphics = False, simulation_app=simulation_app, network_type = config_training["model_type"])
    
    # 4. Plot the results:
    landmark_filename = insect_utils.plot_utils.get_landmark_filename(os.getcwd() + "/maps/forest_" + str(num_locations) + \
                                        "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area_" + str(exp))
    insect_utils.plot_utils.make_nice_plot(predictions, position_map, rotation_map, ground_truths, noisy_position_map, \
                                            config = config_training, homing_positions=trajectories, successes= successes,
                                            landmark_filename = landmark_filename, append_name = 'new1_fact2', show_plot=False)

    simulation_app.close()

print('Done')
