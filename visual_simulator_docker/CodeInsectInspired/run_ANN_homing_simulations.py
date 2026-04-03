# Run homing simulations in sequence.
# Two modes:
#   "variant"  — use 5 pre-built environments (urban, warehouse, park, greenhouse, courtyard)
#   "forest"   — generate random forest environments with scattered trees
import json
with open("config/server.json", 'r') as file:
    options = json.load(file)
server = options['server']

from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": server})
import generate_forest_environment
import render_dataset
import train
import virtual_homing_ANN
import os
import numpy as np
import torch
import random

seed_number = 10
np.random.seed(seed_number)
torch.random.manual_seed(seed_number)
random.seed(seed_number)

# ═══════════════════════════════════════════════════════════════════
#  MODE SELECTION — choose "variant" or "forest"
# ═══════════════════════════════════════════════════════════════════
mode = "variant"  # "variant" or "forest"

# ── Variant mode settings ──
all_environments = ["urban", "warehouse", "park", "greenhouse", "courtyard"]

# ── Forest mode settings ──
generate_environments = True  # whether to generate new forests (forest mode only)
n_experiments = 10
num_locations = 15  # number of trees per forest

# ── Common settings ──
train_models = True
network_type = "attention"  # "simple", "compact", "attention"
# Learning flight pattern:
shape_flight = "bee"  # "home_circle", "spiral", "bee", "grid", "area"
n_positions = 1000
use_noisy_spiral = False

if shape_flight == 'bee':
    learning_rate = 5e-4
    batch_size = 8
    split_ratio = 0.95
    epochs = 150
    shape_params = [5, 150, 0.35]
elif shape_flight == 'home_circle':
    learning_rate = 5e-4
    batch_size = 8
    split_ratio = 0.95
    epochs = 25
    shape_params = [10]

# ── Crash recovery ──
crashed = False
if crashed:
    if mode == "variant":
        crashed_remaining = ["courtyard"]  # env names still to run
    else:
        crashed_remaining = [9]  # experiment indices still to run

# ═══════════════════════════════════════════════════════════════════
#  Build the experiment list depending on mode
# ═══════════════════════════════════════════════════════════════════
if mode == "variant":
    experiment_list = crashed_remaining if crashed else list(all_environments)
    results_tag = "variants"
elif mode == "forest":
    experiment_list = crashed_remaining if crashed else list(range(n_experiments))
    results_tag = str(num_locations) + "_trees"
else:
    raise ValueError(f"Unknown mode: {mode}. Use 'variant' or 'forest'.")

# ── Results filename ──
results_filename = f"results/results_{results_tag}_{shape_flight}"
if use_noisy_spiral:
    results_filename += '_noisy'
results_filename += '_' + network_type

# ── Results storage ──
if not crashed:
    successful = []
    total_distances = []
    straight_line = []
    angular_errors = []
    distance_errors = []
    radii = []
    rates = []
    success_per_run = []
    all_trajectories = []
    completed_keys = []
else:
    with open(results_filename + ".json", "r") as f:
        results = json.load(f)
    successful = results["successful"]
    total_distances = results["total_distances"]
    straight_line = results["straight_line"]
    angular_errors = results["angular_errors"]
    distance_errors = results["distance_errors"]
    radii = results["radii"]
    rates = results["rates"]
    success_per_run = results["success_per_run"]
    all_trajectories = results.get("trajectories", [])
    completed_keys = results.get("experiments", [])

# ═══════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════
my_world = None

for exp in experiment_list:

    if mode == "variant":
        env_name = exp  # exp is a string like "urban"
        exp_label = env_name
    else:
        env_name = None
        exp_label = f"forest_{exp}"

    print(f"\n{'='*60}")
    print(f"  Experiment: {exp_label}")
    print(f"{'='*60}\n")

    # ── 1. Get map path ──
    if mode == "variant":
        # Pre-built USD map, no tree generation
        map_path = os.path.join(os.getcwd(), "maps", f"env_{env_name}.usd")
        if not os.path.exists(map_path):
            raise FileNotFoundError(
                f"Map {map_path} does not exist. "
                f"Run: python generate_variant_environments.py --variant {env_name}")
        my_world = None
    else:
        # Forest mode: generate random trees or use existing
        config_generate_file = "config/config_forest_environment.json"
        with open(config_generate_file, 'r') as file:
            config_generate = json.load(file)
        area_bounds = config_generate['area_bounds']
        min_separation = config_generate['min_separation']
        envparams = (num_locations, area_bounds, min_separation)
        ground_plane = config_generate['ground_plane']
        props = config_generate['props']

        if generate_environments:
            map_path, my_world = generate_forest_environment.main(
                envparams, ground_plane, props,
                simulation_app=simulation_app, my_world=my_world)
        else:
            my_world = None
            map_path = os.path.join(
                os.getcwd(), "maps",
                f"forest_{num_locations}_trees_{area_bounds}x{area_bounds}_area_{exp}.usd")
            if not os.path.exists(map_path):
                raise FileNotFoundError(
                    f"Map {map_path} does not exist. "
                    f"Set generate_environments = True.")

    # ── 2. Generate dataset ──
    print(f"[{exp_label}] Generating dataset...")
    config_render = render_dataset.load_config("config/config_render.json")
    config_render['map'] = map_path

    if mode == "variant":
        output_name = os.path.join(
            os.getcwd(), "forest", f"{env_name}_{shape_flight}")
    else:
        output_name = os.path.join(
            os.getcwd(), "forest",
            f"forest_{num_locations}_trees_{area_bounds}x{area_bounds}_area_{exp}_{shape_flight}")
    if use_noisy_spiral:
        output_name += '_noisy'

    config_render['output_folder'] = output_name
    config_render['shape_flight'] = shape_flight
    config_render['shape_params'] = shape_params
    config_render['n_positions'] = n_positions
    config_render['only_point_north'] = False
    config_render['use_noisy_spiral'] = use_noisy_spiral
    if not os.path.exists(output_name):
        os.makedirs(output_name, exist_ok=True)
        my_world = render_dataset.generate_dataset(
            config_render, simulation_app=simulation_app, my_world=my_world)

    # ── 3. Train model ──
    print(f"[{exp_label}] Training model...")
    config_training = train.load_config("config/config_training.json")
    config_training["dataset_folder"] = config_render["output_folder"]
    config_training["image_folder"] = config_render["output_folder"] + "/Replicator/rgb/"
    config_training["evaluation_dataset_folder"] = config_render["output_folder"]
    config_training["evaluation_image_folder"] = config_render["output_folder"] + "/Replicator/rgb/"

    if mode == "variant":
        config_training["model_suffix"] = f"_{env_name}_{shape_flight}"
    else:
        config_training["model_suffix"] = f"_{num_locations}trees_{shape_flight}"
    if use_noisy_spiral:
        config_training["model_suffix"] += '_noisy'
    config_training["model_suffix"] += '_' + network_type

    config_training["epochs"] = epochs
    config_training["learning_rate"] = learning_rate
    config_training["batch_size"] = batch_size
    config_training['split_ratio'] = split_ratio
    config_training['model_type'] = network_type

    if train_models:
        model_filename = train.train_model(config_training, only_training=True)
    else:
        if mode == "variant":
            model_filename = os.path.join(
                "models", f"model_{env_name}_{shape_flight}")
        else:
            model_filename = os.path.join(
                "models", f"model_{num_locations}trees_{shape_flight}")
        if use_noisy_spiral:
            model_filename += '_noisy'
        if mode == "forest" and exp != 0:
            model_filename += '_' + network_type + f'_{exp}.pth'
        else:
            model_filename += '_' + network_type + '.pth'
        if not os.path.exists(model_filename):
            raise FileNotFoundError(
                f"Model {model_filename} not found. Set train_models = True.")

    # ── 4. Virtual homing ──
    print(f"[{exp_label}] Performing virtual homing...")
    config_homing = virtual_homing_ANN.load_config("config/config_virt_homing.json")
    config_homing['map'] = map_path
    config_homing['model'] = model_filename
    if my_world is not None:
        my_world.reset()
        my_world.clear()
    (successful_runs, total_distances_traveled, straight_line_distances,
     all_angular_errors, all_distance_errors, radius_scales, success_rates,
     trajectories, successes) = virtual_homing_ANN.main(
        config_homing, verbose=True, graphics=False,
        simulation_app=simulation_app, network_type=network_type)

    # ── Store results ──
    successful.append(successful_runs)
    total_distances.append(list(total_distances_traveled))
    straight_line.append(list(straight_line_distances))
    angular_errors.append(all_angular_errors)
    distance_errors.append(list(all_distance_errors))
    radii.append(radius_scales)
    rates.append(success_rates)
    success_per_run.append(list(successes))
    # Convert trajectories: list of runs, each run is list of (x,y,z) positions
    env_trajs = [[(float(p[0]), float(p[1]), float(p[2])) for p in traj]
                 for traj in trajectories]
    all_trajectories.append(env_trajs)
    completed_keys.append(exp_label)

    # Save after each experiment (crash resilience)
    results = {
        "mode": mode,
        "experiments": completed_keys,
        "successful": successful,
        "total_distances": total_distances,
        "straight_line": straight_line,
        "angular_errors": angular_errors,
        "distance_errors": distance_errors,
        "radii": radii,
        "rates": rates,
        "success_per_run": success_per_run,
        "trajectories": all_trajectories
    }
    if not os.path.exists("results"):
        os.makedirs("results", exist_ok=True)
    with open(results_filename + ".json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"[{exp_label}] Results saved to {results_filename}.json")

simulation_app.close()
