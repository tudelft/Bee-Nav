from omni.isaac.kit import SimulationApp
import omni
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
import threading
from mpl_toolkits.mplot3d import Axes3D
import csv
import json
import os
import random
import math

def is_far_enough(x, y, points, min_distance):
    """Check if the new point (x, y) is at least min_distance away from all existing points."""
    for px, py, pz in points:
        if math.sqrt((px - x)**2 + (py - y)**2) < min_distance:
            return False
    return True

def generate_random_locations(num_points, bounds, min_distance):
    points = []
    i = 0
    while len(points) < num_points and i < 10000:  # Limit iterations to avoid infinite loop
        x = random.randint(bounds[0], bounds[1])
        y = random.randint(bounds[0], bounds[1])
        z = 0.01
        if is_far_enough(x, y, points, min_distance):
            points.append((x, y, z))
        i += 1
    if len(points) < num_points:
        print(f"Warning: Only generated {len(points)} points out of {num_points} requested. Increase iterations or reduce min_distance.")
    return points

def plot_locations(locations):
    x_coords = [x for x, y, z in locations]
    y_coords = [y for x, y, z in locations]

    plt.figure(figsize=(10, 10))
    plt.scatter(x_coords, y_coords, s=10, c='blue', alpha=0.5)
    plt.title('Generated Locations')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

def main(envparams, usdpath, props, home_pole=None, simulation_app=None, my_world = None, plot_graphs = False):

    if simulation_app is None:        
        with open("config/server.json", 'r') as file:
            options = json.load(file)
        server = options['server']
        simulation_app = SimulationApp({"headless": server})
    
    from omni.isaac.core.prims import XFormPrim
    from omni.isaac.core.objects import DynamicCuboid
    from omni.isaac.sensor import Camera
    from omni.isaac.core import World
    import omni.isaac.core.utils.numpy.rotations as rot_utils
    from omni.isaac.core.utils.stage import add_reference_to_stage
    import omni.replicator.core as rep

    # create a new stage:
    omni.usd.get_context().new_stage()

    num_locations, area_bounds, min_separation = envparams # (7000,750,10)
    full_area_bounds = (-area_bounds, area_bounds)
    print(f"Generating {num_locations} locations in area bounds {full_area_bounds} with minimum separation {min_separation}")
    
    # generate 10 locations between the minimum and maximum area bounds with the minimum separation:
    num_locations = 25
    x_trees = np.linspace(full_area_bounds[0], full_area_bounds[1], int(np.sqrt(num_locations)))
    y_trees = np.linspace(full_area_bounds[0], full_area_bounds[1], int(np.sqrt(num_locations)))
    locations = []
    for x in x_trees:
        for y in y_trees:
            locations.append((x, y, 0.01))
    
    print(f"Generated {len(locations)} locations")
    if plot_graphs:
        plot_locations(locations)

    WRITE_THREADS = 12
    QUEUE_SIZE = 50

    rep.settings.carb_settings("/omni/replicator/backend/writeThreads", WRITE_THREADS)
    rep.settings.carb_settings("/omni/replicator/backend/queueSize", QUEUE_SIZE)
    PROPS = props   #Folder with props 

    # make a white ground plane:
    mat = rep.create.material_omnipbr(diffuse=(1.0, 1.0, 1.0), roughness=0.5, metallic=0.0)
    ground = rep.create.plane(scale=[area_bounds*2, area_bounds*2, 0.01],
                             position=[0, 0, -0.01], rotation=(0, 0, 0), material=mat)
    

     
    # usd_path_groundplane = usdpath
    # omni.usd.get_context().open_stage(usd_path_groundplane)

    my_world = World(stage_units_in_meters=1.0)  

    if home_pole is not None:
        mat2 = rep.create.material_omnipbr(diffuse=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
        cylinder = rep.create.cylinder(scale=[home_pole['radius'], home_pole['radius'], home_pole['height']],
                             position=[0, 0, 0], rotation=(0, 0, 0), material=mat2)

     # Explicitly instantiate and place props
    prop_files = rep.utils.get_usd_files(PROPS)
    n_props = len(prop_files)
    print(f"Number of props available: {n_props}")

    for i, location in enumerate(locations):
        j = np.random.randint(0, n_props)
        prop_file = prop_files[j]  # Pick a random prop from the list
        prop_prim = add_reference_to_stage(prop_file, f"/World/Prop_{i}")
        xform = XFormPrim(prim_path=f"/World/Prop_{i}", position=location, scale=(0.01, 0.01, 0.01))
        

    my_world.step(render=True)
    current_path = os.getcwd()

    scene_name = "./maps/show_" + str(num_locations) + "_trees_" + str(area_bounds) + "x" + str(area_bounds) +"_area"
    locations_name = scene_name + "_locations"
    # determine how many usd files with that name already exist:
    map_dir = os.path.join(current_path, "maps")
    short_scene_name = scene_name.split("/")[-1]
    existing_files = [f for f in os.listdir(map_dir) if f.startswith(short_scene_name) and f.endswith('.usd')]
    # add an underscode and a number to the name:
    n_existing_files = len(existing_files)
    if n_existing_files > 0:
        scene_name = scene_name + "_" + str(n_existing_files)
        locations_name = locations_name + "_" + str(n_existing_files)
    else:
        scene_name = scene_name + "_0"
        locations_name = locations_name + "_0"
    
    with open(locations_name + ".csv", mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        for loc in locations:
            writer.writerow(loc)
    print(f"Locations saved at {locations_name}")
    scene_name = scene_name + ".usd"
    output_path = os.path.join(current_path, scene_name)
    omni.usd.get_context().save_as_stage(output_path, None)
    print(f"Environment saved at {output_path}")
    return output_path, my_world

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate a random environment")
    
    # default parameters for generating the forest - can be overwritten with the arguments:
    # load the parameters from a json file:
    config_file = "config/config_forest_environment.json"
    with open(config_file, 'r') as file:
        config = json.load(file)
    num_locations = config['num_locations'] # number of trees
    area_bounds = config['area_bounds'] # area bounds as in [-area_bounds, area_bounds]
    min_separation = config['min_separation'] # minimal separation between trees.
    ground_plane = config['ground_plane'] # path to the grass plane USD file
    props = config['props'] # path to the props folder
    if config['home_pole']:
        home_pole = {
            'radius': config['home_pole_radius'],
            'height': config['home_pole_height']
        }
    else:
        home_pole = None

    parser.add_argument("--envparams", type=tuple, default=(num_locations,area_bounds,min_separation), help="(Number of locations, area bounds, min separation)")
    parser.add_argument("--usdpath", type=str, default=ground_plane, help="Path to the USD file")
    parser.add_argument("--props", type=str, default=props, help="Path to the props folder")  # omniverse://localhost/Library/vegetation/Trees
    
    args = parser.parse_args()

    main(args.envparams, args.usdpath, args.props, home_pole=home_pole)