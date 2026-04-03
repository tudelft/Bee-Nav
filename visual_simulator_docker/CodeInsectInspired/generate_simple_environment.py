import json
with open("config/server.json", 'r') as file:
    options = json.load(file)
server = options['server']

import omni
from omni.isaac.kit import SimulationApp
import omni.replicator.core as rep
from omni.isaac.core import World
import os
import matplotlib.pyplot as plt
import csv

def generate_simple_environment(config_simple = None, simulation_app=None):

    if simulation_app is None:
        simulation_app = SimulationApp({"headless": server})

    if config_simple is None:
        # Read the config file:
        config_file = "config/config_simple_environment.json"
        with open(config_file, 'r') as file:
                config_simple = json.load(file)

    # Generate a simple usd environment for ISAAC sim, with a ground plane and a certain number of cylinders.
    usd_path_groundplane = config_simple['ground_plane']
    omni.usd.get_context().open_stage(usd_path_groundplane)
    my_world = World(stage_units_in_meters=1.0)  

    landmarks = config_simple['landmarks']
    n_landmarks = len(landmarks)
    if config_simple['unique_landmarks']:
        if n_landmarks <= 3:
             # red, blue, black (as the floor is green)
             colors = [(1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.0)]
        else:
            # create a color array for the landmarks, according to the color bar 'Set1':
            colors = plt.get_cmap('Set1')(range(n_landmarks))
            new_colors = []
            for i in range(n_landmarks):
                new_colors.append(colors[i][:3])
            colors = new_colors
    else:
        # all landmarks should be red:
        colors = [(1.0, 0.0, 0.0)] * n_landmarks

    for i, landmark in enumerate(landmarks):
        mat1 = rep.create.material_omnipbr(diffuse=tuple(colors[i]), roughness=0.5, metallic=0.0)
        cylinder = rep.create.cylinder(scale=[config_simple['radius'], config_simple['radius'], config_simple['height']],
                             position=landmark, rotation=(0, 0, 0), material=mat1)
        print(f"Landmark {i} at position {landmark}")
    
    # if debug, add a grey cylinder at the origin, and a large black one far north:
    if config_simple['home_pole']:
        mat2 = rep.create.material_omnipbr(diffuse=(0.5, 0.5, 0.5), roughness=0.5, metallic=0.0)
        cylinder = rep.create.cylinder(scale=[config_simple['radius'], config_simple['radius'], 0.2 * config_simple['height']],
                             position=[0, 0, 0], rotation=(0, 0, 0), material=mat2)
    if config_simple['north_pole']:
        mat3 = rep.create.material_omnipbr(diffuse=(0.0, 0.0, 0.0), roughness=0.5, metallic=0.0)
        cylinder = rep.create.cylinder(scale=[config_simple['radius']*10, config_simple['radius']*10, config_simple['height']*10],
                             position=[0, 200, 0], rotation=(0, 0, 0), material=mat3)

    my_world.step(render=True)
    
    omni.usd.get_context().save_as_stage(config_simple['output_name'], None)
    print(f"Environment saved at {config_simple['output_name']}")

    id_dot = config_simple['output_name'].rfind('.')
    locations_name = config_simple['output_name'][:id_dot] + "_locations.csv"
    with open(locations_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["x", "y", "z"])
        for loc in landmarks:
            writer.writerow(loc)
    print(f"Landmark locations saved at {locations_name}")

if __name__ == "__main__":
    generate_simple_environment()