import mushroom_body.run_mb_homing 
import render_dataset
import json
with open("config/server.json", 'r') as file:
    options = json.load(file)
server = options['server']

from omni.isaac.kit import SimulationApp
# Create the simulation app
simulation_app = SimulationApp({"headless": server})
my_world = None

generate_dataset = True


# (1) load the render config, and change parameter settings to generate training images for an MBON experiment
# load config/config_render.json
with open('config/config_render.json', 'r') as f:
    render_config = json.load(f)
# change parameters as needed
name_map = 'forest_40_trees_50x50_area_0'
render_config['map'] = '/isaac-sim/CodeInsectInspired/maps/' + name_map + '.usd'
render_config['shape_params'] = [10, 560, 0.1]
render_config['output_folder'] = '/isaac-sim/CodeInsectInspired/cars/' + name_map + '_1000_mushroom_norotation_bee_10_560_01'
if generate_dataset:
    render_dataset.generate_dataset(render_config, simulation_app=simulation_app, my_world = my_world)

# (2) run the homing simulation with the MB model with run_mb_homing.py
# Load Config
with open('./mushroom_body/config/config_mb_homing.json', 'r') as f:
    config = json.load(f)

# get the name from render_config['map']
# name_map = render_config['map'].split('/')[-1].split('.')[0]
config['homing']['usd_path'] = render_config['map']
config['paths']['rgb_folder'] = render_config['output_folder'] + '/Replicator/rgb'
config['paths']['csv_path'] = render_config['output_folder'] + '/dataset_navigation.csv'
config['homing']['output_folder'] = 'mushroom_body/results_homing_pipeline/' + name_map + '_' + render_config['shape_flight']
# Run
mushroom_body.run_mb_homing.run_homing_simulation(config, simulation_app = simulation_app)