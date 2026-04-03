# run path integration experiments
import os
from datetime import datetime
import determine_LHA_odometry as dlo

# load the config_odometry.json file:
import json
with open('config_odometry.json', 'r') as f:
    config = json.load(f)

# Create a timestamped output directory for this run
timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
output_dir = os.path.join('output', timestamp)
os.makedirs(output_dir, exist_ok=True)
print(f'Saving results to: {output_dir}')

# Figure 2b: Noise ratio investigation
config['num_simulations'] = 1000
config['investigate_noise_ratios'] = True
dlo.main(config, output_dir=output_dir, experiment_name='01_noise_ratio_investigation')

# Simulating the robot's path integration noise:
print('**********************************************')
print('Simulating the robot path integration noise...')
print('**********************************************')
config['num_simulations'] = 1000
config['investigate_noise_ratios'] = False
config['behavior'] = 'outbound_robot'
config['distance_noise_std_per_meter'] = 0.10
config['yaw_rate_noise_std_per_second'] = 0.63
config['heading_measurement'] = False
dlo.main(config, output_dir=output_dir, experiment_name='02_robot_PI_noise')

# SVO-GTSAM:
print('*****************************************')
print('Simulating SVO-GTSAM integration noise...')
print('*****************************************')
config['num_simulations'] = 1000
config['investigate_noise_ratios'] = False
config['behavior'] = 'outbound_robot'
config['distance_noise_std_per_meter'] = 0.015
config['yaw_rate_noise_std_per_second'] = 0.25
config['heading_measurement'] = False
dlo.main(config, output_dir=output_dir, experiment_name='03_SVO_GTSAM')

# Stankiewicz and Webb, 2023:
print('************************************************')
print('Simulating Stankiewicz & Webb integration noise...')
print('************************************************')
config['num_simulations'] = 1000
config['investigate_noise_ratios'] = False
config['behavior'] = 'outbound_robot'
config['distance_noise_std_per_meter'] = 0.15
config['yaw_rate_noise_std_per_second'] = 5.5
config['heading_measurement'] = True
dlo.main(config, output_dir=output_dir, experiment_name='04_stankiewicz_webb')

# Ant model from Wystrach et al. 2019:
print('***************************************************')
print('Simulating Wystrach et al. ant integration noise...')
print('***************************************************')
config['num_simulations'] = 1000
config['investigate_noise_ratios'] = False
config['behavior'] = 'outbound_robot'
config['distance_noise_std_per_meter'] = 0.38
config['yaw_rate_noise_std_per_second'] = 57
config['heading_measurement'] = True
dlo.main(config, output_dir=output_dir, experiment_name='05_wystrach_ant')

# Honeybee model based on data Wang et al. 2025:
print('***************************************************')
print('Simulating Wang et al. honeybee integration noise...')
print('***************************************************')
# show drift after 2300 meters:
config['num_time_steps'] = 2000
config['num_simulations'] = 1000
config['investigate_noise_ratios'] = False
config['behavior'] = 'straight'
config['distance_noise_std_per_meter'] = 0.38
config['yaw_rate_noise_std_per_second'] = 34.5
config['heading_measurement'] = True
config['target_distance'] = [2300]
dlo.main(config, output_dir=output_dir, experiment_name='06_wang_honeybee_drift')

# determine LHA percentage:
config['num_time_steps'] = 240
config['num_simulations'] = 1000
config['investigate_noise_ratios'] = False
config['behavior'] = 'outbound_robot'
config['distance_noise_std_per_meter'] = 0.38
config['yaw_rate_noise_std_per_second'] = 34.5
config['heading_measurement'] = True
config['target_distance'] = []
dlo.main(config, output_dir=output_dir, experiment_name='07_wang_honeybee_LHA')

print(f'\nAll experiments complete. Results saved to: {output_dir}')
