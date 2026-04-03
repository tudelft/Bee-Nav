# Tune parameters for the flight pattern:
from insect_utils.flight_path_functions import generate_bee_path, add_noise_to_path
from matplotlib import pyplot as plt
import numpy as np
import json

# for small learning area: n=4, m=36, b=0.1
    # for bigger learning area: n=5, m=56, b=0.2

n = 5 # how many circles
m = 150 # resolution of the circle
b = 0.35 # magnitude away from the nest
path = generate_bee_path(n, m, b, [0,0,1.5])
path = np.asarray(path)
n_points = path.shape[0]

# load the config/config_render.json file:
# old values: "noise_params": [0.3, 4.6, 0.054, 0.01, 2, 0.025]
with open('config/config_render.json') as f:
    config_render  = json.load(f)
noisy_path = add_noise_to_path(path, config_render['noise_params'])
noisy_path = np.asarray(noisy_path)

min_x = np.min(noisy_path[:,0])
min_y = np.min(noisy_path[:,1])
max_x = np.max(noisy_path[:,0])
max_y = np.max(noisy_path[:,1])

print(f'There are {n_points} points, limits = [{min_x}, {max_x}, {min_y}, {max_y}]')

plt.figure()
plt.plot(path[:,0], path[:,1], 'o', color='green')
plt.plot(noisy_path[:,0], noisy_path[:,1], 'x', color='red')
plt.xlabel('X')
plt.ylabel('Y')
# make the axes equal:
plt.axis('equal')
plt.show()

print('Done')
