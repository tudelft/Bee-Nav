# Compare the results of two different multi-run experiments
import json
import os
import matplotlib.pyplot as plt
import numpy as np

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)

# Set whether to compare different approaches or different network architectures
compare_approaches = True

if compare_approaches:
    n_trees = 40
    condition_names = ['proposed', 'snapshot', 'matching']
    results_files = [f'results{n_trees}_trees_bee_noisy_attention.json',
                    f'results{n_trees}_trees_snapshot.json',
                    f'results{n_trees}_trees_bee_noisy_perfect_memory.json']
    dark_colors = ['darkblue', 'darkred', 'darkgreen']
    light_colors = ['lightblue', 'lightcoral', 'lightgreen']
    line_styles = ['-', '--', '-.']
    n_conditions = len(condition_names)
else:
    # compare networks:
    condition_names = ['attention', 'compact', 'simple']
    results_files = ['results40_trees_bee_noisy_attention.json',
                     'results40_trees_bee_noisy_compact.json',
                     'results40_trees_bee_noisy_simple.json']
    dark_colors = ['darkblue', 'darkred', 'darkgreen']
    light_colors = ['lightblue', 'lightcoral', 'lightgreen']
    line_styles = ['-', '--', '-.']
    n_conditions = len(condition_names)

# Load some other parameters:
config_virt_homing = load_json_file('config/config_virt_homing.json')
base_radius = config_virt_homing['circle_params']['radius']

# Whether to show success rates as a percentage:
percentage = True
# Whether to show histograms with frequency or as a probability distribution:
probability = False

# Result1 : Success rates and radii
plt.figure()

for cond in range(n_conditions - 1, -1, -1):  # Iterate from the back to the front
    results = load_json_file(results_files[cond])
    successful = results['successful']
    total_distances = results['total_distances']
    straight_line = results['straight_line']
    radii = results['radii']
    rates = results['rates']
    if "success_per_run" in results:
        success_per_run = results["success_per_run"]
        process_success = True
    else:
        process_success = False

    # Plot the success rates and radii
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
    
    # plot the 25th and 75th percentiles as a shaded area:
    # use light red for the fill color:
    plt.fill_between(mean_radius, factor * lower_bound, factor * upper_bound, color=light_colors[cond], alpha=0.5)
    # plot all individual lines for rates and radii in light gray:
    for i in range(len(success_rates)):
        plt.plot(initial_radii[i], factor * success_rates[i], color=light_colors[cond])
    # plot the mean line:
    # use dark red:
    plt.plot(mean_radius, factor * median_success_rate, color=dark_colors[cond], linestyle=line_styles[cond], label=condition_names[cond])
    # Indicate where the learning area ends:
    if cond == 0:
        plt.axvline(x=1.0, color='black', linestyle='--')

plt.legend()
# set the y limits:
plt.ylim(0,  factor * 1.02)
plt.xlabel('Radius factor')
if percentage:
    plt.ylabel('Success rate [%]')
else:
    plt.ylabel('Success rate')
plt.show()