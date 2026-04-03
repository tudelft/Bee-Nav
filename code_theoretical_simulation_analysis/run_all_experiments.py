# This is the main script to run all experiments for reproducing the results in the
# supplementary information.

import json
import os
from datetime import datetime
from insect_navigation import insect_navigation
from tortuosity import run_tortuosity_analysis
import torch
import numpy as np

# Create timestamped output directory
timestamp = datetime.now().strftime('%Y-%m-%d_%H%M%S')
output_dir = os.path.join('output', timestamp)
os.makedirs(output_dir, exist_ok=True)

# Define the experiments to run, per section in the supplementary information document
SI1 = True
SI2 = True
distance_from_landmark = True
end_points = True
two_landmarks = True
SI3 = True
SI4 = True
tortuosity = True

def run_experiment(param_file, experiment_name):
    """Helper to load parameters and run an experiment with output directory."""
    with open(param_file, "r") as file:
        parameters = json.load(file)
    exp_dir = os.path.join(output_dir, experiment_name)
    insect_navigation(parameters, output_dir=exp_dir)

if SI1:
    # Experiment for SI-1
    run_experiment("parameters_SI1.json", "01_SI1")
    print("Experiment SI-1 completed.")

if SI2:
    # Experiments for SI-2
    if distance_from_landmark:
        run_experiment("parameters_SI2_close.json", "02_SI2_close")
        run_experiment("parameters_SI2_edge.json", "03_SI2_edge")
        run_experiment("parameters_SI2_edge_theory.json", "04_SI2_edge_theory")

    if end_points:
        run_experiment("parameters_SI2_close_perceptron.json", "05_SI2_close_perceptron")
        run_experiment("parameters_SI2_edge_perceptron.json", "06_SI2_edge_perceptron")
        run_experiment("parameters_SI2_close_perceptron_circle.json", "07_SI2_close_perceptron_circle")

    if two_landmarks:
        run_experiment("parameters_SI2_two_landmarks.json", "08_SI2_two_landmarks")

    print("Experiment SI-2 completed.")

if SI3:
    # Experiments for SI-3
    run_experiment("parameters_SI3_two_landmarks.json", "09_SI3_two_landmarks")
    run_experiment("parameters_SI3_three_landmarks_equilateral.json", "10_SI3_three_landmarks_equilateral")
    run_experiment("parameters_SI3_three_landmarks_isosceles.json", "11_SI3_three_landmarks_isosceles")
    run_experiment("parameters_SI3_three_landmarks_scalene.json", "12_SI3_three_landmarks_scalene")

    torch.manual_seed(2024)
    np.random.seed(2024)
    run_experiment("parameters_SI3_three_landmarks_isosceles_analysis.json", "13_SI3_three_landmarks_isosceles_analysis")

    print("Experiment SI-3 completed.")

if SI4:
    torch.manual_seed(0)
    np.random.seed(0)
    run_experiment("parameters_SI4_bee_flight_noise.json", "14_SI4_bee_flight_noise")
    print("Experiment SI-4 completed.")

if tortuosity:
    run_tortuosity_analysis(output_dir=os.path.join(output_dir, "15_tortuosity"))
    print("Tortuosity analysis completed.")

print(f"\nAll outputs saved to: {output_dir}")
