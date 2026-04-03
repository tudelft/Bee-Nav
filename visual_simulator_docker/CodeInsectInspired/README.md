# Insect-Inspired Navigation Framework for Isaac Sim

This repository provides a simulation framework built around **Isaac Sim** to facilitate research on insect-inspired navigation techniques for autonomous drones. The framework supports:

- Rapid dataset generation.
- Creation of outdoor virtual environments.
- Simulation of the homing process using virtual cameras and drones.
- Analysis of neural network performance for visual navigation tasks.

The code was tested on a 12th Gen Intel(R) Core(TM) i9-12900HK 2.50 GHz, with 32 GB of RAM, under Ubuntu 22.04. We used Visual Studio Code with Python 3.9.13. Please note that the PyTorch version we used was 2.1.0+cu121, i.e., a CUDA enabled one. The computer was equipped with an NVIDIA GeForce RTX 3050. 

============

## Installing NVIDIA Isaac Sim

![IsaacSim 4.2.0](https://img.shields.io/badge/IsaacSim-4.2.0-brightgreen.svg)  
[Isaac Sim 4.2.0](https://developer.nvidia.com/isaac-sim)

![Ubuntu 22.04](https://img.shields.io/badge/Ubuntu-22.04LTS-brightgreen.svg)  
[Ubuntu 22.04](https://releases.ubuntu.com/22.04/)

To install **Isaac Sim** on Linux, download the [Omniverse AppImage here](https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage) or run the following command in the terminal:

    wget https://install.launcher.omniverse.nvidia.com/installers/omniverse-launcher-linux.AppImage


Configuring the environment variables
-------------------------------------

NVIDIA provides Isaac Sim with its own Python interpreter along with some basic extensions such as numpy and pytorch. In
order for the Pegasus Simulator to work, we require the user to use the same python environment when starting a simulation
from python scripts. As such, we recommend setting up a few custom environment variables to make this process simpler.

## Configuring the Environment Variables

NVIDIA provides Isaac Sim with its own Python interpreter along with some basic extensions such as `numpy` and `pytorch`. For the simulator to work, the same Python environment provided by Isaac Sim must be used when starting simulations with Python scripts.

### Steps to Configure Environment Variables

Locate the **Isaac Sim installation folder**. Typically, on Linux, this folder can be found under:  
`${HOME}/.local/share/ov/pkg/isaac_sim-*`  
Replace `*` with the correct version number (e.g., `4.2.0`).

Add the following lines to your `~/.bashrc` or `~/.zshrc` file:  
`export ISAACSIM_PATH="${HOME}/.local/share/ov/pkg/isaac_sim-4.2.0"`  
`alias ISAACSIM_PYTHON="${ISAACSIM_PATH}/python.sh"`  
`alias ISAACSIM="${ISAACSIM_PATH}/isaac-sim.sh"`

### Reference Terminology

In this documentation:  
- **ISAACSIM_PATH** refers to the Isaac Sim root directory.  
- **ISAACSIM_PYTHON** refers to the provided Python interpreter.  
- **ISAACSIM** refers to the simulator application.  

## Running the experiments


### Ratio learned homing area with respect to total flight area
In order to run the path integration experiments with a Gaussian noise model, run the file `run_path_integration_experiments.py`. It will generate the results shown in Figure 2a and 2b, and simulate the noise settings used to model the robot, SVO-GTSAM, the bio-inspired odometry method of Stankiewicz and Webb, ants, and honeybees (printing drift and LHA properties in the terminal).

### Visual homing experiments
In order to replicate the aggregate results for the visual homing experiments, run the Python file: `run_ANN_homing_simulations.py`. The `config/server.json` file has a setting `server` that should be set to true when running the simulator headless. 

The script will perform ten experiments, with the following steps:
- Generate an environment.
- Generate a training dataset in the environment.
- Train the neural network on the training set.
- Use the neural network for visual homing, starting at different angles and distances from the home location.

After running the experiments, the results can be shown with the help of the `show_results_virtual_homing.py` script. Please adapt the `results_file` variable to the appropriate name before running. 

Remark: These experiments may take ~10 hours to complete.

### Snapshot-based navigation experiments
Likewise, you can run the snapshot-based navigation experiments by running the script: `run_snapshot_simulations.py`. Results can be shown with the `show_results_snapshot_simulations.py` script. If you want to run snapshot in the same environments as the ANN, keep `generate_environments = False`.

Remark: These experiments may take ~5 hours to complete.

### Perfect-memory experiments
You can run the perfect memory experimetns by running the script: `run_perfect_memory_homing_simulations.py`. Also this may take a long time.

Plotting the results of all strategies in the same figure can be done with `compare_multirun_experiments.py`. Also there, please change the result file names. This result is shown in Figure 2d in the article. If you want to plot example trajectories for a given environment, run `show_results_virtual_homing.py` with `show_runs_specific_experiment=True` (Figure 2e).

### Simple experiments  
In order to run the experiments that use simple geometric environments for verifying the theoretical and simulation analysis, run the script `run_simple_experiments.py`. Set experiments to `True` or `False` in the `perform_experiments` variable. The experiments take ~15 minutes each.

### MNIST experiments
In the article, we run the different networks used for visual homing on the basic MNIST benchmark dataset. This gives insight into their representational capacity and the effectivity of their learning rules. To reproduce these experiments, 


