# Path Integration Noise Simulation

This folder contains the code for simulating path integration noise in drone flights. The simulations model a drone flying outwards and then attempting to return home using path integration with noisy sensors.

## Files

- `run_path_integration_experiments.py`: The main script to reproduce the path integration experiments and generate data for figures (e.g., Figure 2a, 2b). It runs different simulation scenarios including:
    - Raw Figure 2a & 2b simulations.
    - Robot path integration noise simulation.
    - SVO-GTSAM integration noise simulation.
    - Stankiewicz and Webb (2023) integration noise simulation.
- `determine_LHA_odometry.py`: The core library file containing the simulation logic. It defines the `simulate_outbound` function and the main execution loop for running Monte Carlo simulations of the path integration process. It handles different flight behaviors (e.g., 'random_walk', 'square', 'outbound_robot') and noise models.
- `config_odometry.json`: A JSON configuration file defining the default parameters for the simulations, such as time steps, velocity, noise levels, and behavior settings.
- `requirements.txt`: A list of Python package dependencies required to run the code.

## Usage

1.  **Install Dependencies**:
    Ensure you have Python installed, then install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run Experiments**:
    Execute the main experiment script to run the simulations:
    ```bash
    python run_path_integration_experiments.py
    ```
    This script will run various configurations as defined in the code and print the progress/results to the console. It may also generate plots depending on the configuration in `determine_LHA_odometry.py`.

## Configuration

You can modify `config_odometry.json` to change default simulation parameters. However, `run_path_integration_experiments.py` overrides specific parameters for different experimental runs programmatically.
