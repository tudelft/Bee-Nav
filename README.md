# Efficient Robot Navigation Inspired by Honeybee Learning Flights

This repository contains all code for the paper:

> **Efficient Robot Navigation Inspired by Honeybee Learning Flights**
> [Author names — TODO]
> [Journal/Conference — TODO], 2026
> DOI: [TODO]

## Repository Structure

| Folder | Description | Paper Reference |
|--------|-------------|-----------------|
| `code_theoretical_simulation_analysis/` | Theoretical analysis and ANN simulations | Supplementary Information 1–5 |
| `path_integration_noise_simulation/` | Path integration noise modelling | Figure 2 a,b |
| `visual_simulator_docker/` | Isaac Sim 4.2 visual homing experiments (Docker) | Figures 2 c,d,e |
| `robot_network_training/` | Offline network training (laptop and Raspberry Pi) | Figure 3,4 |
| `robot_onboard/` | Onboard drone code (ROS 2 flight control + camera) | Figure 3,4 |

## Quick Start

Each folder contains its own `README.md` with setup instructions. In brief:

1. **Theoretical analysis** — `cd code_theoretical_simulation_analysis && pip install -r requirements.txt && python run_all_experiments.py`
2. **Path integration** — `cd path_integration_noise_simulation && pip install -r requirements.txt && python run_path_integration_experiments.py`
3. **Visual simulator** — Requires Docker + NVIDIA GPU. See `visual_simulator_docker/README.md` for setup.
4. **Robot network training** — `cd robot_network_training/home-learning_laptop` and follow `README.md`.
5. **Robot onboard** — Requires Raspberry Pi 4 + PX4 flight controller. See `robot_onboard/README.md`.

## External Data

- **Isaac Sim 3D assets** (~2.4 GB): Download from [TODO: insert URL] and place in `visual_simulator_docker/isaac-sim_assets/`.
- **Robot training data**: Download from the link in `robot_network_training/README.md`.

## Citation

```bibtex
@article{TODO,
  title   = {Efficient Robot Navigation Inspired by Honeybee Learning Flights},
  author  = {TODO},
  journal = {TODO},
  year    = {2026}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
