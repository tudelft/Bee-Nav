# Theoretical and Simulation Analysis

This folder contains all code necessary to generate the analysis results reported in Supplementary Information 1–5.

## System Information

The code was tested on a 12th Gen Intel(R) Core(TM) i9-12900HK 2.50 GHz, with 32 GB of RAM. We used Visual Studio Code with Python 3.9.13.

The code has very few dependencies: numpy, torch, matplotlib, and json. We have included a `requirements.txt` file with the exact versions of these packages.

Please note that the PyTorch version we used was 2.1.0+cu121 (CUDA-enabled). The computer was equipped with an NVIDIA GeForce RTX 3050. However, for the theoretical and simulation analysis, we do not train networks on the GPU. Hence, if you have no GPU available, simply install a CPU version of PyTorch.

## Usage

To reproduce all results:

```bash
pip install -r requirements.txt
python run_all_experiments.py
```

At the start of `run_all_experiments.py`, you can define which experiments to run. By default, all experiments are set to `True`. Results are saved as PDF and PNG files in a timestamped `output/` directory. Each experiment takes approximately 1–5 minutes.
