# 1. Use the exact version of Isaac Sim you used (4.2.0)
FROM nvcr.io/nvidia/isaac-sim:4.2.0

# 2. Set the working directory
WORKDIR /isaac-sim

# 3. Install any extra python libraries your experiment needs
# (Isaac Sim's python is accessed via ./python.sh)
RUN ./python.sh -m pip install scipy matplotlib pandas tqdm scikit-image onnx

# 4. Copy your experiment code into the container
COPY ./CodeInsectInspired /isaac-sim/CodeInsectInspired
COPY ./isaac-sim_assets /isaac-sim/isaac-sim_assets

# 5. (Optional) Set a default command to run your script
# This allows them to just type "docker run ..." without complex arguments
CMD ["./python.sh", "CodeInsectInspired/run_ANN_homing_simulations.py"]
