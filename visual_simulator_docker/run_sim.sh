#!/bin/bash

# 1. Allow Docker to draw on your screen (The "Magic" xhost command)
xhost +local:root

# 2. Run the container with all necessary flags for GUI support
docker run --name isaac-sim-experiment --entrypoint bash -it --gpus all --rm --network=host \
    -e "ACCEPT_EULA=Y" \
    -e "PRIVACY_CONSENT=Y" \
    -e "DISPLAY=$DISPLAY" \
    --shm-size=8g \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v $HOME/.Xauthority:/root/.Xauthority \
    -v $HOME/docker/isaac-sim/cache/kit:/isaac-sim/kit/cache:rw \
    -v $HOME/docker/isaac-sim/cache/ov:/root/.cache/ov:rw \
    -v $HOME/docker/isaac-sim/cache/pip:/root/.cache/pip:rw \
    -v $(pwd)/CodeInsectInspired:/isaac-sim/CodeInsectInspired \
    insect-sim:submission
