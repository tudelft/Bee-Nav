# analyze the networks

from simple_network import SimpleCNN, CompactCNN_rgb, IncepAttentionCNN_rgb
import torch
from torchinfo import summary

print("Simple CNN")
model = SimpleCNN()
summary(model, input_size=(1, 3, 400, 1024))
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Parameters: {param_bytes/1024**2:.2f} MiB")

print("\nCompact CNN")
model = CompactCNN_rgb()
summary(model, input_size=(1, 3, 400, 1024))
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Parameters: {param_bytes/1024:.2f} kB")

print("\nInception Attention CNN")
model = IncepAttentionCNN_rgb()
summary(model, input_size=(1, 3, 400, 1024))
param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Parameters: {param_bytes/1024:.2f} kB")

print("\nDone")