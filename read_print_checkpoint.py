import torch
import numpy as np

# Set print options to show full arrays
torch.set_printoptions(profile="full", linewidth=200, threshold=float('inf'))
np.set_printoptions(threshold=np.inf, linewidth=200)

checkpoint = torch.load("/workspace/ColossalAI/checkpoint.step0.stage0.pt")

def print_nested_dict(d, indent=0):
    """Recursively print nested dictionary with full tensor values"""
    for key, value in d.items():
        print("  " * indent + str(key) + ":")
        if isinstance(value, dict):
            print_nested_dict(value, indent + 1)
        elif isinstance(value, (torch.Tensor, np.ndarray)):
            print("  " * (indent + 1) + f"Shape: {value.shape}")
            print("  " * (indent + 1) + f"Dtype: {value.dtype}")
            print("  " * (indent + 1) + "Values:")
            print("  " * (indent + 1) + str(value))
        else:
            print("  " * (indent + 1) + str(value))
        print()  # Add blank line between items

print("=" * 80)
print("CHECKPOINT CONTENTS:")
print("=" * 80)
print_nested_dict(checkpoint)