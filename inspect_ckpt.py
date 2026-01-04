import torch
import glob
import os

env_name = "GoToLocal"
ckpt_pattern = f"lang_model/lang_policy_{env_name}_*.pth"
files = glob.glob(ckpt_pattern)

if not files:
    print("No timestamped files found. Checking default.")
    path = f"lang_model/lang_policy_{env_name}.pth"
else:
    path = max(files, key=os.path.getctime)

print(f"Inspecting: {path}")
try:
    ckpt = torch.load(path, map_location="cpu")
    if "mission_adapter" in ckpt:
        sd = ckpt["mission_adapter"]
        # print keys and shapes
        for k, v in sd.items():
            print(f"{k}: {v.shape}")
    else:
        print("No mission_adapter in checkpoint")
except Exception as e:
    print(f"Error loading: {e}")
