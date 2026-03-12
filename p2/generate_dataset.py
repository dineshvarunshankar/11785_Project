""" this file loads the dataloader from contact_graspnet_pytorck and generated the dataset """

import os
import sys
import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from contact_graspnet_pytorch.contact_graspnet_pytorch.acronym_dataloader import AcryonymDataset
import numpy as np
import yaml

# Load the global_config from the config.yaml file
config_path = os.path.join(os.path.dirname(BASE_DIR), 'contact_graspnet_pytorch', 'contact_graspnet_pytorch', 'config.yaml')
with open(config_path, 'r') as f:
    global_config = yaml.safe_load(f)

# 1. Initialize the dataset once
dataset = AcryonymDataset(global_config, train=True, use_saved_renders=False)

# 2. The loop you write
for i in range(len(dataset)):
    # This automatically calls PyRender and the KD-Tree logic for scene i!
    data_dict = dataset[i] 
    tqdm.tqdm.write(f'Processing scene {i:06d}')
    # 3. Save the results
    np.savez_compressed(f'p2/preprocessed_data/scene_{i:06d}.npz', **data_dict)
