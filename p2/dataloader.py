import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

base_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(base_dir)

class CGNDataset(Dataset):
    """
    Dataset for Contact-GraspNet
    """
    def __init__(self, root_dir, split_file, preload=False):
        """
        Expected data layour:
        root/
            scene_xxxxxx.npz
        """

        self.root_dir = root_dir

        self.preload = preload
        self.split_file = split_file
        self.required_keys = [
            "pc_cam",
            "camera_pose",
            "pos_contact_points",
            "pos_contact_dirs",
            "pos_finger_diffs",
            "pos_approach_dirs",
        ]

        with open(split_file, "r") as f:
            self.scene_files = f.read().splitlines()
        # print(self.scene_files)
        self.scenes = []
        if preload:
            self.scenes = [self._load_scene_as_tensors(os.path.join(root_dir, f)) for f in self.scene_files]

    def _load_scene_as_tensors(self, scene_path):
        scene_np = dict(np.load(scene_path))
        missing = [k for k in self.required_keys if k not in scene_np]
        if missing:
            raise KeyError(f"Missing keys {missing} in scene file: {scene_path}")

        scene_torch = {}
        for key in self.required_keys:
            scene_torch[key] = torch.from_numpy(scene_np[key]).to(torch.float32)
        return scene_torch

    def __len__(self):
        return len(self.scene_files)

    def __getitem__(self, idx):
        if self.preload:
            scene = self.scenes[idx]
            return scene
        else:
            scene = self._load_scene_as_tensors(os.path.join(self.root_dir, self.scene_files[idx]))
            return scene

if __name__ == "__main__":
    print(base_dir)
    print(project_root)
    train_dataset = CGNDataset(root_dir=os.path.join(project_root, "data", "preprocessed_data"), split_file=os.path.join(project_root, "data", "splits", "train.txt"))
    val_dataset = CGNDataset(root_dir=os.path.join(project_root, "data", "preprocessed_data"), split_file=os.path.join(project_root, "data", "splits", "val.txt"))
    test_dataset = CGNDataset(root_dir=os.path.join(project_root, "data", "preprocessed_data"), split_file=os.path.join(project_root, "data", "splits", "test.txt"))
    
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    print(len(train_dataloader))
    print(len(val_dataloader))
    print(len(test_dataloader))