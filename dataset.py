import torch
from torch.utils.data import Dataset
import os
import json
import scanpy as sc
import numpy as np
import pandas as pd


class STDataset(Dataset):
    def __init__(self, 
                 data_dir: str,
                 split: str = 'train',
                 rotation_mode: str = 'fixed',
                 rotation_angle: int = 0,
                 transform: Callable = None,
                 max_cells: int = 2048,
                 target_sum: float = 1e4):
        self.data_dir = data_dir
        self.split = split

        self.rotation_mode = rotation_mode
        if rotation_mode == 'fixed':
            self.rotation_angle = rotation_angle
        elif rotation_mode == 'random':
            self.rotation_angles = np.random.randint(0, 360, size=len(self.slide_paths))
        else:
            raise ValueError(f"Invalid rotation mode: {rotation_mode}")

        self.transform = transform
        self.max_cells = max_cells
        self.target_sum = target_sum

        self.slide_paths = self._load_split()

        if len(self.slide_paths) == 0:
            raise ValueError(f"No slides found for split '{split}' in {data_dir}")

        print(f"Loaded {len(self.slide_paths)} slides for split='{split}', rotation={rotation_angle}°")

    def _load_split(self):
        split_path = os.path.join(self.data_dir, 'split.json')

        with open(split_path, 'r') as f:
            split_dict = json.load(f)

        if self.split not in split_dict:
            raise ValueError(f"Split '{self.split}' not found in split.json")

        filenames = split_dict[self.split]
        return [os.path.join(self.data_dir, fname) for fname in filenames]

    def __len__(self):
        return len(self.slide_paths)

    def __getitem__(self, idx):
        slide_path = self.slide_paths[idx]
        adata = sc.read_h5ad(slide_path)

        adata = self._preprocess(adata)
        coords = adata.obsm['spatial']

        if hasattr(adata.X, 'toarray'):
            genes = adata.X.toarray()
        else:
            genes = adata.X

        ground_truth = pd.Categorical(adata.obs['ground_truth']).codes.astype(np.int64)
        n_cells = coords.shape[0]

        if n_cells > self.max_cells:
            coords, genes, ground_truth = self._sample_cells(coords, genes, ground_truth, self.max_cells)
            n_cells = self.max_cells

        coords = torch.from_numpy(coords).float()
        genes = torch.from_numpy(genes).float()
        ground_truth = torch.from_numpy(ground_truth).long()

        return coords, genes, ground_truth

    def _preprocess(self, adata):
        adata = adata.copy()

        sc.pp.normalize_total(adata, target_sum=self.target_sum)
        sc.pp.log1p(adata)
        coords = adata.obsm['spatial'].copy()

        if self.rotation_mode == 'fixed':
            coords = self._rotate_coords(coords, self.rotation_angle)
        elif self.rotation_mode == 'random':
            coords = self._rotate_coords(coords, self.rotation_angles[idx])
        else:
            raise ValueError(f"Invalid rotation mode: {self.rotation_mode}")

        for i in range(coords.shape[1]):
            col = coords[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:
                coords[:, i] = (col - col_min) / (col_max - col_min)

        adata.obsm['spatial'] = coords

        return adata

    def _rotate_coords(self, coords, angle_deg):
        theta = np.radians(angle_deg)
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        R = np.array([
            [cos_theta, -sin_theta],
            [sin_theta,  cos_theta]
        ])

        return coords @ R.T

    def _sample_cells(self, coords, genes, ground_truth, max_cells):
        n_cells = coords.shape[0]

        if n_cells > max_cells:
            indices = np.random.permutation(n_cells)[:max_cells]
            coords = coords[indices]
            genes = genes[indices]
            ground_truth = ground_truth[indices]

        return coords, genes, ground_truth


if __name__ == "__main__":
    dataset = STDataset(data_dir='Mouse_hypothalamic_processed', split='train', rotation_angle=0)
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)
