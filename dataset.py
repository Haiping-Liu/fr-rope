import torch
from torch.utils.data import Dataset
import glob
import os
import scanpy as sc
import numpy as np
import pandas as pd


class Slide_paths:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.slide_paths = self.get_slide_paths()

    def get_slide_paths(self):
        return sorted(glob.glob(os.path.join(self.data_dir, '*.h5ad')))


class STDataset(Dataset):
    def __init__(self, data_dir, transform=None, max_cells=2048, target_sum=1e4):
        self.data_dir = data_dir
        self.transform = transform
        self.max_cells = max_cells
        self.target_sum = target_sum

        slide_paths_obj = Slide_paths(data_dir)
        self.slide_paths = slide_paths_obj.slide_paths

        if len(self.slide_paths) == 0:
            raise ValueError(f"No .h5ad files found in {data_dir}")

        print(f"Found {len(self.slide_paths)} slides in {data_dir}")

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

        # Sample cells if needed (only if > max_cells)
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
        if 'spatial' not in adata.obsm:
            raise ValueError(f"No spatial coordinates found in AnnData object")

        coords = adata.obsm['spatial'].copy()

        # Min-max normalization to [0, 1] for each dimension
        for i in range(coords.shape[1]):
            col = coords[:, i]
            col_min, col_max = col.min(), col.max()
            if col_max > col_min:  # Avoid division by zero
                coords[:, i] = (col - col_min) / (col_max - col_min)

        adata.obsm['spatial'] = coords

        return adata

    def _sample_cells(self, coords, genes, ground_truth, max_cells):
        n_cells = coords.shape[0]

        if n_cells > max_cells:
            indices = np.random.permutation(n_cells)[:max_cells]
            coords = coords[indices]
            genes = genes[indices]
            ground_truth = ground_truth[indices]
        return coords, genes, ground_truth


if __name__ == "__main__":
    dataset = STDataset(data_dir='Mouse_hypothalamic')
    print(len(dataset))
    print(dataset[0][0].shape)
    print(dataset[0][1].shape)
    print(dataset[0][2].shape)