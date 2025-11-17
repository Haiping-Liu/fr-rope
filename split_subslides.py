import scanpy as sc
import numpy as np
import argparse
import os
import json
from sklearn.cluster import KMeans
from tqdm import tqdm


def preprocess_slide(slide_path, output_dir, target_cells=2048):
    adata = sc.read_h5ad(slide_path)

    if 'spatial' not in adata.obsm:
        raise ValueError(f"No spatial coordinates in {slide_path}")

    coords = adata.obsm['spatial']
    n_cells = coords.shape[0]

    if n_cells < target_cells:
        print(f"Slide {slide_path} has only {n_cells} cells, skipping")
        return []

    n_clusters = max(1, n_cells // target_cells)
    print(f"Splitting {n_cells} cells into {n_clusters} clusters")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(coords)

    subslide_paths = []
    slide_name = os.path.basename(slide_path).replace('.h5ad', '')

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        n_cluster_cells = mask.sum()

        if n_cluster_cells < 100:
            print(f"Cluster {cluster_id} too small ({n_cluster_cells} cells), skipping")
            continue

        sub_adata = adata[mask].copy()
        output_filename = f"{slide_name}_sub_{cluster_id}.h5ad"
        output_path = os.path.join(output_dir, output_filename)
        sub_adata.write_h5ad(output_path)

        subslide_paths.append(output_filename)
        print(f"  Saved {output_filename} with {n_cluster_cells} cells")

    return subslide_paths


def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    slide_paths = []
    for fname in os.listdir(args.input_dir):
        if fname.endswith('.h5ad'):
            slide_paths.append(os.path.join(args.input_dir, fname))

    print(f"Found {len(slide_paths)} slides")

    all_subslides = []
    for slide_path in tqdm(slide_paths, desc="Processing slides"):
        subslides = preprocess_slide(slide_path, args.output_dir, args.target_cells)
        all_subslides.extend(subslides)

    print(f"\nTotal subslides: {len(all_subslides)}")

    np.random.seed(42)
    indices = np.random.permutation(len(all_subslides))

    n_train = int(len(all_subslides) * args.train_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    split = {
        'train': [all_subslides[i] for i in train_indices],
        'val': [all_subslides[i] for i in val_indices]
    }

    split_path = os.path.join(args.output_dir, 'split.json')
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)

    print(f"\nSplit: Train={len(split['train'])}, Val={len(split['val'])}")
    print(f"Saved split to {split_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--target_cells', type=int, default=2048)
    parser.add_argument('--train_ratio', type=float, default=0.8)

    args = parser.parse_args()
    main(args)
