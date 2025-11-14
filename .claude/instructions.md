# MERFISH FR-RoPE Project

## Project Overview

This project implements and compares two position embedding approaches for Transformer models applied to MERFISH (Multiplexed Error-Robust FISH) spatial transcriptomics data:

1. **Standard RoPE (Rotary Position Embedding)**: Applies sinusoidal rotary embeddings directly to 2D spatial coordinates (x, y)
2. **FR-RoPE (Frame Rotation-invariant RoPE)**: A novel approach that achieves rotation invariance by first aligning coordinates to a canonical frame using PCA with 3rd-order moment orientation correction

## Core Innovation

FR-RoPE uses **3rd-order moments to determine coordinate frame orientation**:
- Computes PCA on spatial coordinates to get principal components
- Uses `m3 = mean(projection^3)` to determine axis direction
- Applies standard RoPE to the canonically-aligned coordinates
- Result: Position embeddings that are invariant to input rotation

## Data

- **Format**: H5AD (Anndata format for spatial transcriptomics)
- **Type**: 2D MERFISH data with gene expression and spatial coordinates
- **Location**: External data directory (not in project repo)
- **Preprocessing**: May require manual rotation for testing rotation invariance

## Tech Stack

- PyTorch (deep learning framework)
- Scanpy (single-cell/spatial transcriptomics analysis)
- NumPy (numerical operations)

## Project Files

- `fr.py` - RoPE and FR-RoPE implementations
- `dataset.py` - MERFISH data loading and preprocessing
- `model.py` - Transformer architecture with RoPE attention
- `train.py` - Training loop

## Development Focus

- Testing rotation invariance properties of FR-RoPE vs standard RoPE
- Implementing 2-layer Transformer with classification head
- Validating 3rd-order moment approach for frame alignment
- Dataset construction with rotation augmentation
