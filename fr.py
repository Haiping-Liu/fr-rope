import torch
import numpy as np
import math
from typing import Tuple


def compute_pca_frame(X):
    if isinstance(X, torch.Tensor):
        X = X.cpu().numpy()
    X = np.asarray(X)
    c = X.mean(axis=0, keepdims=True)
    Z = X - c
    Sigma = Z.T @ Z / Z.shape[0]

    eigvals, eigvecs = np.linalg.eigh(Sigma)
    idx = np.argsort(eigvals)[::-1]
    V = eigvecs[:, idx]

    U = V.copy()
    for j in range(U.shape[1]):
        u = U[:, j]
        proj = Z @ u
        m3 = np.mean(proj ** 3)
        if m3 < 0:
            U[:, j] = -U[:, j]

    if np.linalg.det(U) < 0:
        U[:, -1] = -U[:, -1]

    return c.squeeze(0), U


def compute_axial_rope(dim: int,
                       X: torch.Tensor = None,
                       theta: float = 100.0,
                       ) -> torch.Tensor:
    N = X.shape[0]
    x_1, x_2 = X[:, 0], X[:, 1]

    assert dim % 4 == 0
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: dim // 4].float() / dim))
    freqs_x = freqs_x.to(X.device)
    freqs_y = freqs_y.to(X.device)

    freqs_x = torch.outer(x_1, freqs_x)
    freqs_y = torch.outer(x_2, freqs_y)

    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)

    cis = torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)

    return cis


def compute_fr_rope(dim: int, 
                    X: torch.Tensor,    
                    ) -> torch.Tensor:
    c, U = compute_pca_frame(X)
    c = torch.from_numpy(c).to(X.device)
    U = torch.from_numpy(U).to(X.device)
    X_pca = (X - c) @ U
    return compute_axial_rope(dim=dim, X=X_pca)


def compute_mixed_rope(dim: int, 
                    X: torch.Tensor,
                    theta: float = 100.0,
                    ) -> torch.Tensor:
    assert dim % 8 == 0
    half_dim = dim // 2
    axial_cis = compute_axial_rope(dim=half_dim, X=X, theta=theta)
    fr_cis = compute_fr_rope(dim=half_dim, X=X)
    mixed_cis = torch.cat([axial_cis, fr_cis], dim=-1)
    return mixed_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
        Args:
            freqs_cis: [length, head_dim / 2] or [num_heads, length, head_dim / 2]
            x: [batch_size, num_heads, length, head_dim / 2]
    """
    # freqs_cis shape: torch.Size([196, 32])
    # x shape: torch.Size([128, 3, 49, 32])
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == x.shape:
       return freqs_cis
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 else 1 for i, d in enumerate(x.shape)]  # [1, 1, length, head_dim/2, 2]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-3 else 1 for i, d in enumerate(x.shape)]# [1, num_heads, length, head_dim/2, 2]
    elif freqs_cis.shape[-2:] == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim-2 or i == 0 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        purpose:
            broadcast the freqs_cis and apply the rotary embedding to the query and key
        Args:
            xq/xk: [batch_size, num_heads, length, head_dim]
            freqs_cis: [length, head_dim / 2] or [num_heads, length, head_dim / 2]
    """
    # xq shape: torch.Size([128, 3, 49, 64])
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2)) # [batch_size, num_heads, length, head_dim/2]
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_) # [1, num_heads, length, head_dim/2]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3) # merge the dimesions from the third dimension to the last dimension
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3) # [batch_size, num_heads, length, head_dim/2, 2] -> [batch_size, num_heads, length, head_dim]
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device) # [batch_size, num_heads, length, head_dim]

