import torch
import torch.nn as nn
from fr import apply_rotary_emb, compute_axial_rope


class RoPEAttention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor = None):
        B, N, C = x.shape

        # QKV projection: [B, N, 3*C] -> [B, N, 3, num_heads, head_dim] -> [3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each: [B, num_heads, N, head_dim]

        # skip CLS token at position 0
        if freqs_cis is not None and N > 1:
            q[:, :, 1:], k[:, :, 1:] = apply_rotary_emb(q[:, :, 1:], k[:, :, 1:], freqs_cis=freqs_cis)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # [B, N, C]
        x = self.proj(x)
        x = self.proj_drop(x)

        return x    


class TransformerBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 qkv_bias=True,
                 drop=0, 
                 attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = RoPEAttention(dim, num_heads, qkv_bias, attn_drop, drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(4 * dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor = None):
        x = x + self.attn(self.norm1(x), freqs_cis)
        x = x + self.mlp(self.norm2(x))
        return x    

class SpatialTransformer(nn.Module):
    def __init__(self, gene_dim, dim, num_heads, n_layers, num_classes,
                 qkv_bias=True, drop=0., attn_drop=0.):
        super().__init__()
        self.gene_dim = gene_dim
        self.dim = dim
        self.num_classes = num_classes
        self.n_layers = n_layers

        self.gene_proj = nn.Linear(gene_dim, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.encoder = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(dim)

        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(dim, num_classes),
        )

    def forward(self, genes: torch.Tensor, freqs_cis: torch.Tensor):
        B, N, _ = genes.shape

        x = self.gene_proj(genes)  # [B, N, dim]
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat([cls_tokens, x], dim=1)  

        for block in self.encoder:
            x = block(x, freqs_cis)

        x = self.norm(x)

        cell_tokens = x[:, 1:, :]  
        logits = self.head(cell_tokens)  

        return logits


if __name__ == "__main__":
    # Test model
    batch_size = 1
    gene_dim = 150
    dim = 64
    num_heads = 2
    n_layers = 2
    num_classes = 10
    n_cells = 100

    model = SpatialTransformer(
        gene_dim=gene_dim,
        dim=dim,
        num_heads=num_heads,
        n_layers=n_layers,
        num_classes=num_classes
    )

    genes = torch.randn(batch_size, n_cells, gene_dim)
    coords = torch.randn(n_cells, 2)  # Spatial coordinates

    from fr import compute_axial_rope
    freqs_cis = compute_axial_rope(dim=dim // num_heads, X=coords)

    logits = model(genes, freqs_cis)

    print(f"Input genes shape: {genes.shape}")
    print(f"Input coords shape: {coords.shape}")
    print(f"RoPE freqs_cis shape: {freqs_cis.shape}")
    print(f"Output logits shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
