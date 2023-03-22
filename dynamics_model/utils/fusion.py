from __future__ import annotations
import torch 
import torch.nn as nn
from typing import Callable, Type

# ===================================
# Fusion Preprocessing
# ===================================
def fuse_preprocess_concat(embeddings: list[torch.Tensor]) -> torch.Tensor:
    # Input is a list of history_window number of frames of dimension (..., num_views, embedding_dim)
    # Output is a single torch tensor of dimension (..., history_window * num_views * embedding_dim)
    if isinstance(embeddings[0], torch.Tensor):
        return torch.stack(embeddings, dim=-3).flatten(start_dim=-3)
    else:
        print("Unsupported embedding format in fuse_preprocess_transformer: each embedding should be a torch.Tensor")
        quit()

def fuse_preprocess_flare(embeddings: list[torch.Tensor]) -> torch.Tensor:
    # Input is a list of history_window number of frames of dimension (..., num_views, embedding_dim)
    # Output is a single torch tensor of dimension (..., history_window * num_views * embedding_dim)
    if isinstance(embeddings[0], torch.Tensor):
        history_window = len(embeddings)
        delta = [embeddings[i + 1] - embeddings[i] for i in range(history_window - 1)]
        delta.append(embeddings[-1])
        return torch.stack(delta, dim=-3).flatten(start_dim=-3)
    else:
        print("Unsupported embedding format in fuse_preprocess_transformer: each embedding should be a torch.Tensor")
        quit()

def fuse_preprocess_conv1d(embeddings: list[torch.Tensor]) -> torch.Tensor:
    # Input is a list of history_window number of frames of dimension (..., num_views, embedding_dim)
    # Output is a single torch tensor of dimension (..., history_window * num_views, embedding_dim)
    if isinstance(embeddings[0], torch.Tensor):
        return torch.stack(embeddings, dim=-3).flatten(start_dim=-3, end_dim=-2)
    else:
        print("Unsupported embedding format in fuse_preprocess_transformer: each embedding should be a torch.Tensor")
        quit()

def fuse_preprocess_transformer(embeddings: list[torch.Tensor]) -> torch.Tensor:
    # Input is a list of history_window number of frames of dimension (..., num_views, embedding_dim)
    # Output is a single torch tensor of dimension (..., history_window * num_views, embedding_dim)
    if isinstance(embeddings[0], torch.Tensor):
        return torch.stack(embeddings, dim=-3).flatten(start_dim=-3, end_dim=-2)
    else:
        print("Unsupported embedding format in fuse_preprocess_transformer: each embedding should be a torch.Tensor")
        quit()

fuse_preprocess: dict[str, Callable[[list[torch.Tensor]], torch.Tensor]] = {
    'concat' : fuse_preprocess_concat,
    'flare' : fuse_preprocess_flare,
    'conv1d' : fuse_preprocess_conv1d,
    'transformer' : fuse_preprocess_transformer
}

# ===================================
# Fusion Module Base
# ===================================

class IdentityBase(nn.Module):
    def __init__(self, embedding_dim: int, history_window: int, num_views: int) -> None:
        super().__init__()
        self.latent_dim = history_window * num_views * embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Conv1DBase(nn.Module):
    def __init__(self, embedding_dim: int, history_window: int, num_views: int, projection_dim: int = 256, **kwargs) -> None:
        super().__init__()
        self.projection = nn.Linear(in_features=embedding_dim, out_features=projection_dim)
        self.latent_dim = history_window * num_views * projection_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.projection(x)
        return projected.reshape(projected.shape[0], -1)

class TransformerLayer(nn.Module):
    def __init__(self, embedding_dim: int, num_heads: int, mlp_hidden: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.layernorm1 = nn.LayerNorm(embedding_dim)
        self.msa = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.layernorm2 = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embedding_dim),
            nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = self.layernorm1(x)
        attn, _ = self.msa(norm, norm, norm, need_weights=False)
        attn = self.attn_dropout(attn)
        x = x + attn
        norm = self.layernorm2(x)
        x = x + self.mlp(norm)
        return x

class TransformerBase(nn.Module):
    def __init__(
        self, 
        embedding_dim: int, 
        history_window: int, 
        num_views: int, 
        projection_dim: int = 384, 
        num_layers: int = 1, 
        classify_cls: bool = False, 
        **kwargs
    ) -> None:
        super().__init__()
        self.classify_cls = classify_cls
        if self.classify_cls:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, projection_dim))
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, history_window * num_views + self.classify_cls, projection_dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerLayer(projection_dim, 4, projection_dim * 4))

        self.layernorm = nn.LayerNorm(projection_dim)
        self.latent_dim = (self.classify_cls or history_window * num_views) * projection_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)

        x = self.projection(x)
        if self.classify_cls:
            x = torch.cat([torch.tile(self.cls_token, (N, 1, 1)), x], dim=1)
        
        x = x + torch.tile(self.position_embedding, (N, 1, 1))
        for layer in self.layers:
            x = layer(x)
        x = self.layernorm(x)

        if self.classify_cls:
            return torch.squeeze(x[:, 0], dim=1)
        else:
            return x.reshape(N, -1)

fuse_base: dict[str, Type[nn.Module]] = {
    'concat' : IdentityBase,
    'flare' : IdentityBase,
    'conv1d' : Conv1DBase,
    'transformer' : TransformerBase
}
