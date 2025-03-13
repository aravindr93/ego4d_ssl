from typing import Callable
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

str_to_activation: dict[str, nn.Module] = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'tanh': nn.Tanh(),
    'sigmoid': nn.Sigmoid(),
    'identity': nn.Identity(),
}

def build_mlp(input_dim: int, output_dim: int, hidden_sizes: tuple, activation: str = 'relu', 
              output_activation: str = 'identity', batchnorm: bool = True, dropout: float = 0.0) -> nn.Module:
    layer_sizes = (input_dim,) + hidden_sizes
    activation = str_to_activation[activation]
    output_activation = str_to_activation[output_activation]

    layers = []
    if batchnorm:
        layers.append(nn.BatchNorm1d(num_features=input_dim))
    for i in range(len(layer_sizes) - 1):
        layers.extend([nn.Linear(layer_sizes[i], layer_sizes[i + 1]), activation, nn.Dropout(dropout)])
    layers.extend([nn.Linear(layer_sizes[-1], output_dim), output_activation])
    return nn.Sequential(*layers)


class EmbPropModel(nn.Module):
    def __init__(
        self, 
        fused_embedding_dim: int, 
        proprioception_dim: int, 
        proj_dim: int, 
        state_dim: int
    ) -> None:
        super().__init__()
        self.emb_batchnorm = nn.BatchNorm1d(fused_embedding_dim)
        self.emb_mlp = nn.Sequential(
            nn.Linear(fused_embedding_dim, proj_dim), 
            nn.ReLU(),
        )
        self.prop_batchnorm = nn.BatchNorm1d(proprioception_dim)
        self.prop_mlp = nn.Sequential(
            nn.Linear(proprioception_dim, proj_dim), 
            nn.ReLU(),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(2 * proj_dim, state_dim),
            nn.ReLU()
        )

    def forward(self, emb: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        emb = torch.transpose(emb, 1, 2) if len(emb.shape) == 3 else emb
        emb = self.emb_batchnorm(emb)
        emb = torch.transpose(emb, 1, 2) if len(emb.shape) == 3 else emb
        prop = torch.transpose(prop, 1, 2) if len(prop.shape) == 3 else prop
        prop = self.prop_batchnorm(prop)
        prop = torch.transpose(prop, 1, 2) if len(prop.shape) == 3 else prop
        
        emb_proj = self.emb_mlp(emb)
        prop_proj = self.prop_mlp(prop)
        state_emb = torch.cat([emb_proj, prop_proj], dim=-1)
        state_emb = self.state_mlp(state_emb)
        return state_emb

class EmbModel(nn.Module):
    def __init__(
        self, 
        fused_embedding_dim: int, 
        proprioception_dim: int, 
        proj_dim: int, 
        state_dim: int, 
    ) -> None:
        super().__init__()
        self.state_batchnorm = nn.BatchNorm1d(fused_embedding_dim)
        self.state_mlp = nn.Sequential(
            nn.Linear(fused_embedding_dim, state_dim),
            nn.ReLU()
        )

    def forward(self, emb: torch.Tensor, prop: torch.Tensor) -> torch.Tensor:
        emb = torch.transpose(emb, 1, 2) if len(emb.shape) == 3 else emb
        emb = self.state_batchnorm(emb)
        emb = torch.transpose(emb, 1, 2) if len(emb.shape) == 3 else emb
        state_emb = self.state_mlp(emb)
        return state_emb


class InverseDynamicsModel(nn.Module):
    def __init__(
        self, 
        fused_embedding_dim: int, 
        proprioception_dim: int,
        latent_state_dim: int,
        action_dim: int, 
        pvr_model: nn.Module,
        args: DictConfig,
        fusion_preprocess: Callable[[list[torch.Tensor]], torch.Tensor],
        fusion_base: nn.Module,
    ) -> None:
        super().__init__()
        self.head = build_mlp(
            latent_state_dim + latent_state_dim, action_dim, eval(args.hidden_sizes), args.activation, 
            'identity', args.batchnorm, args.dropout)
        self.pvr_model = pvr_model

        if proprioception_dim:
            self.state_model = EmbPropModel(fused_embedding_dim, proprioception_dim, args.state_model_proj, latent_state_dim)
        else:
            self.state_model = EmbModel(fused_embedding_dim, proprioception_dim, args.state_model_proj, latent_state_dim)
        self.fusion_preprocess = fusion_preprocess
        self.fusion_base = fusion_base

        self.freeze_bn = args.freeze_bn

    def forward(
        self, 
        observation_window: list[torch.Tensor], 
        embedding_window: torch.Tensor, 
        curr_prop: torch.Tensor, 
        next_prop: torch.Tensor, 
        action: torch.Tensor
    ):
        embeddings = []
        for frame_batch in observation_window:
            embeddings.append(self.pvr_model(frame_batch).unsqueeze(dim=1))
        curr_observation = self.fusion_preprocess(embeddings[:-1])
        next_observation = self.fusion_preprocess(embeddings[1:])
        curr_latent = self.state_model(self.fusion_base(curr_observation), curr_prop)
        next_latent = self.state_model(self.fusion_base(next_observation), next_prop)

        action_pred = self.head(torch.cat([curr_latent, next_latent], dim=-1))

        inverse_dynamics_loss = F.mse_loss(action_pred, action)

        embeddings = torch.cat(embeddings, dim=1).detach()
        embedding_loss = F.mse_loss(embeddings, embedding_window)
        return inverse_dynamics_loss, embedding_loss.detach()


class ForwardDynamicsModel(nn.Module):
    def __init__(
        self, 
        fused_embedding_dim: int, 
        proprioception_dim: int,
        latent_state_dim: int,
        action_dim: int, 
        pvr_model: nn.Module,
        args: DictConfig,
        fusion_preprocess: Callable[[list[torch.Tensor]], torch.Tensor],
        fusion_base: nn.Module,
    ) -> None:
        super().__init__()
        self.head = build_mlp(
            latent_state_dim + action_dim, latent_state_dim, eval(args.hidden_sizes), args.activation, 
            'identity', args.batchnorm, args.dropout)
        self.pvr_model = pvr_model

        if proprioception_dim:
            self.state_model = EmbPropModel(fused_embedding_dim, proprioception_dim, args.state_model_proj, latent_state_dim)
        else:
            self.state_model = EmbModel(fused_embedding_dim, proprioception_dim, args.state_model_proj, latent_state_dim)
        self.fusion_preprocess = fusion_preprocess
        self.fusion_base = fusion_base

        self.freeze_bn = args.freeze_bn

    def forward(
        self, 
        observation_window: list[torch.Tensor], 
        embedding_window: torch.Tensor, 
        curr_prop: torch.Tensor, 
        next_prop: torch.Tensor, 
        action: torch.Tensor
    ):
        embeddings = []
        for frame_batch in observation_window:
            embeddings.append(self.pvr_model(frame_batch).unsqueeze(dim=1))
        curr_observation = self.fusion_preprocess(embeddings[:-1])
        next_observation = self.fusion_preprocess(embeddings[1:])
        curr_latent = self.state_model(self.fusion_base(curr_observation), curr_prop)
        next_latent = self.state_model(self.fusion_base(next_observation), next_prop)

        next_latent_pred = self.head(torch.cat([curr_latent, action], dim=-1))

        forward_dynamics_loss = F.mse_loss(next_latent_pred, next_latent)

        embeddings = torch.cat(embeddings, dim=1).detach()
        embedding_loss = F.mse_loss(embeddings, embedding_window)
        return forward_dynamics_loss, embedding_loss.detach()
