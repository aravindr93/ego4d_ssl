from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
import os
import sys
import importlib
from functools import partial, reduce
from operator import mul

from typing import Union, Callable
from torchvision.transforms import InterpolationMode
from torch.nn.modules.linear import Identity
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.layers import PatchEmbed
from r3m import load_r3m
import clip
sys.path.append(os.environ['MAE_PATH'])
import models_mae
import mvp

CHECKPOINT_DIR = os.environ['CHECKPOINT_DIR']

clip_vit_model, _clip_vit_preprocess = clip.load("ViT-B/32", device='cpu')
clip_rn50_model, _clip_rn50_preprocess = clip.load("RN50x16", device='cpu')

_resnet_transforms = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

_moco_transforms = T.Compose([
                        T.Resize(256),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])

_mae_transforms = T.Compose([
                        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        T.CenterCrop(224),
                        T.ToTensor(),
                        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                    ])

_r3m_transforms = T.Compose([
                        T.Resize(256, interpolation=InterpolationMode.BICUBIC),
                        T.CenterCrop(224),
                        T.ToTensor(),  # this divides by 255
                        T.Normalize(mean=[0.0, 0.0, 0.0], std=[1/255, 1/255, 1/255]), # this will scale bact to [0-255]
                    ])


MODEL_LIST = [
    'random',
    'resnet50', 'resnet50_rand', 
    'clip_vit', 'clip_rn50',
    'mae-B', 'mae-L', 'mae-H',
    'moco', 'moco_vit',
    'moco_conv5', 'moco_conv4', 'moco_conv3',
    'r3m', 'mvp'
]


# ===================================
# Model Loading
# ===================================
def load_pvr_model(
    embedding_name: str, 
    cwd: str, 
    convert_pil: bool = True
) -> tuple[nn.Module, int, Callable[[Union[np.ndarray, torch.Tensor]], torch.Tensor]]:
    
    # ============================================================
    # Random
    # ============================================================
    if embedding_name == 'random':
        # A small randomly initialized CNN, used for training from scratch baseline
        model = small_cnn(in_channels=3)
        embedding_dim, transforms = 1568, _resnet_transforms
    # ============================================================
    # ResNet50
    # ============================================================
    elif embedding_name == 'resnet50':
        # ResNet50 pretrained on ImageNet
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT, progress=False)
        model.fc = Identity()
        embedding_dim, transforms = 2048, _resnet_transforms
    elif embedding_name == 'resnet50_rand':
        # Randomly initialized ResNet50 features
        model = models.resnet50(weights=None, progress=False)
        model.fc = Identity()
        embedding_dim, transforms = 2048, _resnet_transforms
    # ============================================================
    # MAE
    # ============================================================
    elif embedding_name == 'mae-B':
        model = MAE_embedding_model(checkpoint_path = CHECKPOINT_DIR + 'mae_pretrain_vit_base.pth', arch='mae_vit_base_patch16')
        embedding_dim = 768
        transforms = _mae_transforms
    elif embedding_name == 'mae-L':
        model = MAE_embedding_model(checkpoint_path = CHECKPOINT_DIR + 'mae_pretrain_vit_large.pth', arch='mae_vit_large_patch16')
        embedding_dim = 1024
        transforms = _mae_transforms
    elif embedding_name == 'mae-H':
        model = MAE_embedding_model(checkpoint_path = CHECKPOINT_DIR + 'mae_pretrain_vit_huge.pth', arch='mae_vit_huge_patch14')
        embedding_dim = 1280
        transforms = _mae_transforms
    # ============================================================
    # CLIP
    # ============================================================
    elif embedding_name == 'clip_vit':
        # CLIP with Vision Transformer architecture
        model = clip_vit_model.visual
        transforms = _clip_vit_preprocess
        embedding_dim = 512
    elif embedding_name == 'clip_rn50':
        # CLIP with ResNet50x16 (large model) architecture
        model = clip_rn50_model.visual
        transforms = _clip_rn50_preprocess
        embedding_dim = 768
    # ============================================================
    # MoCo (Aug+)
    # ============================================================
    elif embedding_name == 'moco_conv3':
        model, embedding_dim = moco_conv3_compression_model(CHECKPOINT_DIR + '/moco_v2_conv3.pth.tar')
        transforms = _moco_transforms
    elif embedding_name == 'moco_conv4':
        model, embedding_dim = moco_conv4_compression_model(CHECKPOINT_DIR + '/moco_v2_conv4.pth.tar')
        transforms = _moco_transforms
    elif embedding_name == 'moco_conv5':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_v2_800ep_pretrain.pth.tar')
        transforms = _moco_transforms
    elif embedding_name == 'moco':
        model, embedding_dim = moco_conv5_model(CHECKPOINT_DIR + '/moco_v2_800ep_pretrain.pth.tar')
        transforms = _moco_transforms
    elif embedding_name == 'moco_vit':
        model, embedding_dim = moco_vit_model(CHECKPOINT_DIR + 'vit-b-300ep.pth.tar')
        transforms = _moco_transforms
    # ============================================================
    # PVRs for Robotics Manipulation
    # ============================================================
    elif embedding_name == 'r3m':
        model = load_r3m("resnet50")
        model = model.module.eval()
        model = model.to('cpu')
        embedding_dim = 2048
        transforms = _r3m_transforms
    elif embedding_name == 'mvp':
        model = mvp.load("vits-mae-hoi")
        model.freeze()
        embedding_dim = 384
        transforms = _mae_transforms
    elif embedding_name in os.listdir(f"{cwd}/rep_eval/representations/"):
        representation = importlib.import_module(f"rep_eval.representations.{embedding_name}.load_representation")
        model, embedding_dim, transforms = representation.load_representation(cwd)
    else:
        print("Model not implemented.")
        raise NotImplementedError

    if convert_pil:
        transforms = T.Compose([T.ToPILImage(), transforms])
            
    return model, embedding_dim, transforms


def moco_conv5_model(checkpoint_path: str) -> tuple[nn.Module, int]:
    model = models.resnet50(weights=None, progress=False)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    # rename moco pre-trained keys
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    model.fc = Identity()
    return model, 2048


def moco_conv4_compression_model(checkpoint_path: str) -> tuple[nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    # construct the compressed model
    model = models.resnet.resnet50(weights=None, progress=False)
    downsample = nn.Sequential(
                    nn.Conv2d(2048,
                    42,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=1,
                    dilation=1), model._norm_layer(42))
    model.layer4 = nn.Sequential(
                    model.layer4,
                    models.resnet.BasicBlock(2048,
                        42,
                        stride=1,
                        norm_layer=model._norm_layer,
                        downsample=downsample))
    # Remove the avgpool layer
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.2' in n  for n in msg.unexpected_keys])
    assert len(msg.missing_keys)==0
    # manually computed the embedding dimension to be 2058
    return model, 2058


def moco_conv3_compression_model(checkpoint_path: str) -> tuple[nn.Module, int]:
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    # construct the compressed model
    model = models.resnet.resnet50(weights=None, progress=False)
    downsample1 = nn.Sequential(
        nn.Conv2d(1024,
                  11,
                  kernel_size=3,
                  stride=1,
                  padding=1,
                  groups=1,
                  dilation=1), model._norm_layer(11))

    model.layer3 = nn.Sequential(
        model.layer3,
        models.resnet.BasicBlock(1024,
                                 11,
                                 stride=1,
                                 norm_layer=model._norm_layer,
                                 downsample=downsample1)
    )

    # Remove the avgpool layer
    model.layer4 = nn.Sequential()
    model.avgpool = nn.Sequential()
    model.fc = nn.Sequential()

    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q'
                        ) and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    assert all(['fc.' in n or 'layer4.' in n or 'layer3.2' in n for n in msg.unexpected_keys])
    assert len(msg.missing_keys)==0
    # manually computed the embedding dimension to be 2156
    return model, 2156


class VisionTransformerMoCo(VisionTransformer):
    def __init__(self, stop_grad_conv1=True, **kwargs):
        super().__init__(**kwargs)
        # Use fixed 2D sin-cos position embedding
        self.num_tokens = 1
        self.build_2d_sincos_position_embedding()

        # weight initialization
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                if "qkv" in name:
                    # treat the weights of Q, K, V separately
                    val = math.sqrt(
                        6.0 / float(m.weight.shape[0] // 3 + m.weight.shape[1])
                    )
                    nn.init.uniform_(m.weight, -val, val)
                else:
                    nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.cls_token, std=1e-6)

        if isinstance(self.patch_embed, PatchEmbed):
            # xavier_uniform initialization
            val = math.sqrt(
                6.0
                / float(
                    3 * reduce(mul, self.patch_embed.patch_size, 1) + self.embed_dim
                )
            )
            nn.init.uniform_(self.patch_embed.proj.weight, -val, val)
            nn.init.zeros_(self.patch_embed.proj.bias)

            if stop_grad_conv1:
                self.patch_embed.proj.weight.requires_grad = False
                self.patch_embed.proj.bias.requires_grad = False

    def build_2d_sincos_position_embedding(self, temperature=10000.0):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert (
            self.embed_dim % 4 == 0
        ), "Embed dimension must be divisible by 4 for 2D sin-cos position embedding"
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1.0 / (temperature**omega)
        out_w = torch.einsum("m,d->md", [grid_w.flatten(), omega])
        out_h = torch.einsum("m,d->md", [grid_h.flatten(), omega])
        pos_emb = torch.cat(
            [torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)],
            dim=1,
        )[None, :, :]

        assert self.num_tokens == 1, "Assuming one and only one token, [cls]"
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


def moco_vit_model(checkpoint_path: str) -> tuple[nn.Module, int]:
    model = VisionTransformerMoCo(
        patch_size=16,
        num_classes=4096,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    )
    model.default_cfg = _cfg()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    old_state_dict = checkpoint["state_dict"]
    state_dict = {}
    for k in list(old_state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith("module.base_encoder") and not (
            k.startswith("module.base_encoder.head")
            or k.startswith("module.base_encoder.fc")
        ):
            # remove prefix
            updated_key = k[len("module.base_encoder.") :]
            state_dict[updated_key] = old_state_dict[k]
        # delete renamed or unused k
        del old_state_dict[k]
    msg = model.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"head.bias", "head.weight"} 
    model.head = Identity()
    return model, model.embed_dim


def small_cnn(in_channels: int = 3) -> nn.Module:
    """
        Make a small CNN visual encoder
        Architecture based on DrQ-v2
    """
    model = nn.Sequential(
        nn.Conv2d(in_channels, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
        nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(), 
        nn.Flatten()
    )
    return model


class CombinedModel(nn.Module):
    def __init__(self, model_list: list[nn.Module] = None) -> None:
        """
            Combines features (outputs) from multiple models.
        """
        super(CombinedModel, self).__init__()
        self.models = model_list
        
    def to(self, device: str) -> 'CombinedModel':
        for idx in range(len(self.models)):
            self.models[idx] = self.models[idx].to(device)
        return super().to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layer_outs = [model(x) for model in self.models]
        return torch.cat(layer_outs, axis=-1)


class MAE_embedding_model(nn.Module):
    def __init__(self, checkpoint_path: str, arch: str = 'mae_vit_large_patch16') -> None:
        super().__init__()
        # build model
        model = getattr(models_mae, arch)()
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        self.mae_model = model
    
    def forward(self, imgs: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        latent, _, _ = self.mae_model.forward_encoder(imgs, mask_ratio)
        cls_latent = latent[:, 0, :]
        return cls_latent