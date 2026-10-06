from __future__ import annotations

import torch
from torch import nn


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.num_features = 32

    def forward(self, x):
        return self.net(x).flatten(1)


def build_encoder(name: str, pretrained: bool):
    if name == "tiny":
        return TinyEncoder(), 32
    try:
        import timm
    except ImportError as exc:
        raise RuntimeError("Install timm to use pretrained backbones") from exc
    encoder = timm.create_model(name, pretrained=pretrained, num_classes=0, global_pool="avg")
    return encoder, encoder.num_features


class SimCLR(nn.Module):
    def __init__(self, encoder: nn.Module, feature_dim: int, projection_dim: int):
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, projection_dim)
        )

    def forward(self, x):
        return self.projector(self.encoder(x))


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("z1 and z2 must be matching [batch, projection] tensors")
    if z1.shape[0] < 2:
        raise ValueError("NT-Xent requires at least two source images per batch")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    features = torch.cat([z1, z2], dim=0)
    logits = features @ features.T / temperature
    n = z1.shape[0]
    diagonal = torch.eye(2 * n, dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(diagonal, float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(logits.device)
    return nn.functional.cross_entropy(logits, targets)
