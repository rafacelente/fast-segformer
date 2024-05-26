from dataclasses import dataclass, field
import torch

@dataclass
class segformerConfig:
    img_size: int = 224
    num_classes: int = 19
    in_channels: int = 3
    num_encoders: int = 4
    patch_size: list[int] = field(default_factory=lambda: [7,3,3,3])
    stride: list[int] = field(default_factory=lambda: [4,2,2,2])
    embed_dims: list[int] = field(default_factory=lambda: [4, 8, 16, 32])
    depths: list[int] = field(default_factory=lambda: [2, 2, 2, 2])
    num_heads: list[int] = field(default_factory=lambda: [1, 2, 2, 2])
    mlp_ratios: list[int] = field(default_factory=lambda: [4, 4, 4, 4])
    drop_rate: float = 0.0
    attn_drop_rate: float = 0.0

@dataclass
class trainerConfig:
    batch_size: int = 32
    num_workers: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    num_epochs: int = 2
    log_steps: int = 100
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

