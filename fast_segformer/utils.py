from typing import Optional
import torch
import torch.nn as nn


class DWConv(nn.Module):
    def __init__(
            self,
            dim: int,
            kernel_size: Optional[int]= 3,
            stride: Optional[int] = 1,
            padding: Optional[int] = 1
        ):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, stride, padding, groups=dim, bias=True)
        
    def forward(self, x, height, width):
        B, N, C = x.shape
        print(f"dwconv input shape: {x.shape}")
        x = x.transpose(1, 2).view(B, C, height, width)
        x = self.dwconv(x)
        x = x.view(B, C, N).transpose(1, 2)

        return x
        
class OverlapPatchEmbed(nn.Module):
    def __init__(
            self,
            img_size: int,
            dim: int,
            in_channels: int,
            patch_size: int,
            stride: int,
        ):
        super().__init__()

        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size

        H,W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = H * W

        self.proj = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=patch_size,
            stride=stride,
            padding=(patch_size[0] // 2, patch_size[1] // 2)
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        print(f"x shape: {x.shape}")
        x = x.flatten(2).transpose(1, 2)
        print(f"x shape after flatten: {x.shape}")
        x = self.norm(x)
        return x, H, W