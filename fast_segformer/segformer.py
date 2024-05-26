from typing import Optional, Tuple, List, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .utils import DWConv, OverlapPatchEmbed
from .config import segformerConfig

class EagerAttention(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            attn_dropout: Optional[float] = 0.,
            proj_dropout: Optional[float] = 0.
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(
            self,
            x: torch.Tensor,
            height,
            width
            ) -> torch.Tensor:
        print(f"Executing EagerAttention")
        bsz, seqlen, _ = x.shape
        xq: torch.Tensor = self.wq(x)
        xk: torch.Tensor = self.wk(x)
        xv: torch.Tensor = self.wv(x)
        queries = xq.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        keys = xk.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        values = xv.view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1).type_as(queries)
        attn_output = torch.matmul(self.attn_drop(attn), values)

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)    
        return self.proj_drop(self.wo(attn_output))

class TransformerBlock(nn.Module):
    def __init__(
            self,
            hidden_size: int,
            num_heads: int,
            mlp_ratio: Optional[int] = 4,
            attn_dropout = 0.,
            proj_dropout = 0.,
        ):
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.attn = EagerAttention(
            hidden_size,
            num_heads,
            attn_dropout,
            proj_dropout
        )
        self.ff_norm = nn.LayerNorm(hidden_size)
        self.ff = FeedForward(
            hidden_size,
            mlp_ratio,
            proj_dropout
        )

    def forward(self, x: torch.Tensor, height, width) -> torch.Tensor:
        print(f"Executing transformer block")
        h = x + self.attn(self.attn_norm(x), height, width)
        h = h + self.ff(self.ff_norm(h), height, width)
        return h
        
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.dwconv = DWConv(hidden_dim)

    def forward(self, x, height, width):
        print(f"Executing FeedForward")
        return self.dropout(self.w2(self.dropout(F.gelu(self.dwconv(self.w1(x), height, width)))))
    
class Transformer(nn.Module):
    def __init__(
            self,
            config: segformerConfig,
        ):
        super().__init__()
        self.img_size = config.img_size
        self.num_classes = config.num_classes
        self.in_channels = config.in_channels
        self.embed_dims = config.embed_dims
        self.depths = config.depths
        self.num_heads = config.num_heads
        self.mlp_ratios = config.mlp_ratios
        self.drop_rate = config.drop_rate
        self.attn_drop_rate = config.attn_drop_rate

        self.patch_embeds = nn.ModuleList([
            OverlapPatchEmbed(
                img_size=self.img_size,
                dim=self.embed_dims[i],
                in_channels=self.in_channels if i == 0 else self.embed_dims[i-1],
                patch_size=config.patch_size[i],
                stride=config.stride[i]
            ) for i in range(config.num_encoders)
        ])

        blocks = []
        for i in range(config.num_encoders):
            layers = nn.ModuleList([
                TransformerBlock(
                    hidden_size=self.embed_dims[i],
                    num_heads=self.num_heads[i],
                    mlp_ratio=self.mlp_ratios[i],
                    attn_dropout=self.attn_drop_rate,
                    proj_dropout=self.drop_rate
                ) for _ in range(self.depths[i])
            ])
            blocks.append(layers)
        self.blocks = nn.ModuleList(blocks)

        self.norm = nn.ModuleList([
            nn.LayerNorm(self.embed_dims[i]) for i in range(config.num_encoders)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        for i in range(len(self.patch_embeds)):
            x, height, width = self.patch_embeds[i](x)
            print(f"{i}. Executing patch embed")
            for j,blk in enumerate(self.blocks[i]):
                print(f"- {i}.{j}: Executing block")
                x = blk(x, height, width)
            x = self.norm[i](x)
            x = x.reshape(B, height, width, -1).permute(0, 3, 1, 2).contiguous()
        return x
        