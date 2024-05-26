import pytest

def test_transformer_forward():
    import torch
    from fast_segformer import Transformer, segformerConfig
    config = segformerConfig(
        img_size=128,
        num_classes=5,
        in_channels=3,
        num_encoders=2,
        patch_size=[7,3],
        stride=[4,2],
        embed_dims=[4, 8],
        depths=[2, 2],
        num_heads=[1, 2],
        mlp_ratios=[4, 4],
        drop_rate=0.0,
        attn_drop_rate=0.0
    )

    model = Transformer(config)

    x = torch.randn(1, 3, 128, 128)
    out = model(x)
    assert out.shape == (1, 5, 128, 128)

