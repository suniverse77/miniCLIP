from collections import OrderedDict

import torch
import torch.nn as nn

from .adapter import Text_Adapter

"""Subclass torch's LayerNorm to handle fp16."""
class LayerNorm(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)
    

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x


class Transformer(nn.Module):
    def __init__(self, embed_dim: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.embed_dim = embed_dim
        self.layers = layers

        self.resblocks = nn.ModuleList(
            [ResidualAttentionBlock(embed_dim, heads, attn_mask) for _ in range(layers)]
        )

        self.text_adapter = Text_Adapter(in_dim=256, out_dim=256)

        print("Transformer blocks: ", len(self.resblocks))
        
    def forward(self, x: torch.Tensor, fearure_layers=None, visual_prompt=None, insert_adapter=False, adapter_layer=4, _lambda=0.1):
        out = []
        prefix_len = len(visual_prompt) if visual_prompt is not None else 0

        # Transformer layer 반복 (현재 10번 반복)
        for i, block in enumerate(self.resblocks):
            if i < prefix_len:
                x = torch.cat([visual_prompt[i:i+1].repeat(x.size(0), 1, 1), x], dim=1)

            # Transformer layer 통과
            x = block(x)

            if insert_adapter and i < adapter_layer:
                x = (1 - _lambda) * x + _lambda * self.text_adapter(x)

            if i < prefix_len:
                x = x[:, visual_prompt[i:i+1].size(1):]

            # 특정 layer의 feature를 추출
            if fearure_layers is not None and i+1 in fearure_layers:
                out.append(x)

        if fearure_layers is None:
            return x
        else:
            return out
        

class VisionTransformer(nn.Module):
    def __init__(
            self,
            input_resolution: int, 
            patch_size: int, 
            embed_dim: int, 
            layers: int, 
            heads: int, 
            output_dim: int
    ):
        super().__init__()

        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = embed_dim ** -0.5
        # shape: [embed_dim]
        self.class_embedding = nn.Parameter(scale * torch.randn(embed_dim))
        # shape: [num_patch + 1, embed_dim]
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, embed_dim))

        self.ln_pre = LayerNorm(embed_dim)

        self.transformer = Transformer(embed_dim, layers, heads)

        self.ln_post = LayerNorm(embed_dim)
        self.proj = nn.Parameter(scale * torch.randn(embed_dim, output_dim))
    
    def forward(self, x: torch.Tensor, feature_layers=[24], visual_prompt=None, insert_adapter=False):
        # Patch Embedding: [B, embed_dim, grid, grid] → [B, embed_dim, num_patch] → [B, num_patch, embed_dim]
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)

        # Make CLS token: [embed_dim] → [B, 1, embed_dim] → boadcasting
        cls_token = self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)

        # Concat CLS token: [B, num_patch + 1, embed_dim]
        x = torch.cat([cls_token, x], dim=1)

        # Positional Embedding의 grid 한 변 개수
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        # Input의 grid 한 변 개수
        new_side = int((x.shape[1] - 1) ** 0.5)

        # 학습과 추론 때 input의 사이즈가 다른 경우 → 보간 수행
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(x.shape[-1], new_side * new_side).transpose(0, 1)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos], 0)

        # Add Positional Embedding
        x = x + self.positional_embedding.to(x.dtype)
        
        if visual_prompt is not None:
            x = torch.cat([x, visual_prompt[:1].repeat(x.size(0), 1, 1)], dim=1)

        # Transformer 입력전 정규화 및 차원 변환: [B, num_patch + 1, embed_dim] → [num_patch + 1, B, embed_dim]
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)

        # Transformer layer 통과 → 출력: 각 layer에서 추출한 feature
        out = self.transformer(x, feature_layers, insert_adapter=insert_adapter)

        for i, o in enumerate(out):
            out[i] = o.permute(1, 0, 2)
            if visual_prompt is not None:
                out[i] = out[i][:, :-visual_prompt.size(1), :]

        return out
    