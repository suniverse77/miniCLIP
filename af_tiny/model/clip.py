import math
from typing import Tuple, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adapter import Attention_Adapter, CLS_Projector, Text_Adapter, Text_Projector
from .transformer import Transformer, VisionTransformer, LayerNorm

import matplotlib.pyplot as plt # 추가
import os


def gaussian_kernel(size, sigma=2.0):
    x = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    y = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
    x, y = torch.meshgrid(x, y, indexing='ij')
    
    kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    return kernel


class CLIP(nn.Module):
    def __init__(
            self,
            args,
            embed_dim: int,
            # vision
            vision_embed_dim: int,
            vision_heads: int,
            vision_layers: int,
            image_resolution: int,
            vision_patch_size: int,
            # text
            text_embed_dim: int,
            transformer_heads: int,
            transformer_layers: int,
            context_length: int,
            vocab_size: int,          
    ):
        super().__init__()

        self.args = args
        self.context_length = context_length

        self.visual = VisionTransformer(
            input_resolution=image_resolution,
            patch_size=vision_patch_size,
            embed_dim=vision_embed_dim,
            layers=vision_layers,
            heads=vision_heads,
            output_dim=embed_dim
        )

        self.transformer = Transformer(
            embed_dim=text_embed_dim,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, text_embed_dim)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, text_embed_dim))
        self.ln_final = LayerNorm(text_embed_dim)

        self.text_projection = nn.Parameter(torch.empty(text_embed_dim, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.embed_dim ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.embed_dim ** -0.5
        fc_std = (2 * self.transformer.embed_dim) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.embed_dim ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def create_prompt(self, class_names, tokenizer, device):
        prompt_len = self.args.prompt_len

        if class_names is not None:
            self.class_token_dict = {}

            for real_name, alias in class_names.items():
                normal_prompt = f'a photo of {alias} without defect.'
                abnormal_prompt = f'a photo of {alias} with defect.' 

                tokens = tokenizer([normal_prompt, abnormal_prompt]).to(device)
                self.class_token_dict[real_name] = tokens

            self.learnable_prompt = None
        else:
            self.normal_prompt = f'without defect.'
            self.abnormal_prompt = f'with defect.'
            self.tokens = tokenizer([self.normal_prompt, self.abnormal_prompt]).to(device)

            # 학습 가능한 프롬프트
            self.learnable_prompt = nn.Parameter(torch.empty(1, prompt_len, self.token_embedding.weight.shape[-1]).to(device))
            nn.init.normal_(self.learnable_prompt, std=0.01)
            self.learnable_prompt.requires_grad_(True)

    # adapter 삽입 & 학습 파라미터 설정
    def insert_adapter(self, device):
        self.gaussian_kernel = {
            '3': gaussian_kernel(size=3, sigma=4).to(device), 
            '5': gaussian_kernel(size=5, sigma=4).to(device)
        }

        # 이미지 어댑터 추가
        self.cls_proj = CLS_Projector(in_dim=256, out_dim=512).to(device)
        self.attn_adapter = Attention_Adapter(inplanes=self.visual.proj.shape[0], outplanes=self.visual.proj.shape[0]).to(device)

        self.visual_params = list(self.cls_proj.parameters()) + list(self.attn_adapter.parameters())

        self.visual_params_dict = {
            'attn_adapter': self.attn_adapter.state_dict(),
            'cls_proj': self.cls_proj.state_dict()
        }

        # 텍스트 어댑터 추가
        if self.learnable_prompt is not None:
            self.textual_params = list([self.state_prompt_embedding])
            self.textual_params_dict = {
                'prompt': self.learnable_prompt
            }
        else:
            self.textual_params = list(self.transformer.text_adapter.parameters())
            self.textual_params_dict = {
                'text_adapter': self.transformer.text_adapter.state_dict(),
            }

        return self.visual_params, self.textual_params, self.textual_params_dict, self.visual_params_dict

    # 이미지 인코더만 학습 or 텍스트 인코더만 학습
    def setup_training_phase(self, phase):
        self.phase = phase

        # Visual Adapter 활성 / Textual Adapter 고정
        if phase == 'visual':
            for param in self.visual_params:
                param.requires_grad = True

            for param in self.textual_params:
                param.requires_grad = False
            
        # Visual Adapter 고정 / Textual Adapter 활성
        elif phase == 'textual':
            for param in self.visual_params:
                param.requires_grad = False

            for param in self.textual_params:
                param.requires_grad = True

    # 텐서 shape 변경: bchw → blc / blc → bchw
    def tensor_reshape(self, x, target):
        if target == 'bchw':
            b, l, c = x.size()
            h = w = int(math.sqrt(l))
            x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)

        if target == 'blc':
            b, c, h, w = x.size()
            l = h * w
            x = x.reshape(b, l, -1)

        return x

    def spatial_aggregation(self, img_tokens, windows):
        img_token_list = []

        for img_token in img_tokens:
            b, l, c = img_token.shape
            h = w = int(math.sqrt(l))
            
            # x shape: [B, D, H, W]
            x = img_token.view(b, h, w, c).permute(0, 3, 1, 2)

            for r in windows:
                if r == 1:
                    img_token_list.append(img_token)
                else:
                    # k: rxr 크기의 가우시안 커널 [1, r, r]
                    k = self.gaussian_kernel[str(r)]
                    # weight shape: [D, 1, r, r]
                    weight = k.view(1, 1, r, r).repeat(c, 1, 1, 1)
                    
                    padding = r // 2

                    out = F.conv2d(x, weight, padding=padding, groups=c)
                    # out shape: [B, L, D]
                    out = out.permute(0, 2, 3, 1).reshape(b, l, c)

                    img_token_list.append(out)

        return img_token_list

    def encode_text(self):
        state_x = self.token_embedding(self.state_prompt_tokens).type(self.dtype)
        state_x = torch.cat([self.learnable_prompt.repeat(2, 1, 1), state_x], dim=1)[:, :77, :]
        state_x = state_x + self.positional_embedding.type(self.dtype)

        # shape: [N, L, D] → [L, N, D]
        state_x = state_x.permute(1, 0, 2)
        state_x = self.transformer(state_x)
        # shape: [L, N, D] → [N, L, D]
        state_x = state_x.permute(1, 0, 2)
        state_x = self.ln_final(state_x).type(self.dtype)
        state_x = state_x[
            torch.arange(state_x.shape[0]), self.args.prompt_len + self.state_prompt_tokens.argmax(dim=-1)
        ] @ self.text_projection

        text_feature = F.normalize(state_x, dim=-1)

        return text_feature

    def encode_text_batch(self, class_names):
        batch_tokens = torch.stack([self.class_token_dict[name] for name in class_names])
        # B: Batch size, N: 2 (Normal/Anomaly), L: 77
        B, N, L = batch_tokens.shape

        # batch_tokens shape: [B*2, 77]
        batch_tokens = batch_tokens.view(B * N, L)
        # state_x shape: [B*2, 77, D]
        state_x = self.token_embedding(batch_tokens).type(self.dtype)

        state_x = state_x + self.positional_embedding.type(self.dtype)

        # 4. Transformer 통과
        state_x = state_x.permute(1, 0, 2)  # [L, B*2, D]
        state_x = self.transformer(state_x, insert_adapter=True, adapter_layer=4, _lambda=0.1)
        # state_x dim = 256 -> 256
        # state_x = self.text_proj(state_x)
        state_x = state_x.permute(1, 0, 2)  # [B*2, L, D]
        state_x = self.ln_final(state_x).type(self.dtype)

        # 5. EOT(End of Token) 위치의 feature 추출
        # 각 클래스마다 단어 길이가 다르므로 argmax로 EOT 위치를 찾음
        eot_indices = batch_tokens.argmax(dim=-1)
        state_x = state_x[torch.arange(state_x.shape[0]), eot_indices] @ self.text_projection
        # state_x = state_x[torch.arange(state_x.shape[0]), eot_indices]

        # 6. Normalize 및 Shape 복원 [B, 2, D]
        text_feature = F.normalize(state_x, dim=-1)
        text_feature = text_feature.view(B, N, -1)

        return text_feature
    
    def encode_image(self, image):
        # pre-trained CLIP의 각 layer에서 feature 추출
        img_tokens = self.visual(image.type(self.dtype), self.args.feature_layers, insert_adapter=False)

        # =============================================================== #
        # CLS token 분리 후 mlp 통과
        # =============================================================== #
        cls_tokens = [img_token[:, 0, :].unsqueeze(1) for img_token in img_tokens]

        # 마지막 layer만 사용
        # cls_token = cls_tokens[-1]
        # cls_token = self.cls_adapter(cls_token)

        # 뒤에서 2번째 layer까지 사용
        cls_tokens = cls_tokens[-2:]
        cls_tokens = [
            self.cls_proj(cls_token) for cls_token in cls_tokens
        ]
        cls_feature = 0
        for token in cls_tokens:
            cls_feature += F.normalize(token, dim=-1)
        cls_token = F.normalize(cls_feature, dim=-1)

        # =============================================================== #
        # img patch는 attn layer 통과
        # =============================================================== #
        img_patches = [img_token[:, 1:, :] for img_token in img_tokens]

        windows = [1, 3, 5]
        img_patches = self.spatial_aggregation(img_patches, windows=windows)

        img_patches = [
            self.visual.ln_post(self.attn_adapter(img_patch)) @ self.visual.proj
            for img_patch in img_patches
        ]
        ## 같은 layer의 다른 r값의 patch들을 서로 더함
        aggregated_patches = []
        for i in range(0, len(img_patches), len(windows)):
            chunk = img_patches[i : i + len(windows)]
            chunk_sum = torch.stack(chunk).sum(dim=0)
            aggregated_patches.append(chunk_sum)

        # 각 feature들을 더함
        img_feature = 0
        for patch in aggregated_patches:
            img_feature += F.normalize(patch, dim=-1)

        img_feature = F.normalize(img_feature, dim=-1)

        return img_feature, cls_token, aggregated_patches
     
    def forward(self, image, class_names=None):
        '''
        B=8 / D=256 / num_patch=1024 / h=w=32

        text_feature shape  : tn[2, D]
        img_feature shape   : tn[B, num_patch+1, D]
        prob shape          : tn[B, num_patch+1, 2] → 각 patch와 프롬프트 간의 유사성
        cls_label shape     : tn[B]                 → 해당 이미지가 anomaly일 확률
        predict_map shape   : tn[B, 1, h, w]        → 이미지의 각 patch가 anomaly일 확률
        '''
        img_feature, cls_token, aggregated_patches = self.encode_image(image)

        if class_names is not None:
            # text_feature shape: [B, 2, D]
            text_feature = self.encode_text_batch(class_names)

            if cls_token is not None:
                # Class Prediction
                # [B, D] x [B, 2, D] -> [B, 2]
                cls_scores = torch.einsum('bd,bwd->bw', cls_token.squeeze(1), text_feature) / 0.07
                predict_label = torch.softmax(cls_scores, dim=-1)[:, 1]

                # Map Prediction
                # [B, P, D] x [B, 2, D] -> [B, P, 2]
                pixel_scores = torch.einsum('bpd,bwd->bpw', img_feature, text_feature) / 0.07
                prob = torch.softmax(pixel_scores, dim=-1)
                predict_map = prob[:, :, 1] # Anomaly score

                # Shape 복원
                b, l = predict_map.size()
                h = w = int(math.sqrt(l))
                predict_map = predict_map.reshape(b, 1, h, w)

            else:
                # cls_token이 없는 경우 (Patch only)
                pixel_scores = torch.einsum('bpd,bwd->bpw', img_feature, text_feature) / 0.07
                prob = torch.softmax(pixel_scores, dim=-1)
                
                predict_map = prob[:, 1:, 1]
                predict_label = prob[:, 0, 1]

                b, l = predict_map.size()
                h = w = int(math.sqrt(l))
                predict_map = predict_map.reshape(b, 1, h, w)
        # 기존 logic         
        else:
            text_feature = self.encode_text()

            if cls_token is not None:
                scores = torch.matmul(img_feature, text_feature.permute(1, 0)) / 0.07
                prob = torch.softmax(scores, dim=-1)
                predict_map = prob[:, :, 1]
                b, l = predict_map.size()
                h = w = int(math.sqrt(l))
                predict_map = predict_map.reshape(b, 1, h, w)

                predict_label = torch.softmax(
                    torch.matmul(cls_token, text_feature.permute(1, 0)) / 0.07, dim=-1
                )[:, 1].view(-1) # index수정 (0,1 -> 1이 anomaly)
            else:
                scores = torch.matmul(img_feature, text_feature.permute(1, 0)) / 0.07
                prob = torch.softmax(scores, dim=-1)
                predict_map = prob[:, 1:, 1]
                b, l = predict_map.size()
                h = w = int(math.sqrt(l))
                predict_map = predict_map.reshape(b, 1, h, w)

                predict_label = prob[:, 0, 1].view(-1)
        
        return predict_label, predict_map, img_feature, cls_token, aggregated_patches
