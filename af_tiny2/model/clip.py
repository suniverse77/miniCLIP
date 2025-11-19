import math
from typing import Tuple, Union

import numpy as np

import torch
import torch.nn as nn

from .adaptor import Adaptor, CNN_Adaptor
from .resnet import ModifiedResNet
from .transformer import Transformer, VisionTransformer, LayerNorm


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

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

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
    
    def insert(self, args, tokenizer, device):
        self.prompt_len = args.prompt_len

        self.normal_cls_prompt = f'without defect.'
        self.anomaly_cls_prompt = f'with defect.'
        self.state_prompt_tokens = tokenizer([self.normal_cls_prompt, self.anomaly_cls_prompt]).to(device)

        # 학습 가능한 프롬프트
        self.state_prompt_embedding = nn.Parameter(torch.empty(1, args.prompt_len, self.token_embedding.weight.shape[-1]).to(device))
        nn.init.normal_(self.state_prompt_embedding, std=0.01)
        self.state_prompt_embedding.requires_grad_(True)
        
        # Adaptor 모듈 생성
        self.adaptor = Adaptor(inplanes=self.visual.proj.shape[0], outplanes=self.visual.proj.shape[0]).to(device)
        self.cnn_adaptor =  CNN_Adaptor(in_dim=1024, out_dim=1024).to(device)
        self.memorybank = None
        self.memory_backbone = None
        self.gaussian_kernel = {'3': gaussian_kernel(size=3, sigma=4).to(device), '5': gaussian_kernel(size=5, sigma=4).to(device)}
        
    
    def encode_state_prompt(self):
        state_x = self.token_embedding(self.state_prompt_tokens).type(self.dtype)
        state_x = torch.cat([self.state_prompt_embedding.repeat(2, 1, 1), state_x], dim=1)[:, :77, :]
        state_x = state_x + self.positional_embedding.type(self.dtype)
        state_x = state_x.permute(1, 0, 2)  # NLD -> LND
        state_x = self.transformer(state_x)
        state_x = state_x.permute(1, 0, 2)  # LND -> NLD
        state_x = self.ln_final(state_x).type(self.dtype)
        state_x = state_x[torch.arange(state_x.shape[0]), self.prompt_len + self.state_prompt_tokens.argmax(dim=-1)] @ self.text_projection
        return state_x 
    
    
    def get_trainable_parameters(self):
        return list([self.state_prompt_embedding]) + list(self.adaptor.parameters())

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return x
    
    def encode_image(self, image, feature_layers=None):
        return self.visual(image.type(self.dtype), feature_layers)
    
    def aggerate_neighbor(self, x, patchsize, stride=1):
        if patchsize == 1:
            return x
        cls_token = x[:, :1, :]
        x = x[:, 1:, :]
        padding = patchsize // 2
        b, l, c = x.size()
        h = w = int(math.sqrt(l))
        x = x.reshape(b, h, w, c).permute(0, 3, 1, 2)
        
        x = torch.nn.functional.unfold(x, kernel_size=patchsize, padding=padding, stride=stride) #b, (c * r * r), h * w
        x = x.permute(0, 2, 1).reshape(-1, c, patchsize * patchsize).permute(0, 2, 1) # (b * h * w,  r * r, c)
        kernel = self.gaussian_kernel[str(patchsize)].reshape(1, patchsize * patchsize, 1)
        x = torch.sum(x * kernel, dim=1).reshape(b, l, c)
        x = torch.cat([cls_token, x], dim=1)
        return x
    
    
    def aggerate_neighbors(self, img_tokens):
        img_token_list = []

        for img_token in img_tokens:
            for r in [1, 3, 5]:
                new_img_token = img_token + self.cnn_adaptor(img_token, r=r)
                img_token_list.append(new_img_token)

        return img_token_list
    
    
    def detect_encode_image(self, image, args):
        # 추출한 feature 개수 * [B, N, D]
        img_tokens = self.encode_image(image, args.feature_layers)

        img_tokens = self.aggerate_neighbors(img_tokens)

        img_tokens = [self.visual.ln_post(self.adaptor(img_token)) @ self.visual.proj for img_token in img_tokens]

        return img_tokens
    
    
    def store_memory(self, image, args):
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens)
        b, l, c = img_tokens[0].size()
        self.memorybank = [torch.nn.functional.normalize(img_token[:, 1:], dim=-1).reshape(-1, c) for img_token in img_tokens]
        
    # 학습 때 이 부분 호출됨
    def detect_forward_seg(self, image):
        text_features = self.encode_state_prompt()
        text_features = torch.nn.functional.normalize(text_features, dim=-1)
        img_tokens = self.detect_encode_image(image, self.args)
        scores = 0
        for img_token in img_tokens:
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = torch.matmul(img_token, text_features.permute(1, 0)) / 0.07
            scores += score
        prob = torch.softmax(scores, dim=-1)
        cls_label = prob[:, 0, 1].view(-1)
        predict_map = prob[:, 1:, 1]
        
        b, l = predict_map.size()
        h = w = int(math.sqrt(l))
        predict_map = predict_map.reshape(b, 1, h, w)
        
        return cls_label, predict_map, img_tokens
        
    
    def detect_forward_memorybank(self, image, args):
        scores = 0
        img_tokens = self.encode_image(image, args.memory_layers)
        img_tokens = self.aggerate_neighbors(img_tokens)
        for i, img_token in enumerate(img_tokens):
            img_token = torch.nn.functional.normalize(img_token, dim=-1)
            score = (1 - torch.matmul(img_token, self.memorybank[i].T)) .min(dim=-1)[0] / 2
            scores += score[:, 1:]
        scores = scores / len(img_tokens)
        cls_label = torch.max(scores, dim=-1)[0]
        b, l = scores.size()
        h = w = int(math.sqrt(l))
        predict_map = scores.reshape(b, 1, h, w)
        return cls_label, predict_map
    
    
    
    def detect_forward(self, image, args):
        cls_label, predict_map, _= self.detect_forward_seg(image)
        if self.memorybank is not None:
            cls_label_memory, predict_map_memory = self.detect_forward_memorybank(image, args)
            predict_map = predict_map_memory + args.alpha * predict_map
            cls_label = cls_label_memory + args.alpha * cls_label
        return cls_label, predict_map
    

    def forward(self, image, text):
        image_features = self.encode_image(image)
        if isinstance(image_features, (list, tuple)):
            image_features = image_features[0]

        text_features = self.encode_text(text)
        if isinstance(text_features, (list, tuple)):
            text_features = text_features[0]

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]

        return logits_per_image, logits_per_text
    
    