import os
import random
import logging
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from model.clip import CLIP

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    return seed

def get_logger(filename, perf_filename, args, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}

    # 로그 출력 형식
    formatter = logging.Formatter(
        "[%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    perf_formatter = logging.Formatter("%(message)s")
    perf_logger = logging.getLogger("perf_logger")
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False       

    # 로그를 터미널 출력으로 보낼 준비
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    # 로그를 파일로 보낼 준비 (학습 및 추론때만)
    if args.vis == 0:
        l_fh = logging.FileHandler(filename, "w")
        l_fh.setFormatter(formatter)
        logger.addHandler(l_fh)

        if os.path.exists(perf_filename):
            perf_logger = None
        else:
            p_fh = logging.FileHandler(perf_filename, "w")
            p_fh.setFormatter(perf_formatter)
            perf_logger.addHandler(p_fh)

    return logger, perf_logger

def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def load_model(path, args, device):
    # =============================================================== #
    # 가중치 불러오기
    ckpt = torch.load(path, map_location="cpu")

    if 'state_dict' in ckpt:
        checkpoint = ckpt['state_dict']
    else:
        checkpoint = ckpt

    new_state_dict = OrderedDict()
    for k, v in checkpoint.items():
        new_k = k
        if k.startswith('_image_encoder.module.'):
            new_k = k.removeprefix('_image_encoder.module.')
        elif k.startswith('_text_encoder.module.'):
            new_k = k.removeprefix('_text_encoder.module.')
        elif k.startswith('_logit_scale.module.'):
            new_k = k.removeprefix('_logit_scale.module.')
        
        new_state_dict[new_k] = v

    state_dict = new_state_dict
    # =============================================================== #

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    context_length = state_dict["positional_embedding"].shape[0]

    # projection 후 임베딩 차원 (공통 차원) → 현재: 512
    embed_dim = state_dict["text_projection"].shape[1]
    # ViT 임베딩 차원 → 현재: 256
    vision_embed_dim = state_dict["visual.conv1.weight"].shape[0]
    # 트랜스포머 임베딩 차원 → 현재: 256
    text_embed_dim = state_dict["ln_final.weight"].shape[0]

    # ViT layer 수 → 현재: 10
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    # 트랜스포머 layer 수 → 현재: 3
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    # ViT 헤드 → 현재: 4
    vision_heads = vision_embed_dim // 64
    # 트랜스포머 헤드 → 현재: 4
    transformer_heads = text_embed_dim // 64

    # ViT 패치 크기 → 현재: 16
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    # ViT grid 크기 (한 변의 패치 개수) → 현재: 14
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    # 이미지 해상도 → 현재: 224
    image_resolution = vision_patch_size * grid_size

    model = CLIP(
        args=args,
        embed_dim=embed_dim,
        # vision
        vision_embed_dim=vision_embed_dim,
        vision_heads=vision_heads, 
        vision_layers=vision_layers,
        image_resolution=image_resolution,
        vision_patch_size=vision_patch_size,
        # text
        text_embed_dim=text_embed_dim,
        transformer_heads=transformer_heads,
        transformer_layers=transformer_layers, 
        context_length=context_length,
        vocab_size=vocab_size,
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    # strict=False: 기존 가중치 파일에 있는 건 가져오고, 없는 건 그냥 무시
    model.load_state_dict(state_dict, strict=False)

    if str(device) == "cpu":
        model.float()

    clip_transform = _transform(model.visual.input_resolution)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    return model, clip_transform, target_transform

def check_params_learn(visual_params, textual_params):
    for param in visual_params:
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            print(f"Visual params | Grad Mean: {grad_mean:.6f} | Requires_grad: {param.requires_grad}")
        else:
            print(f"Visual params | Grad: None (학습 안됨!) | Requires_grad: {param.requires_grad}")
        break

    for param in textual_params:
        if param.grad is not None:
            grad_mean = param.grad.abs().mean().item()
            print(f"Textual params | Grad Mean: {grad_mean:.6f} | Requires_grad: {param.requires_grad}")
        else:
            print(f"Textual params | Grad: None (학습 안됨!) | Requires_grad: {param.requires_grad}")
        break    
