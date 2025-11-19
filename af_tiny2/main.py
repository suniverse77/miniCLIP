import os
import random
import logging
import argparse
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from dataset import *
from util.utils import eval_all_class, load_dataset
from util.calcul import calcul_params, calcul_flops
from util.loss_fn import focal_loss, l1_loss, patch_alignment_loss

from model.clip import CLIP
from model.tokenizer import tokenize

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

    model.load_state_dict(state_dict)

    if str(device) == "cpu":
        model.float()

    return model.eval(), _transform(model.visual.input_resolution)
    
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    prompt_path = os.path.join(args.weight_path, "{}_prompt.pt".format(args.dataset))
    adaptor_path = os.path.join(args.weight_path, "{}_adaptor.pt".format(args.dataset))
    exist_weight = os.path.exists(prompt_path) and os.path.exists(adaptor_path)

    if exist_weight:
        mode = 'test'
    else:
        mode = 'train'

    # log 저장
    logger, perf_logger = get_logger(
        filename=os.path.join(args.result_dir, '{}_log_{}.txt'.format(mode, args.dataset)),
        perf_filename=os.path.join(args.result_dir, 'performance_{}.txt'.format(args.dataset)),
        args=args
    )
    print_args(logger, args)

    # =============================================================== #
    # TinyCLIP 모델 불러오기
    clip_model, clip_transform = load_model(path=args.clip_weight, args=args, device=device)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])
    clip_model.eval()
    
    # CLIP 모델의 모든 파라미터를 고정
    for param in clip_model.parameters():
        param.requires_grad_(False)
    
    clip_model = clip_model.to(device)

    # 기존 CLIP에 새로운 모듈 Adaptor와 Prompt를 삽입
    clip_model.insert(args=args, tokenizer=tokenize, device=device)

    # calcul_flops(model=clip_model, args=args)
    # calcul_params(model=clip_model)
    print(f"*** current device  : {device}")
    print(f"*** using dataset   : {args.dataset_list}")
    # =============================================================== #

    train_dataset, test_dataset_dict = load_dataset(args, clip_transform, target_transform)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    
    # Adaptor, Prompt 가중치가 존재하는 경우
    if exist_weight:
        clip_model.state_prompt_embedding = torch.load(
            prompt_path, map_location=torch.device('cpu'), weights_only=False
        )
        clip_model.adaptor = torch.load(
            adaptor_path, map_location=torch.device('cpu'), weights_only=False
        )
        clip_model = clip_model.to(device)
    # 가중치 없는 경우 → 어댑터, 프롬프트 파인튜닝
    else:
        optimizer = torch.optim.Adam(clip_model.get_trainable_parameters(), lr=args.lr, betas=(0.5, 0.999))
       
        for epoch in range(1, args.epochs + 1):
            total_loss = []
            
            for items in tqdm(train_dataloader):
                imgs, labels, gts = items[:3]
                labels = labels.to(device)
                imgs = imgs.to(device)
                gts = gts.to(device)

                predict_labels, predict_masks, img_tokens = clip_model.detect_forward_seg(imgs)
                
                gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
                gts[gts < 0.5] = 0
                gts[gts > 0.5] = 1
                
                loss = focal_loss(predict_labels, labels) + args.lambda1 * (focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)) + args.lambda2 * patch_alignment_loss(img_tokens, labels, gts) 
                optimizer.zero_grad()
                
                loss.backward()
                optimizer.step()
                total_loss.append(loss.item())
                
            logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, np.mean(total_loss)))

            # 가중치 저장
            torch.save(clip_model.state_prompt_embedding, prompt_path)
            torch.save(clip_model.adaptor, adaptor_path)
            logger.info(f"Prompt weights saved to: {prompt_path}")
            logger.info(f"Adaptor weights saved to: {adaptor_path}")

    for dataset_name, test_ds in test_dataset_dict.items():
        logger.info("---------------------------{}------------------------------".format(dataset_name))
        eval_all_class(clip_model, dataset_name, test_ds, args, logger, perf_logger, device)
        logger.info("-------------------------------------------------------------")

      
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')
    
    parser.add_argument('--clip_weight', type=str, default="./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt", help='model')
    parser.add_argument('--weight_path', type=str, default=None, help='load weight path')
    parser.add_argument('--result_dir', type=str, default='./results/', help='dataset list')
    parser.add_argument('--data_dir', type=str, default='./data', help='training dataset')
    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset', choices=['mvtec', 'visa'])
    parser.add_argument('--dataset_list', nargs='+', type=str, default='mvtec', help='dataset list')

    parser.add_argument('--vis', type=int, default=0, help='visualization results')
    parser.add_argument('--seed', type=int, default=122, help='seed')

    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='label combination')
    parser.add_argument('--epochs', type=int, default=2, help='training epoch')
    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')
    parser.add_argument('--lambda1', type=float, default=1, help='lambda1 for loss')
    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')
    parser.add_argument('--category', type=str, default=None, help='normal class')
    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')
    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')
    parser.add_argument('--feature_layers', nargs='+', type=int, default=[6, 12, 18, 24], help='choose vit layers to extract features')
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')

    args = parser.parse_args()
    
    args.seed = setup_seed(args.seed)
    train(args)
    