import os
import argparse
from collections import OrderedDict

import torch
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode
import numpy as np
from PIL import Image
from utils.visualization import plot_attention
from dataset import *
from model.clip import CLIP
from utils.util import load_model
from model.tokenizer import tokenize

import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from scipy.ndimage import gaussian_filter

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def visualize_attention_map(model, dataloader, target_layer_name, device, save_dir):
    '''
    Input
        model               : 학습된 모델
        dataloader          : 시각화할 dataloader
        target_layer_name   : hook할 layer의 이름
        save_dir            : 시각화 결과 저장 경로
    ---
    Output
        
    '''
    model.eval()
    model.to(device)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    attention_maps = {}
    def get_attention_hook(name):
        def hook(model, input, output):
            attention_maps[name] = output.detach() 
        return hook

    hook_handle = None
    layer_found = False
    for name, module in model.named_modules():
        if name == target_layer_name:
            hook_handle = module.register_forward_hook(get_attention_hook(target_layer_name))
            print(f"Hook registered at: {target_layer_name}")
            layer_found = True
            break
    
    if not layer_found:
        print(f"Error: Layer '{target_layer_name}' not found in the model.")
        return

    print(f"Start processing... Saving to {save_dir}")

    # ---------------------------------------------------------
    # 2. Inference 및 시각화 루프
    # ---------------------------------------------------------
    cnt = 0
    for batch_idx, (imgs, labels, gt_maps, class_name, img_paths) in enumerate(dataloader):
        # imgs shape: [B, C, H, W]
        imgs = imgs.to(device)
        
        # 모델 추론 (여기서 Hook이 작동해 attention_maps에 데이터가 담김)
        _ = model(imgs)

        # img_patchs shape: [B, num_patch, D]
        img_patchs = attention_maps[target_layer_name]

        # 배치 내 각 이미지별 처리
        for i in range(imgs.size(0)):
            # -----------------------------------------------------
            # A. 원본 이미지 복원 (Denormalization)
            # -----------------------------------------------------
            img_tensor = imgs[i].permute(1, 2, 0).cpu() # (H, W, C)
            
            # CLIP/ImageNet 표준 Mean/Std (데이터셋에 맞게 수정 가능)
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
            
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1).numpy()
            
            # OpenCV 사용을 위해 uint8 (0~255)로 변환
            img_uint8 = (img_denorm * 255).astype(np.uint8)
            # RGB -> BGR (OpenCV 호환용, 나중에 plt 그릴땐 다시 바꿈)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            # feat shape: [num_patch, D]
            img_patch = img_patchs[i]

            # spatial_feat shape: [H, W, D]
            grid_size = int(np.sqrt(img_patch.shape[0]))
            spatial_feat = img_patch.reshape(grid_size, grid_size, -1)

            # heatmap shape: [H, W]
            heatmap = spatial_feat.mean(dim=-1).cpu().numpy()
            heatmap = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
            heatmap = gaussian_filter(heatmap, sigma=4)

            # 3. 정규화 (0~1)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)

            # 4. 컬러맵 입히기 (JET)
            heatmap_uint8 = np.uint8(255 * heatmap)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

            # -----------------------------------------------------
            # D. Overlay (합성)
            # -----------------------------------------------------
            # cv2.addWeighted 사용: src1 * alpha + src2 * beta + gamma
            overlay = cv2.addWeighted(img_bgr, 0.5, heatmap_color, 0.5, 0)
            
            # BGR -> RGB 변환 (Matplotlib 저장용)
            img_final = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR) # 원본
            img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
            
            overlay_final = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
            heatmap_final = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

            # -----------------------------------------------------
            # E. 저장
            # -----------------------------------------------------
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 3, 1)
            plt.imshow(img_final)
            plt.title("Original")
            plt.axis('off')

            plt.subplot(1, 3, 2)
            plt.imshow(heatmap_final)
            plt.title("Feature Map (Heatmap)")
            plt.axis('off')

            plt.subplot(1, 3, 3)
            plt.imshow(overlay_final)
            plt.title("Overlay")
            plt.axis('off')

            save_path = os.path.join(save_dir, f'vis_{cnt}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            cnt += 1

        # 테스트용: 1개 배치만 하고 멈추고 싶으면 주석 해제
        # break

    if hook_handle:
        hook_handle.remove()
    print("Visualization Complete.")



def load_model(state_dict, args):
    new_state_dict = OrderedDict()
    
    for k, v in state_dict.items():
        new_k = k
        if k.startswith('_image_encoder.module.'):
            new_k = k.removeprefix('_image_encoder.module.')
        elif k.startswith('_text_encoder.module.'):
            new_k = k.removeprefix('_text_encoder.module.')
        elif k.startswith('_logit_scale.module.'):
            new_k = k.removeprefix('_logit_scale.module.')
        
        new_state_dict[new_k] = v

    state_dict = new_state_dict

    vocab_size = state_dict["token_embedding.weight"].shape[0]
    context_length = state_dict["positional_embedding"].shape[0]
    embed_dim = state_dict["text_projection"].shape[1]
    vision_embed_dim = state_dict["visual.conv1.weight"].shape[0]
    text_embed_dim = state_dict["ln_final.weight"].shape[0]
    vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))
    vision_heads = vision_embed_dim // 64
    transformer_heads = text_embed_dim // 64
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
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

    clip_transform = _transform(model.visual.input_resolution)

    clip_transform.transforms[0] = transforms.Resize(size=(args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BICUBIC)
    clip_transform.transforms[1] = transforms.CenterCrop(size=(args.img_size, args.img_size))
    target_transform = transforms.Compose([
        transforms.Resize(size=clip_transform.transforms[0].size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

    return model.eval(), clip_transform, target_transform

def load_args():
    parser = argparse.ArgumentParser(description='Pytorch implemention of AF-CLIP')
    
    parser.add_argument('--run_name', type=str, default='00')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    parser.add_argument('--clip_weight', type=str, default="./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt", help='model')
    parser.add_argument('--adapter_weight', type=str, default=None, help='load weight path')

    parser.add_argument('--result_dir', type=str, default='./results/', help='dataset list')
    parser.add_argument('--data_dir', type=str, default='./data', help='training dataset')
    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset', choices=['mvtec', 'visa'])
    parser.add_argument('--dataset_list', nargs='+', type=str, default='mvtec', help='dataset list')

    parser.add_argument('--vis', type=int, default=0, help='visualization results')
    parser.add_argument('--eval_epoch', type=int, default=8, help='')
    parser.add_argument('--seed', type=int, default=122, help='seed')

    parser.add_argument('--img_size', type=int, default=518)
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--alpha', type=float, default=0.1, help='label combination')
    parser.add_argument('--epochs', type=int, default=8, help='training epoch')
    parser.add_argument('--prompt_len', type=int, default=12, help='prompt length')
    parser.add_argument('--lambda1', type=float, default=1, help='lambda1 for loss')
    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')
    parser.add_argument('--category', type=str, default=None, help='normal class')
    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')
    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')
    parser.add_argument('--feature_layers', nargs='+', type=int, default=[5, 7, 9, 10], help='choose vit layers to extract features')
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    device = 'cuda'
    args = load_args()

    # 가중치 로드
    weight = torch.load('./weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt', map_location="cpu")
    textual_weight = torch.load('./weight/small/12_visa_textual.pt', map_location='cpu')
    visual_weight = torch.load('./weight/small/12_visa_visual.pt', map_location='cpu')

    if 'state_dict' in weight:
        state_dict = weight['state_dict']
    else:
        state_dict = weight

    # 모델 로드 및 가중치 연결
    model, clip_transform, target_transform = load_model(state_dict, args)
    _ = model.insert(tokenizer=tokenize, device=device)

    model.load_state_dict(state_dict, strict=False)
    model.state_prompt_embedding.data = textual_weight['prompt'].data
    model.attn_adapter.load_state_dict(visual_weight['attn_adapter'])
    model.cls_adapter.load_state_dict(visual_weight['cls_adapter'])

    # print(model)
    # breakpoint()

    # dataloader 생성
    test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)

    test_dataset = test_dataset_mvtec

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # 
    visualize_attention_map(
        model,
        test_dataloader,
        target_layer_name=['attn_adapter.attention.attn', 'cls_adapter.mlp'],
        device='cuda',
        save_dir='./vis_attn'
    )
