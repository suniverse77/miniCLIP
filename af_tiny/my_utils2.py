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


mvtec_class_names = {
    'bottle': 'a bottle',
    'cable': 'a cable',
    'capsule': 'a capsule',
    'carpet': 'carpet',
    'grid': 'grid',
    'hazelnut': 'a hazelnut',
    'leather': 'leather',
    'metal_nut': 'a metal nut',
    'pill': 'a pill',
    'screw': 'a screw',
    'tile': 'tile',
    'toothbrush': 'a toothbrush',
    'transistor': 'a transistor',
    'wood': 'wood',
    'zipper': 'a zipper'
}
visa_class_names = {
    'candle': 'a candle',
    'capsules': 'multiple capsules',
    'cashew': 'a cashew',
    'chewinggum': 'a chewing gum',
    'fryum': 'a ried snack',
    'macaroni1': 'macaroni pasta',
    'macaroni2': 'macaroni pasta',
    'pcb1': 'a printed circuit board',
    'pcb2': 'a printed circuit board',
    'pcb3': 'a printed circuit board',
    'pcb4': 'a printed circuit board',
    'pipe_fryum': 'a fried snack'
}
class_names = mvtec_class_names | visa_class_names

def visualize_attention_map(model, dataloader, device, save_dir):
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

    cnt = 0
    for batch_idx, (imgs, labels, gt_maps, class_names, img_paths) in enumerate(dataloader):
        # imgs shape: [B, C, H, W]
        imgs = imgs.to(device)
        
        # 모델 추론
        _, predict_maps, _, cls_tokens, aggregated_patchs = model(imgs, class_names=class_names)

        first_layer_patchs = aggregated_patchs[0]
        second_layer_patchs = aggregated_patchs[1]
        third_layer_patchs = aggregated_patchs[2]
        fourth_layer_patchs = aggregated_patchs[3]

        for idx, (a, b, c, d) in enumerate(zip(first_layer_patchs, second_layer_patchs, third_layer_patchs, fourth_layer_patchs)):
            path_dir = os.path.join(save_dir, f'{class_names[idx]}')
            if not os.path.exists(path_dir):
                os.makedirs(path_dir)
            # 1. 원본 이미지 복원
            img_tensor = imgs[idx].permute(1, 2, 0).cpu()
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
            std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
            img_denorm = img_tensor * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1).numpy()
            img_uint8 = (img_denorm * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)

            # 2. Ground Truth 처리
            gt_tensor = gt_maps[idx].cpu()
            if gt_tensor.dim() == 3:
                gt_tensor = gt_tensor.squeeze(0)
            gt_np = gt_tensor.numpy()

            if gt_np.shape != img_uint8.shape[:2]:
                gt_np = cv2.resize(gt_np, (img_uint8.shape[1], img_uint8.shape[0]), interpolation=cv2.INTER_NEAREST)

            gt_vis = (gt_np * 255).astype(np.uint8)
            gt_vis = cv2.cvtColor(gt_vis, cv2.COLOR_GRAY2RGB)

            # 3. Predict Map 처리 (추가된 부분)
            # interpolate -> (1, 512, 512)
            predict_map_raw = F.interpolate(predict_maps[idx].unsqueeze(0), size=(img_uint8.shape[0], img_uint8.shape[1]), mode='bilinear').detach().cpu().numpy()
            predict_map_raw = predict_map_raw.squeeze(0).squeeze(0) # (H, W) 형태로 만듦
            predict_map_raw = gaussian_filter(predict_map_raw, sigma=4)
            
            # 시각화를 위한 정규화 (Min-Max)
            p_min, p_max = predict_map_raw.min(), predict_map_raw.max()
            predict_norm = (predict_map_raw - p_min) / (p_max - p_min + 1e-8)
            predict_uint8 = (predict_norm * 255).astype(np.uint8)
            predict_vis = cv2.applyColorMap(predict_uint8, cv2.COLORMAP_JET)
            predict_vis = cv2.cvtColor(predict_vis, cv2.COLOR_BGR2RGB) # plt용 RGB 변환

            # 4. Attention Maps 처리 (a, b, c, d)
            # 반복되는 코드를 줄이기 위해 함수형태로 생각하거나 루프를 돌릴 수 있지만, 
            # 기존 구조를 유지하며 작성합니다.
            
            grid_size = int(np.sqrt(a.size(0)))
            a = a.reshape(grid_size, grid_size, -1)
            b = b.reshape(grid_size, grid_size, -1)
            c = c.reshape(grid_size, grid_size, -1)
            d = d.reshape(grid_size, grid_size, -1)
            final = (a+b+c+d) / 4.0

            # 헬퍼 함수: heatmap 생성
            def process_heatmap(tensor_map, img_shape):
                hm = tensor_map.mean(dim=-1).detach().cpu().numpy()
                hm = cv2.resize(hm, (img_shape[1], img_shape[0]))
                hm = gaussian_filter(hm, sigma=4)
                hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
                hm_u8 = np.uint8(255 * hm)
                hm_c = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
                return cv2.cvtColor(hm_c, cv2.COLOR_BGR2RGB)

            heatmap_a = process_heatmap(a, img_uint8.shape)
            heatmap_b = process_heatmap(b, img_uint8.shape)
            heatmap_c = process_heatmap(c, img_uint8.shape)
            heatmap_d = process_heatmap(d, img_uint8.shape)
            heatmap_final = process_heatmap(final, img_uint8.shape)

            # 5. 시각화 (Matplotlib) - 1x8 Grid로 변경
            plt.figure(figsize=(16, 4)) # 가로 사이즈를 조금 늘렸습니다
            plt.subplots_adjust(wspace=0.1) # 간격 조정
            


            # img_final 생성을 위해 BGR -> RGB 변환
            img_final = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR) # 원본 코드의 흐름상 필요하다면 유지
            img_final = cv2.cvtColor(img_final, cv2.COLOR_BGR2RGB)
            
            # 리스트 다시 매핑 (위에서 img_final이 정의되지 않아 순서 조정)
            titles = ["Original", "Ground Truth", "Predict Map", "Layer5", "Layer7", "Layer9", "Layer10", "Sum"]
            images = [img_final, gt_vis, predict_vis, heatmap_a, heatmap_b, heatmap_c, heatmap_d, heatmap_final]

            for i in range(8):
                plt.subplot(1, 8, i+1)
                plt.imshow(images[i])
                plt.title(titles[i], fontsize=10)
                plt.axis('off')

            save_path = os.path.join(path_dir, f'vis_{cnt}.png')
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            plt.close()

            cnt += 1
            # 한 배치의 첫 번째 이미지만 저장하고 break (원하는 대로 조절 가능)


    print("Visualization Complete.")

def load_model(clip_path, textual_path, visual_path):
    device = 'cuda'
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

    weight = torch.load(clip_path, map_location="cpu")
    textual_weight = torch.load(textual_path, map_location='cpu')
    visual_weight = torch.load(visual_path, map_location='cpu')

    if 'state_dict' in weight:
        state_dict = weight['state_dict']
    else:
        state_dict = weight

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

    model.create_prompt(class_names=class_names, tokenizer=tokenize, device=device)
    _ = model.insert_adapter(device=device)

    model.load_state_dict(state_dict, strict=False)
    model.transformer.text_adapter.load_state_dict(textual_weight['text_adapter'])
    model.attn_adapter.load_state_dict(visual_weight['attn_adapter'])
    model.cls_proj.load_state_dict(visual_weight['cls_proj'])

    return model.eval(), clip_transform, target_transform, args

if __name__ == '__main__':
    clip_path = './weight/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt'
    textual_path = './weight/small-alter-prompt/8_mvtec_textual.pt'
    visual_path = './weight/small-alter-prompt/8_mvtec_visual.pt'
    # 모델 로드 및 가중치 연결
    model, clip_transform, target_transform, args = load_model(clip_path, textual_path, visual_path)

    # print(model)
    # breakpoint()

    # dataloader 생성
    test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
    test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)

    test_dataset = test_dataset_visa

    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    visualize_attention_map(
        model,
        test_dataloader,
        device='cuda',
        save_dir='./vis_attn/small-alter-prompt/visa'
    )
