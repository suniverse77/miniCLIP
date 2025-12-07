import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
from scipy.ndimage import gaussian_filter

from model.clip import CLIP


def transform_invert(img_, transform_train):
    if 'Normalize' in str(transform_train):
        norm_transform = list(filter(lambda x: isinstance(x, transforms.Normalize), transform_train.transforms))
        mean = torch.tensor(norm_transform[0].mean, dtype=img_.dtype, device=img_.device)
        std = torch.tensor(norm_transform[0].std, dtype=img_.dtype, device=img_.device)
        img_.mul_(std[:, None, None]).add_(mean[:, None, None]) 
    return img_

# 원본 이미지 위에 heat map을 겹쳐서 그림 (alpha로 투명도 조절)
def show_cam_on_image(img, anomaly_map, alpha=0.5):
    img = np.float32(img)
    anomaly_map = np.float32(anomaly_map)
    cam  = alpha * img + (1 - alpha) * anomaly_map
    return np.uint8(cam)

# 흑백의 anomaly score map을 컬러로 변환
def cvt2heatmap(gray):
    gray = np.float32(gray)
    gray = normalize(gray)
    gray = gray * 255
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

# 모델 내부의 Attention 가중치를 시각화
def plot_attention(attention_weights, filename, vmax=None):
    nrows, ncols = attention_weights.shape[0], attention_weights.shape[1]
    
    for row in range(nrows):
        for col in range(ncols):
            fig, ax = plt.subplots(figsize=(10, 5))
           
            im = ax.imshow(attention_weights[row, col], 
                         cmap='viridis', 
                         interpolation='nearest',
                         vmax=vmax
                         )
            ax.axis('off')  
            file_path = f"{filename}_{row}_{col}.png"
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            plt.savefig(file_path, bbox_inches='tight', pad_inches=0, transparent=True,)
            plt.close()
            
def visualize(clip_model:CLIP, test_dataset, args, transform, device):
    cnt = 0
    with torch.no_grad():
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        cnt = 0
        for data in test_dataloader:
            img_paths = data[-1]
            labels = data[1]
            if torch.sum(labels) >= 1:
                imgs = data[0].to(device)
                _, anomaly_maps, _, _, _ = clip_model(imgs, class_names=class_names)
                anomaly_maps = F.interpolate(anomaly_maps, size=(imgs.size(-2), imgs.size(-1)), mode='bilinear').cpu().numpy()
                anomaly_maps = np.stack([gaussian_filter(mask, sigma=4) for mask in anomaly_maps])
                anomaly_maps = anomaly_maps.reshape(anomaly_maps.shape[0], anomaly_maps.shape[2],  anomaly_maps.shape[3])
                imgs = transform_invert(imgs, transform)
                gts = data[2].squeeze()
                if len(gts.shape) == 3:
                    pack = zip(imgs, anomaly_maps, gts, labels, img_paths)
                else:
                    pack = zip(imgs, anomaly_maps, labels, img_paths)
                for p in pack:
                    if p[-2] != 0:
                        print(p[-1])
                        save_file_name = '_'.join(p[-1].split('/')[-2:])
                        ano_map = cvt2heatmap(p[1])
                        img = cv2.cvtColor((p[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                        cam_map = show_cam_on_image(img, ano_map)
                        result_path = os.path.join(args.result_dir, 'vis', '{}-shot'.format(args.fewshot), test_dataset.dataset_name, test_dataset.category)
                        if not os.path.exists(result_path):
                            os.makedirs(result_path)
                        if len(p) == 5:
                            gt = cvt2heatmap(p[2])
                            cam_gt = show_cam_on_image(img, gt)
                            res = np.concatenate((img, cam_gt, cam_map), axis=1)
                        else:
                            res = np.vstack((img, ano_map, cam_map))
                        img_path = os.path.join(result_path, save_file_name)
                        cv2.imwrite(img_path, res)
                        cnt += 1