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
import torch.optim.lr_scheduler as lr_scheduler
from torchvision import transforms
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

from utils.calcul import calcul_params, measure_inference_speed
from utils.evaluation import eval_all_class, load_dataset
from utils.visualization import plot_attention
from utils.loss_fn import focal_loss, l1_loss, patch_alignment_loss
from utils.util import setup_seed, get_logger, print_args, load_model, check_params_learn
from utils.latency import measure_latency, measure_throughput

from model.clip import CLIP
from model.tokenizer import tokenize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import wandb
from functools import partial


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


def textual_train(train_dataloader, clip_model, optimizer, args, device, logger):
    total_loss_list = []
    cls_loss_list = []
    mask_loss_list = []
    align_loss_list = []

    clip_model.setup_training_phase(phase = 'textual')

    for i, (imgs, labels, gts, class_names, img_paths) in enumerate(tqdm(train_dataloader)):
        imgs, labels, gts = imgs.to(device), labels.to(device), gts.to(device)

        predict_labels, predict_masks, img_feature, cls_token, aggregated_patches = clip_model(imgs, class_names=class_names)
        
        gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
        gts[gts < 0.5] = 0
        gts[gts > 0.5] = 1

        loss_cls = focal_loss(predict_labels, labels, gamma=4)
        loss_mask = focal_loss(predict_masks, gts, gamma=4) + l1_loss(predict_masks, gts)
        loss_align = patch_alignment_loss(img_feature, labels, gts, margin=0.0)
        
        loss = loss_cls + args.lambda1 * loss_mask + args.lambda2 * loss_align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_list.append(loss.item())
        cls_loss_list.append(loss_cls.item())
        mask_loss_list.append(loss_mask.item())
        align_loss_list.append(loss_align.item())

        if i % 4 == 0:
            wandb.log({"Training loss": np.mean(total_loss_list)})

    logger.info(
        "Loss: {:.6f}, Loss_cls: {:.6f}, Loss_mask: {:.6f}, Loss_align: {:.6f}".format(
        np.mean(total_loss_list), np.mean(cls_loss_list), np.mean(mask_loss_list), np.mean(align_loss_list)
    ))

    return np.mean(total_loss_list)

def visual_train(train_dataloader, clip_model, optimizer, args, device, logger):
    total_loss_list = []
    cls_loss_list = []
    mask_loss_list = []
    align_loss_list = []

    clip_model.setup_training_phase(phase = 'visual')

    for i, (imgs, labels, gts, class_names, img_paths) in enumerate(tqdm(train_dataloader)):
        imgs, labels, gts = imgs.to(device), labels.to(device), gts.to(device)

        predict_labels, predict_masks, img_feature, cls_token, aggregated_patches = clip_model(imgs, class_names=class_names)
        
        gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
        gts[gts < 0.5] = 0
        gts[gts > 0.5] = 1
        
        loss_cls = focal_loss(predict_labels, labels)
        loss_mask = focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)
        loss_align = patch_alignment_loss(img_feature, labels, gts, margin=0.0)
        
        loss = loss_cls + args.lambda1 * loss_mask + args.lambda2 * loss_align

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss_list.append(loss.item())
        cls_loss_list.append(loss_cls.item())
        mask_loss_list.append(loss_mask.item())
        align_loss_list.append(loss_align.item())

        if i % 4 == 0:
            wandb.log({"Training loss": np.mean(total_loss_list)})

    logger.info(
        "Loss: {:.6f}, Loss_cls: {:.6f}, Loss_mask: {:.6f}, Loss_align: {:.6f}".format(
        np.mean(total_loss_list), np.mean(cls_loss_list), np.mean(mask_loss_list), np.mean(align_loss_list)
    ))

    return np.mean(total_loss_list)

def joint_train(train_dataloader, clip_model, optimizer, args, device):
    total_loss = []

    for i, items in enumerate(tqdm(train_dataloader)):
        imgs, labels, gts = items[:3]
        imgs, labels, gts = imgs.to(device), labels.to(device), gts.to(device)

        predict_labels, predict_masks, img_feature, cls_token, aggregated_patches = clip_model(imgs)
        
        gts = F.interpolate(gts, size=predict_masks[0].shape[-2:], mode='bilinear')
        gts[gts < 0.5] = 0
        gts[gts > 0.5] = 1

        loss_cls = focal_loss(predict_labels, labels)
        loss_patch = args.lambda1 * (focal_loss(predict_masks, gts) + l1_loss(predict_masks, gts)) + args.lambda2 * patch_alignment_loss(img_feature, labels, gts)
        
        loss = loss_cls + loss_patch
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

        if i % 2 == 0:
            wandb.log({"Training loss": np.mean(total_loss)})

    return total_loss

def train(args):
    with wandb.init(project=f"miniCLIP-AD-{args.run_name}", config=args):      
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(args.result_dir):
            os.makedirs(args.result_dir)
        if not os.path.exists(args.adapter_weight):
            os.makedirs(args.adapter_weight)

        logger, perf_logger = get_logger(
            filename=os.path.join(args.result_dir, '{}_log_{}.txt'.format(args.mode, args.dataset)),
            perf_filename=os.path.join(args.result_dir, 'performance_{}.txt'.format(args.dataset)),
            args=args
        )
        print_args(logger, args)

        # TinyCLIP 모델 불러오기
        clip_model, clip_transform, target_transform = load_model(path=args.clip_weight, args=args, device=device)
        clip_model.eval()
        
        # CLIP 모델 파라미터 고정
        for param in clip_model.parameters():
            param.requires_grad_(False)
        clip_model = clip_model.to(device)

        # 기존 CLIP에 학습 가능한 모듈 삽입
        clip_model.create_prompt(class_names=class_names, tokenizer=tokenize, device=device)
        visual_params, textual_params, textual_params_dict, visual_params_dict = clip_model.insert_adapter(device=device)
        learnable_params = visual_params + textual_params

        #measure_inference_speed(clip_model, args, device, args.batch_size, iterations=100, warmup=100)
        # calcul_params(model=clip_model)
        print(f"*** current device  : {device}")
        print(f"*** using dataset   : {args.dataset_list}")
        # =============================================================== #

        train_dataset, test_dataset_dict = load_dataset(args, clip_transform, target_transform)
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

        # =============================================================== #
        # Inference 단계
        # =============================================================== #
        if args.mode == 'test':
            textual_weight = os.path.join(args.adapter_weight, "{}_{}_textual.pt".format(args.eval_epoch, args.dataset))
            visual_weight = os.path.join(args.adapter_weight, "{}_{}_visual.pt".format(args.eval_epoch, args.dataset))

            textual_ckpt = torch.load(textual_weight, map_location='cpu')
            visual_ckpt = torch.load(visual_weight, map_location='cpu')

            # clip_model.state_prompt_embedding.data = textual_ckpt['prompt'].data
            clip_model.transformer.text_adapter.load_state_dict(textual_ckpt['text_adapter'])
            clip_model.attn_adapter.load_state_dict(visual_ckpt['attn_adapter'])
            clip_model.cls_proj.load_state_dict(visual_ckpt['cls_proj'])

            clip_model = clip_model.to(device)

        # =============================================================== #
        # Train 단계
        # =============================================================== #
        joint = 0
        if args.mode == 'train':
            optimizer = torch.optim.Adam(learnable_params, lr=args.lr, betas=(0.5, 0.999))
            t_optimizer = torch.optim.Adam(textual_params, lr=0.0005, betas=(0.5, 0.999))
            v_optimizer = torch.optim.Adam(visual_params, lr=0.0005, betas=(0.5, 0.999))
        
            for epoch in range(1, args.epochs + 1):
                logger.info("Epoch: {}/{}".format(epoch, args.epochs))

                if joint == 1:
                    total_loss = joint_train(train_dataloader, clip_model, optimizer, args, device)

                    logger.info("Epoch: {}/{}, Loss: {:.6f}".format(epoch, args.epochs, np.mean(total_loss)))
                else:
                    t_loss = textual_train(
                        train_dataloader, clip_model, t_optimizer, args, device, logger
                    )
                    v_loss = visual_train(
                        train_dataloader, clip_model, v_optimizer, args, device, logger
                    )
                    total_loss = (t_loss + v_loss) / 2

                wandb.log({
                    "Training loss": total_loss,
                    "T_Loss": t_loss if joint == 0 else 0,
                    "V_Loss": v_loss if joint == 0 else 0,
                    "epoch": epoch
                })

                # 가중치 저장
                if epoch >= 6 and epoch % 2 == 0:
                    textual_weight = os.path.join(args.adapter_weight, "{}_{}_textual.pt".format(epoch, args.dataset))
                    visual_weight = os.path.join(args.adapter_weight, "{}_{}_visual.pt".format(epoch, args.dataset))

                    torch.save(textual_params_dict, textual_weight)
                    torch.save(visual_params_dict, visual_weight)
                    logger.info(f"Textual weights saved to: {textual_weight}")
                    logger.info(f"Visual weights saved to: {visual_weight}")

    for dataset_name, test_ds in test_dataset_dict.items():
        logger.info("---------------------------{}------------------------------".format(dataset_name))
        eval_all_class(clip_model, dataset_name, test_ds, args, logger, perf_logger, device)
        logger.info("-------------------------------------------------------------")

      
if __name__ == '__main__':
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
    parser.add_argument('--lambda1', type=float, default=3, help='lambda1 for loss')
    parser.add_argument('--lambda2', type=float, default=1, help='lambda2 for loss')
    parser.add_argument('--category', type=str, default=None, help='normal class')
    parser.add_argument('--fewshot', type=int, default=0, help='few shot num')
    parser.add_argument('--suffix', type=str, default='defect', help='prompt suffix')
    parser.add_argument('--feature_layers', nargs='+', type=int, default=[6, 8, 10], help='choose vit layers to extract features')
    parser.add_argument('--test_dataset', nargs='+', type=str, default=[], help='choose vit layers to extract features')

    args = parser.parse_args()
    
    args.seed = setup_seed(args.seed)

    # sweep_id = wandb.sweep(sweep_config, project="miniCLIP-AD-Sweep")
    # wandb.agent(sweep_id, function=partial(train, args=args), count=10)

    train(args)
    