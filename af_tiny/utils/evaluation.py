import copy
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy.ndimage import gaussian_filter

import torch
import torch.nn.functional as F
import torch.utils.data

from dataset import *
from model.clip import CLIP
from utils.visualization import visualize


# AUROC & F1 score 계산
def calculate_metrics(scores, labels):
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-16)
    max_f1 = np.max(f1_scores)
    roc = roc_auc_score(labels, scores)

    return {"AUROC": roc, 'max-F1': max_f1}

def get_res_str(metrics):
    score_res_str = ""
    for key, value in metrics.items():
        # score_res_str += '\n'
        for item, v in value.items():
            score_res_str += "{}_{}: {:.6f} ".format(key, item, v) 
    return score_res_str

# 전체 평균 metric 계산
def cal_average_res(total_res):
    avergae = {}
    category_num = len(total_res)
    for res in total_res: # every category res
        for key, ip in res.items(): # sample or pixel
            if key not in avergae:
                avergae[key] = {}
            for m, v in ip.items():
                if m not in avergae[key]:
                    avergae[key][m] = 0
                avergae[key][m] += v
    
    for key, ip in avergae.items():
        for m, v in ip.items():
            avergae[key][m] = v / category_num
    
    return avergae
    
# Image-level & Pixel-level 성능 각각 계산
def evaluation_pixel(clip_model:CLIP, dataset_name, dataloader, args, device):
    pixel_gt_list = []
    pixel_score_list = []
    sample_gt_list = []
    sample_score_list = []
    aupro_list = []
    res = {}

    with torch.no_grad():
        for imgs, labels, gt, class_names, img_paths in tqdm(dataloader):
            imgs = imgs.to(device)
            predict_labels, predict_masks, _, _, _ = clip_model(imgs, class_names=class_names)

            sample_gt_list.append(labels)
            sample_score_list.append(predict_labels.cpu().numpy())
            
            # predict_masks shape: [B, 1, img_h, img_w] ([8, 1, 518, 518])
            predict_masks = F.interpolate(predict_masks, size=(imgs.size(-2), imgs.size(-1)), mode='bilinear').cpu().numpy()
            predict_masks = np.stack([gaussian_filter(mask, sigma=4) for mask in predict_masks])
            
            if dataset_name not in ['br35h', 'brainmri', 'headct']:
                gt[gt > 0.5] = 1
                gt[gt <= 0.5] = 0
                gt = gt.cpu().numpy().astype(int)
                
                pixel_gt_list.append(gt)
                pixel_score_list.append(predict_masks)

        if dataset_name not in ['isic', 'clinic', 'colon', 'kvasir', 'endo']:                
            sample_gt_list = np.concatenate(sample_gt_list)
            sample_score_list = np.concatenate(sample_score_list)
            res['Image'] = calculate_metrics(sample_score_list, sample_gt_list)

        if dataset_name not in ['br35h', 'brainmri', 'headct']:
            pixel_gt_list = np.concatenate(pixel_gt_list)
            pixel_score_list = np.concatenate(pixel_score_list)

            if len(pixel_gt_list.shape) == 4:
                pixel_gt_list = pixel_gt_list.squeeze(1)
            if len(pixel_score_list.shape) == 4:
                pixel_score_list = pixel_score_list.squeeze(1)
            
            res['Pixel'] = calculate_metrics(pixel_score_list.reshape(-1), pixel_gt_list.reshape(-1))
            
    return res


def eval_all_class(clip_model: CLIP, dataset_name, test_dataset, args, logger, perf_logger, device):
    total_res = []

    if args.vis == 0:
        logger.info(get_header_str())
        if perf_logger is not None:
            perf_logger.info(get_header_str())

    for category in test_dataset.categories:
        test_dataset.update(category)            
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        if args.vis != 0:
            logger.info(f"Visualizing results for {category}...")
            visualize(clip_model, test_dataset, args, test_dataset.transform, device)
        else:
            category_res = evaluation_pixel(clip_model, dataset_name, test_dataloader, args, device)
            total_res.append(category_res)

            log_message = (
                f"{category:<12} "
                f"{(category_res['Image']['AUROC'] * 100):<10.2f} "
                f"{(category_res['Image']['max-F1'] * 100):<10.2f} "
                f"{(category_res['Pixel']['AUROC'] * 100):<10.2f} "
                f"{(category_res['Pixel']['max-F1'] * 100):<10.2f} "
            )

            logger.info(log_message)
            if perf_logger is not None:
                perf_logger.info(log_message)

    if args.vis == 0:
        average_res = cal_average_res(total_res)
        
        log_message = (
            f"{'Average':<12} "
            f"{(average_res['Image']['AUROC'] * 100):<10.2f} "
            f"{(average_res['Image']['max-F1'] * 100):<10.2f} "
            f"{(average_res['Pixel']['AUROC'] * 100):<10.2f} "
            f"{(average_res['Pixel']['max-F1'] * 100):<10.2f} "
        )

        logger.info("=============================================================")
        logger.info(log_message)
        if perf_logger is not None:
            perf_logger.info(log_message)
        logger.info("=============================================================")

def get_header_str():
    return (
        f"{'Category':<12} "
        f"{'I_AUROC':<10} {'I_F1':<10} "
        f"{'P_AUROC':<10} {'P_F1':<10}"
    )

def load_dataset(args, clip_transform, target_transform):
    all_test_dataset_dict = {}

    # Industrial dataset
    if 'mvtec' in args.dataset_list:
        test_dataset_mvtec = MVTecDataset(root=args.data_dir, train=False, category=None, transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['mvtec'] = test_dataset_mvtec
    if 'visa' in args.dataset_list:
        test_dataset_visa = VisaDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['visa'] = test_dataset_visa
    if 'btad' in args.dataset_list:
        test_dataset_btad = BTADDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['btad'] = test_dataset_btad
    if 'dagm' in args.dataset_list:
        test_dataset_dagm = DAGMDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['dagm'] = test_dataset_dagm
    if 'dtd' in args.dataset_list:
        test_dataset_dtd = DTDDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['dtd'] = test_dataset_dtd

    # Medical dataset
    if 'brainmri' in args.dataset_list:
        test_dataset_brainmri = BrainMRIDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['brainmri'] = test_dataset_brainmri
    if 'br35h' in args.dataset_list:
        test_dataset_br35h = Br35HDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['br35h'] = test_dataset_br35h
    if 'isic' in args.dataset_list:
        test_dataset_isic = ISICDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['isic'] = test_dataset_isic
    if 'colon' in args.dataset_list:
        test_dataset_colon = ColonDBDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['colon'] = test_dataset_colon
    if 'clinic' in args.dataset_list:
        test_dataset_clinic = ClinicDBDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['clinic'] = test_dataset_clinic
    if 'kvasir' in args.dataset_list:
        test_dataset_kvasir = KvasirDataset(root=args.data_dir, train=False, category=None,transform=clip_transform, gt_target_transform=target_transform)
        all_test_dataset_dict['kvasir'] = test_dataset_kvasir

    if len(args.test_dataset) < 1:
        test_dataset_dict = all_test_dataset_dict
    else:
        test_dataset_dict = {}
        for ds_name in args.test_dataset:
            test_dataset_dict[ds_name] = all_test_dataset_dict[ds_name]

    if args.dataset in test_dataset_dict:
        del test_dataset_dict[args.dataset]

    if args.dataset == 'mvtec':
        train_dataset = test_dataset_mvtec
    else:
        train_dataset = test_dataset_visa

    return train_dataset, test_dataset_dict
