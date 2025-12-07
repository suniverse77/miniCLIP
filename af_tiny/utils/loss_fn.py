import copy

import torch
import torch.nn.functional as F

def focal_loss(inputs, targets, alpha=-1, gamma=4, reduction="mean"):
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy(inputs, targets, reduction="none")
    '''
    p_t: 정답의 확신 정도
    실제 정상 픽셀 → p_t = 1 - targets → 높을수록 잘 맞춘 것임
    실제 비정상 픽셀 → p_t = targets → 높을수록 잘 맞춘 것임
    '''
    p_t = inputs * targets + (1 - inputs) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def l1_loss(inputs, targets, reduction="mean"):
    return F.l1_loss(inputs, targets, reduction=reduction)

def patch_alignment_loss(img_feature, labels, gts, margin):
    gts = gts.reshape(img_feature.size(0), -1)
    labels = labels.reshape(labels.size(0), 1)
    # labels = torch.cat([labels, gts], dim=1)
    new_gts = copy.copy(gts)
    if(len(new_gts[new_gts == 0])) == 0:
        return torch.tensor(0.0, device=img_feature.device)
    
    # 정상 픽셀의 라벨값 변경 0 → -1
    new_gts[new_gts == 0] = -1
    b, l = new_gts.size()

    '''
    정상 x 정상: (-1) x (-1) = 1
    비정상 x 비정상: 1 x 1 = 1
    정상 x 비정상: (-1) x 1 = -1
    '''
    # mask: 같은 종류인지 다른 종류인지 나타내는 행렬
    mask = torch.matmul(new_gts.reshape(b, l, 1), new_gts.reshape(b, 1, l))
    total_sim = 0

    img_feature = img_feature[:, :, :]
    # sim: 이미지 feature의 각 픽셀의 유사도 행렬
    sim = torch.matmul(img_feature, img_feature.permute(0, 2, 1))
    '''
    postive_sim: 같은 픽셀끼리의 유사도 평균 → 높아야됨
    negative_sim: 다른 픽셀끼리의 유사도 평균 → 낮아야됨
    '''
    pos_mask = (mask == 1)
    if pos_mask.sum() > 0:
        positive_sim = sim[pos_mask].mean()
    else:
        positive_sim = torch.tensor(0.0, device=img_feature.device)

    # 2. Negative (다른 종류끼리) -> 여기가 NaN의 주범!
    neg_mask = (mask == -1)
    if neg_mask.sum() > 0:
        negative_sim = sim[neg_mask].mean()
    else:
        # 다른 종류의 쌍이 없다면(즉, 모두 정상이거나 모두 불량이면)
        # Loss를 계산할 수 없으므로 0을 반환하거나, 페널티를 주지 않음
        return torch.tensor(0.0, device=img_feature.device)

    loss = F.relu(negative_sim - positive_sim + margin)

    return loss
