#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim2(img1, img2, window_size=11):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean(0)


def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape
    bottom_point = img[..., 2:hd, 1:wd - 1]
    top_point = img[..., 0:hd - 2, 1:wd - 1]
    right_point = img[..., 1:hd - 1, 2:wd]
    left_point = img[..., 1:hd - 1, 0:wd - 2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None, None], (1, 1, 1, 1), mode='constant', value=1.0).squeeze()
    return grad_img


def lncc(ref, nea):
    # ref_gray: [batch_size, total_patch_size]
    # nea_grays: [batch_size, total_patch_size]
    bs, tps = nea.shape
    patch_size = int(np.sqrt(tps))

    ref_nea = ref * nea
    ref_nea = ref_nea.view(bs, 1, patch_size, patch_size)
    ref = ref.view(bs, 1, patch_size, patch_size)
    nea = nea.view(bs, 1, patch_size, patch_size)
    ref2 = ref.pow(2)
    nea2 = nea.pow(2)

    # sum over kernel
    filters = torch.ones(1, 1, patch_size, patch_size, device=ref.device)
    padding = patch_size // 2
    ref_sum = F.conv2d(ref, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea_sum = F.conv2d(nea, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref2_sum = F.conv2d(ref2, filters, stride=1, padding=padding)[:, :, padding, padding]
    nea2_sum = F.conv2d(nea2, filters, stride=1, padding=padding)[:, :, padding, padding]
    ref_nea_sum = F.conv2d(ref_nea, filters, stride=1, padding=padding)[:, :, padding, padding]

    # average over kernel
    ref_avg = ref_sum / tps
    nea_avg = nea_sum / tps

    cross = ref_nea_sum - nea_avg * ref_sum
    ref_var = ref2_sum - ref_avg * ref_sum
    nea_var = nea2_sum - nea_avg * nea_sum

    cc = cross * cross / (ref_var * nea_var + 1e-8)
    ncc = 1 - cc
    ncc = torch.clamp(ncc, 0.0, 2.0)
    ncc = torch.mean(ncc, dim=1, keepdim=True)
    mask = (ncc < 0.9)
    return ncc, mask


def loss_cls_3d(features, predictions, k=5, lambda_val=2.0, max_points=200000, sample_size=800):
    # Randomly downsample
    if features.size(0) > max_points:
        indices = torch.randperm(features.size(0))[:max_points]
        features = features[indices]
        predictions = predictions[indices]
    # Normalize predictions to [0, 1] range
    min_value = predictions.min()
    max_value = predictions.max()
    if max_value > min_value:
        predictions = (predictions - min_value) / (max_value - min_value)

    # Randomly sample points for which we'll compute the loss
    indices = torch.randperm(features.size(0))[:sample_size]
    sample_features = features[indices]
    sample_preds = predictions[indices]

    # Compute top-k nearest neighbors directly in PyTorch
    dists = torch.cdist(sample_features, features)  # Compute pairwise distances
    _, neighbor_indices_tensor = dists.topk(k, largest=False)  # Get top-k smallest distances

    # Fetch neighbor predictions using indexing
    neighbor_preds = predictions[neighbor_indices_tensor]

    # Compute KL divergence
    kl = sample_preds.unsqueeze(1) * (torch.log(sample_preds.unsqueeze(1) + 1e-10) - torch.log(neighbor_preds + 1e-10))
    loss = torch.abs(kl).mean()

    return lambda_val * loss


def get_loss_semantic_group(gt_seg, language_feature, num=10000):
    # Randomly select num indices from gt_seg
    if gt_seg.size(0) < num:
        indices = torch.randperm(gt_seg.size(0))
        num = gt_seg.size(0)
    else:
        indices = torch.randperm(gt_seg.size(0))[:num]
    input_id1 = input_id2 = gt_seg[indices]
    language_feature = language_feature[indices]

    # Expand labels, create masks for valid positive pairs, excluding self-pairs.
    labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
    labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)
    mask_full_positive = labels1_expanded == labels2_expanded
    block_mask = torch.ones(num, num, dtype=torch.bool, device=gt_seg.device)
    block_mask = torch.triu(block_mask, diagonal=0)
    diag_mask = torch.eye(block_mask.shape[0], device=gt_seg.device, dtype=torch.bool)
    # Compute semantic loss for positive pairs
    total_loss = 0
    mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
    semantic_loss = torch.norm(
        language_feature[mask[0]] - language_feature[mask[1]], p=2, dim=-1
    ).nansum()
    total_loss += semantic_loss
    total_loss = total_loss / torch.sum(block_mask).float()
    return 2 * total_loss


def get_loss_instance_group(sam_seg, instance_feature, language_feature, num=10000):
    # Randomly select num indices from gt_seg
    margin = 1.0
    if sam_seg.size(0) < num:
        indices = torch.randperm(sam_seg.size(0))
        num = sam_seg.size(0)
    else:
        indices = torch.randperm(sam_seg.size(0))[:num]
    instance_feature = instance_feature[indices]
    input_id1 = input_id2 = sam_seg[indices]
    language_feature = language_feature[indices]

    # Expand labels, create masks for valid positive pairs, excluding self-pairs.
    labels1_expanded = input_id1.unsqueeze(1).expand(-1, input_id1.shape[0])
    labels2_expanded = input_id2.unsqueeze(0).expand(input_id2.shape[0], -1)
    mask_full_positive = labels1_expanded == labels2_expanded
    mask_full_negative = ~mask_full_positive
    block_mask = torch.ones(num, num, dtype=torch.bool, device=sam_seg.device)
    block_mask = torch.triu(block_mask, diagonal=0)
    diag_mask = torch.eye(block_mask.shape[0], device=sam_seg.device, dtype=torch.bool)

    # Compute instance loss for positive pairs
    total_loss = 0
    mask = torch.where(mask_full_positive * block_mask * (~diag_mask))
    instance_loss_1 = torch.norm(
        instance_feature[mask[0]] - instance_feature[mask[1]], p=2, dim=-1
    ).nansum()
    total_loss += instance_loss_1

    # Create mask for negative pairs and compute language similarity using cosine similarity
    mask = torch.where(mask_full_negative * block_mask)
    language_similarity = torch.nn.functional.cosine_similarity(
        language_feature[mask[0]], language_feature[mask[1]], dim=-1
    )

    # Compute instance loss for negative pairs with margin and language similarity
    instance_loss_2 = (
            torch.relu(
                margin - torch.norm(instance_feature[mask[0]] - instance_feature[mask[1]], p=2, dim=-1)
            ) * (1 + language_similarity)
    ).nansum()
    total_loss += instance_loss_2
    total_loss = total_loss / torch.sum(block_mask).float()
    return 2 * total_loss
