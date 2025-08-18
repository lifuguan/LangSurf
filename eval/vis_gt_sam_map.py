from __future__ import annotations

import json
import os
import glob
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, Union
from argparse import ArgumentParser
import logging
import cv2
import numpy as np
import torch
import time
from tqdm import tqdm

import sys
sys.path.append("..")
import colormaps
from utils_eval import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result

def read_gt_fmap(feature_path, image_id, image_shape, feature_level = 3):
    language_feature_name = os.path.join(feature_path, 'frame_{:05d}'.format(image_id))
    seg_map = torch.from_numpy(np.load(language_feature_name + '_s.npy'))
    if feature_level == 0: # default
        mask = seg_map[0:1, ...]
    elif feature_level == 1: # s
        mask = seg_map[1:2, ...]
    elif feature_level == 2: # m
        mask = seg_map[2:3, ...]
    elif feature_level == 3: # l
        mask = seg_map[3:4, ...]
    else:
        raise ValueError("feature_level=", feature_level)
    
    return mask

def evaluate(feat_dir, output_path, json_folder, mask_thresh, logger):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]

    # instantiate autoencoder and openclip
    clip_model = OpenCLIPNetwork(device)

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx+1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)
        
        # RGB image input
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        # restore clip image feature from npy
        restored_masks = read_gt_fmap(feat_dir, idx+1, image_shape, 3).unsqueeze(0)


    # localization acc
    total_bboxes = 0
    for img_ann in gt_ann.values():
        total_bboxes += len(list(img_ann.keys()))
    acc = acc_num / total_bboxes
    logger.info("Localization accuracy: " + f'{acc:.4f}')


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_num = 42
    seed_everything(seed_num)
    
    parser = ArgumentParser(description="prompt any label")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--scene", type=str, default=None)
    parser.add_argument('--feat_dir', type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
    args = parser.parse_args()

    # NOTE config setting
    dataset_name = args.dataset_name
    mask_thresh = args.mask_thresh
    output_path = os.path.join(dataset_name, args.scene, args.output_dir)
    json_folder = os.path.join(args.json_folder, args.scene)
    feat_dir = os.path.join(dataset_name, args.scene, args.feat_dir)

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    # NOTE get feature dirs

    evaluate(feat_dir, output_path, json_folder, mask_thresh, logger)
