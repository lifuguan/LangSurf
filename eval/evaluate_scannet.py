from __future__ import annotations
import os
import glob
import random
from argparse import ArgumentParser
import numpy as np
import torch
import time
import logging
from tqdm import tqdm
import clip
import cv2

import colormaps
from autoencoder.model import Autoencoder
from eval_utils import get_logger, create_color_palette

CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                'bathtub']


def build_text_embedding(clip_model, categories):
    with torch.no_grad():
        all_text_embeddings = []
        for category in categories:
            texts = clip.tokenize(category).cuda()  # tokenize
            text_embeddings = clip_model.encode_text(texts)  # embed with text encoder
            text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
            text_embedding = text_embeddings.mean(dim=0)
            text_embedding /= text_embedding.norm()
            all_text_embeddings.append(text_embedding)

    all_text_embeddings = torch.stack(all_text_embeddings, dim=1).cuda()
    return all_text_embeddings.cpu().numpy().T


def get_relevancy(clip_model, restored_feats: torch.Tensor, categories_names: list, use_neg: bool) -> torch.Tensor:
    if use_neg == True:
        negative_names = ["background"]
        pos_text_embedding = torch.from_numpy(build_text_embedding(clip_model, categories_names)).cuda()
        neg_text_embedding = torch.from_numpy(build_text_embedding(clip_model, negative_names)).cuda()
        phrases_embeds = torch.cat([pos_text_embedding, neg_text_embedding], dim=0)
        pred_masks_lvl = restored_feats.half() @ phrases_embeds.t()

        pos_phrases = len(categories_names)
        n_levels, h, w, _ = pred_masks_lvl.shape

        n_phrases_sims = [None for _ in range(pos_phrases)]
        n_levels_sims = [None for _ in range(n_levels)]
        for i in range(n_levels):
            positive_vals = pred_masks_lvl[i, ..., :pos_phrases].reshape(h * w, -1)
            negative_vals = pred_masks_lvl[i, ..., pos_phrases:].reshape(h * w, -1)
            for j in range(pos_phrases):
                repeated_pos = positive_vals[..., j:j + 1].repeat(1, len(negative_names))
                sims = torch.stack((repeated_pos, negative_vals), dim=-1)
                softmax = torch.softmax(3 * sims, dim=-1)
                best_id = softmax[..., 0].argmin(dim=-1)
                probs = torch.gather(softmax, 1,
                                     best_id[..., None, None].expand(best_id.shape[0], len(negative_names), 2))[:, 0, :]
                pos_prob = probs[..., 0:1]
                n_phrases_sims[j] = pos_prob
            n_levels_sims[i] = torch.stack(n_phrases_sims)
        relev_map = torch.stack(n_levels_sims).view(n_levels, pos_phrases, h, w)
    else:
        pos_text_embedding = torch.from_numpy(build_text_embedding(clip_model, categories_names)).cuda()
        pred_masks_lvl = restored_feats.half() @ pos_text_embedding.t()
        pred_masks_lvl = torch.stack([torch.softmax(pred_mask, dim=-1) for pred_mask in pred_masks_lvl])
        relev_map = pred_masks_lvl.permute(0, 3, 1, 2)
    return relev_map


def activate_stream(valid_map,
                    CLASS_LABELS,
                    output_path,
                    index,
                    mask_thresh,
                    image_lable_path=None):
    n_head, n_prompt, h, w = valid_map.shape
    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        for i in range(n_head):
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            VALID_CLASS_IDS = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36])
            gt_img = cv2.imread(image_lable_path, cv2.IMREAD_UNCHANGED)
            pred_img = (output > mask_thresh).float().cpu().numpy()
            pred_img = pred_img * VALID_CLASS_IDS[k]
            pred_mask = (pred_img == VALID_CLASS_IDS[k]).astype(np.uint8)
            gt_mask = (gt_img == VALID_CLASS_IDS[k]).astype(np.uint8)
            intersection = np.sum(pred_mask & gt_mask)
            union = np.sum(pred_mask | gt_mask)
            if np.sum(gt_mask) == 0:
                iou_lvl[i] = -1
            else:
                iou_lvl[i] = intersection / union
            height = pred_img.shape[0]
            width = pred_img.shape[1]
            vis_image = np.zeros([height, width, 3], dtype=np.uint8)
            color_palette = create_color_palette()
            for idx, color in enumerate(color_palette):
                vis_image[pred_img == idx] = color
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
            vis_path = os.path.join(output_path, "iou_" + str(mask_thresh), CLASS_LABELS[k], str(i), f'{index}.png')
            cv2.imwrite(vis_path, vis_image)

        # compute 2D mIoU metric
        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())
    return chosen_iou_list, chosen_lvl_list


def evaluate(feat_dir, output_path, ae_ckpt_path, encoder_hidden_dims,
             decoder_hidden_dims,
             mask_thresh, logger=None, gt_lable_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip.available_models()
    clip_model, preprocess = clip.load("ViT-L/14@336px")
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    ################## evaluate 2d mIoU ##################
    for folder_name in CLASS_LABELS:
        for i in range(len(feat_dir)):
            folder_path = os.path.join(output_path, "iou_" + str(mask_thresh), folder_name, str(i))
            os.makedirs(folder_path, exist_ok=True)
    image_lable_paths = glob.glob(os.path.join(gt_lable_path, "*.png"))
    image_lable_paths = sorted(image_lable_paths)
    chosen_iou_list_sum = [0] * len(CLASS_LABELS)
    num_sum = [0] * len(CLASS_LABELS)
    logger.info("CLASS_LABELS: " + ', '.join(CLASS_LABELS))
    logger.info(f'mask_thresh: {mask_thresh}')
    use_neg = True if 'neg' in output_path else False

    compressed_sem_feats = np.zeros((len(feat_dir), len(glob.glob(os.path.join(feat_dir[0], '*.npy'))),
                                     *np.load(glob.glob(os.path.join(feat_dir[0], '*.npy'))[0]).shape),
                                    dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                key=lambda file_name: int(
                                    os.path.basename(file_name).split(".npy")[0].replace('frame_', '')))
        for j, idx in enumerate(feat_paths_lvl):
            compressed_sem_feats[i][j] = np.load(idx)

    for i in tqdm(range(compressed_sem_feats.shape[1])):
        sem_feat = compressed_sem_feats[:, i, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512

        pred_masks_lvl = get_relevancy(clip_model, restored_feat, CLASS_LABELS, use_neg)
        chosen_iou_list, chosen_lvl_list = activate_stream(pred_masks_lvl, CLASS_LABELS, output_path,
                                                           i, mask_thresh,
                                                           image_lable_path=image_lable_paths[i])
        output = []
        for iou, lvl in zip(chosen_iou_list, chosen_lvl_list):
            output.append(f"{iou:.5f}: ({lvl})")
        logger.info(f"img_{i}: " + ", ".join(output))
        for idx, value in enumerate(chosen_iou_list):
            if value == -1:
                continue
            chosen_iou_list_sum[idx] += value
            num_sum[idx] += 1

    iou_result = [iou / num if num != 0 else -1 for iou, num in zip(chosen_iou_list_sum, num_sum)]
    miou = [x for x in iou_result if x != -1]
    logger.info("iou: " + ", ".join(f"{iou:.5f}" for iou in iou_result))
    logger.info(f"miou: {np.mean(miou):.5f}")


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
    seed_everything(seed_value=42)

    parser = ArgumentParser(description="Evaluation for ScanNet Dataset")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default="autoencoder/ckpt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.53)
    parser.add_argument('--encoder_dims',
                        nargs='+',
                        type=int,
                        default=[512, 256, 128, 64, 32, 16, 3],
                        )
    parser.add_argument('--decoder_dims',
                        nargs='+',
                        type=int,
                        default=[16, 32, 64, 128, 256, 512, 768],
                        )
    args = parser.parse_args()

    ####### config setting ######
    dataset_name = args.dataset_name
    feat_dir = [
        os.path.join(f"data/scannet/{dataset_name}/output", f"{dataset_name}_{i}/test/ours_None/renders_language_npy")
        for i in
        range(1, 4)]
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    gt_label_path = os.path.join(f"data/scannet/{dataset_name}", "gt_iou")
    mask_thresh = args.mask_thresh

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(str(output_path), f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    ###### load Gaussian models ######
    max_iteration = \
    os.listdir(os.path.join(f"data/scannet/{dataset_name}/output", f"{dataset_name}_1/point_cloud"))[-1].split("_")[-1]
    pc_hierarchy_dir = [
        os.path.join(f"data/scannet/{dataset_name}/output",
                     f"{dataset_name}_{i}/point_cloud/iteration_{max_iteration}/point_cloud.ply") for
        i in range(1, 4)]

    evaluate(feat_dir, output_path, ae_ckpt_path, args.encoder_dims, args.decoder_dims, mask_thresh,
             logger=logger, gt_lable_path=gt_label_path)
