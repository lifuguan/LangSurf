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
import colormaps
from autoencoder.model import Autoencoder
from eval_utils import smooth, colormap_saving, vis_mask_save, polygon_to_mask, stack_mask, show_result
import clip


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    logger = logging.getLogger(name)
    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)
    logger.setLevel(log_level)
    return logger


def eval_gt_lerfdata(json_folder: Union[str, Path] = None, ouput_path: Path = None) -> Dict:
    """
    organise lerf's gt annotations
    gt format:
        file name: frame_xxxxx.json
        file content: labelme format
    return:
        gt_ann: dict()
            keys: str(int(idx))
            values: dict()
                keys: str(label)
                values: dict() which contain 'bboxes' and 'mask'
    """
    gt_json_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.json')))
    img_paths = sorted(glob.glob(os.path.join(str(json_folder), 'frame_*.jpg')))
    gt_ann = {}
    for js_path in gt_json_paths:
        img_ann = defaultdict(dict)
        with open(js_path, 'r') as f:
            gt_data = json.load(f)

        h, w = gt_data['info']['height'], gt_data['info']['width']
        idx = int(gt_data['info']['name'].split('_')[-1].split('.jpg')[0]) - 1
        for prompt_data in gt_data["objects"]:
            label = prompt_data['category']
            box = np.asarray(prompt_data['bbox']).reshape(-1)  # x1y1x2y2
            mask = polygon_to_mask((h, w), prompt_data['segmentation'])
            if img_ann[label].get('mask', None) is not None:
                mask = stack_mask(img_ann[label]['mask'], mask)
                img_ann[label]['bboxes'] = np.concatenate(
                    [img_ann[label]['bboxes'].reshape(-1, 4), box.reshape(-1, 4)], axis=0)
            else:
                img_ann[label]['bboxes'] = box
            img_ann[label]['mask'] = mask

            # # save for visulsization
            save_path = ouput_path / 'gt' / gt_data['info']['name'].split('.jpg')[0] / f'{label}.jpg'
            save_path.parent.mkdir(exist_ok=True, parents=True)
            vis_mask_save(mask, save_path)
        gt_ann[f'{idx}'] = img_ann

    return gt_ann, (h, w), img_paths


def render_label_masks(valid_map,
                       categories_names,
                       image_name: Path = None,
                       thresh: float = 0.5):
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    for k in range(n_prompt):
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale, scale)) / (scale ** 2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

            # NOTE truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)

            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        # save for visulsization
        save_path = image_name / f'chosen_{categories_names[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)


def activate_stream(sem_map,
                    image,
                    categories_names,
                    image_name: Path = None,
                    img_ann: Dict = None,
                    thresh: float = 0.5,
                    colormap_options=None):
    valid_map = sem_map  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    chosen_iou_list, chosen_lvl_list = [], []
    for k in range(n_prompt):
        iou_lvl = np.zeros(n_head)
        mask_lvl = np.zeros((n_head, h, w))
        for i in range(n_head):
            # NOTE 加滤波结果后的激活值图中找最大值点
            scale = 30
            kernel = np.ones((scale, scale)) / (scale ** 2)
            np_relev = valid_map[i][k].cpu().numpy()
            avg_filtered = cv2.filter2D(np_relev, -1, kernel)
            avg_filtered = torch.from_numpy(avg_filtered).to(valid_map.device)
            valid_map[i][k] = 0.5 * (avg_filtered + valid_map[i][k])

            output_path_relev = image_name / 'heatmap' / f'{categories_names[k]}_{i}'
            output_path_relev.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_map[i][k].unsqueeze(-1), colormap_options,
                            save_path=output_path_relev)

            # NOTE 与lerf一致，激活值低于0.5的认为是背景

            # truncate the heatmap into mask
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)
            p_i = torch.clip(output - 0.5, 0, 1).unsqueeze(-1)
            valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
            mask = (output < 0.5).squeeze()
            valid_composited[mask, :] = image[mask, :] * 0.3
            output_path_compo = image_name / 'composited' / f'{categories_names[k]}_{i}'
            output_path_compo.parent.mkdir(exist_ok=True, parents=True)
            colormap_saving(valid_composited, colormap_options, save_path=output_path_compo)
            mask_pred = (output.cpu().numpy() > thresh).astype(np.uint8)
            mask_pred = smooth(mask_pred)
            mask_lvl[i] = mask_pred
            mask_gt = img_ann[categories_names[k]]['mask'].astype(np.uint8)

            # calculate iou
            intersection = np.sum(np.logical_and(mask_gt, mask_pred))
            union = np.sum(np.logical_or(mask_gt, mask_pred))
            iou = np.sum(intersection) / np.sum(union)
            iou_lvl[i] = iou

        score_lvl = torch.zeros((n_head,), device=valid_map.device)
        for i in range(n_head):
            score = valid_map[i, k].max()
            score_lvl[i] = score
        chosen_lvl = torch.argmax(score_lvl)

        chosen_iou_list.append(iou_lvl[chosen_lvl])
        chosen_lvl_list.append(chosen_lvl.cpu().numpy())

        # save for visulsization
        save_path = image_name / f'chosen_{categories_names[k]}.png'
        vis_mask_save(mask_lvl[chosen_lvl], save_path)

    return chosen_iou_list, chosen_lvl_list


def lerf_localization(sem_map, image, image_name, img_ann):
    output_path_loca = image_name / 'localization'
    output_path_loca.mkdir(exist_ok=True, parents=True)

    valid_map = sem_map  # 3xkx832x1264
    n_head, n_prompt, h, w = valid_map.shape

    # positive prompts
    acc_num = 0
    positives = list(img_ann.keys())
    for k in range(len(positives)):
        select_output = valid_map[:, k]
        select_output = select_output - torch.min(select_output)
        select_output = select_output / (torch.max(select_output) + 1e-9)
        select_output = select_output * (1.0 - (-1.0)) + (-1.0)
        select_output = torch.clip(select_output, 0, 1)
        # NOTE 平滑后的激活值图中找最大值点
        scale = 30
        kernel = np.ones((scale, scale)) / (scale ** 2)
        np_relev = select_output.cpu().numpy().astype(np.float32)
        avg_filtered = cv2.filter2D(np_relev.transpose(1, 2, 0), -1, kernel)

        score_lvl = np.zeros((n_head,))
        coord_lvl = []
        for i in range(n_head):
            score = avg_filtered[..., i].max()
            coord = np.nonzero(avg_filtered[..., i] == score)
            score_lvl[i] = score
            coord_lvl.append(np.asarray(coord).transpose(1, 0)[..., ::-1])

        selec_head = np.argmax(score_lvl)
        coord_final = coord_lvl[selec_head]

        for box in img_ann[positives[k]]['bboxes'].reshape(-1, 4):
            flag = 0
            x1, y1, x2, y2 = box
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            for cord_list in coord_final:
                if (cord_list[0] >= x_min and cord_list[0] <= x_max and
                        cord_list[1] >= y_min and cord_list[1] <= y_max):
                    acc_num += 1
                    flag = 1
                    break
            if flag != 0:
                break

        # NOTE 将平均后的结果与原结果相加，抑制噪声并保持激活边界清晰
        avg_filtered = torch.from_numpy(avg_filtered[..., selec_head]).unsqueeze(-1).to(select_output.device)
        torch_relev = 0.5 * (avg_filtered + select_output[selec_head].unsqueeze(-1))
        p_i = torch.clip(torch_relev - 0.5, 0, 1)
        valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6), colormaps.ColormapOptions("turbo"))
        mask = (torch_relev < 0.5).squeeze()
        valid_composited[mask, :] = image[mask, :] * 0.3

        save_path = output_path_loca / f"{positives[k]}.png"
        show_result(valid_composited.cpu().numpy(), coord_final,
                    img_ann[positives[k]]['bboxes'], save_path)
    return acc_num


def build_text_embedding(clip_model, categories):
    with torch.no_grad():
        all_text_embeddings = []
        print("Building text embeddings...")
        for category in tqdm(categories):
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
        # negative_names = ["object", "things", "stuff", "texture"]
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
                softmax = torch.softmax(3 * sims, dim=-1)  # NOTE ’4‘是超参，需要debug
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


def evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, encoder_hidden_dims, decoder_hidden_dims,
             logger):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )

    gt_ann, image_shape, image_paths = eval_gt_lerfdata(Path(json_folder), Path(output_path))
    eval_index_list = [int(idx) for idx in list(gt_ann.keys())]
    compressed_sem_feats = np.zeros((len(feat_dir), len(eval_index_list), *image_shape, 3), dtype=np.float32)
    for i in range(len(feat_dir)):
        feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                key=lambda file_name: int(
                                    os.path.basename(file_name).split(".npy")[0].replace('frame_', '')))
        for j, idx in enumerate(eval_index_list):
            compressed_sem_feats[i][j] = np.load(feat_paths_lvl[idx])

    clip.available_models()
    clip_model, preprocess = clip.load("ViT-L/14@336px")
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    chosen_iou_all, chosen_lvl_list = [], []
    acc_num = 0
    for j, idx in enumerate(tqdm(eval_index_list)):
        image_name = Path(output_path) / f'{idx + 1:0>5}'
        image_name.mkdir(exist_ok=True, parents=True)

        sem_feat = compressed_sem_feats[:, j, ...]
        sem_feat = torch.from_numpy(sem_feat).float().to(device)
        rgb_img = cv2.imread(image_paths[j])[..., ::-1]
        rgb_img = (rgb_img / 255.0).astype(np.float32)
        rgb_img = torch.from_numpy(rgb_img).to(device)

        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512

        img_ann = gt_ann[f'{idx}']
        categories_names = list(img_ann.keys())
        use_neg = True if 'neg' in output_path else False
        pred_masks_lvl = get_relevancy(clip_model, restored_feat, categories_names, use_neg)
        c_iou_list, c_lvl = activate_stream(pred_masks_lvl, rgb_img, categories_names, image_name, img_ann,
                                            thresh=mask_thresh, colormap_options=colormap_options)
        chosen_iou_all.extend(c_iou_list)
        chosen_lvl_list.extend(c_lvl)

        acc_num_img = lerf_localization(pred_masks_lvl, rgb_img, image_name, img_ann)
        acc_num += acc_num_img
    # # iou
    mean_iou_chosen = sum(chosen_iou_all) / len(chosen_iou_all)
    logger.info(f'trunc thresh: {mask_thresh}')
    logger.info(f"iou chosen: {mean_iou_chosen:.4f}")
    logger.info(f"chosen_lvl: \n{chosen_lvl_list}")

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
    seed_everything(seed_value = 42)

    parser = ArgumentParser(description="Evaluation for LERF-OVS Dataset")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--ae_ckpt_dir", type=str, default="autoencoder/ckpt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.4)
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

    # NOTE config setting
    dataset_name = args.dataset_name
    feat_dir = [os.path.join(f"data/lerf_ovs/{dataset_name}/output", f"{dataset_name}_{i}/train/ours_None/renders_language_npy") 
                for i in range(1, 4)]
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    json_folder = os.path.join("data/lerf_ovs/label", dataset_name)
    mask_thresh = args.mask_thresh if args.dataset_name != "ramen" else 0.6

    # NOTE logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    evaluate(feat_dir, output_path, ae_ckpt_path, json_folder, mask_thresh, args.encoder_dims, args.decoder_dims,
             logger)
