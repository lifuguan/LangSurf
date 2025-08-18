import tensorflow as tf2
from utils.fusion_util import extract_openseg_img_feature
import tensorflow._api.v2.compat.v1 as tf
# cpu=tf.config.list_physical_devices("CPU")
# tf.config.set_visible_devices(cpu)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10240)])
    except RuntimeError as e:
        print(e)

import os
import random
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import cv2
import torch
import torch.nn.functional as F
from eval.evaluate_lerf_ovs import eval_gt_lerfdata

def create(image_list, data_list, np_image_string_list, save_folder):
    assert image_list is not None, "image_list must be provided to generate features"
    embed_size=768
    seg_maps = []
    total_lengths = []
    timer = 0
    img_embeds = torch.zeros((len(image_list), 300, embed_size))
    seg_maps = torch.zeros((len(image_list), 4, *image_list[0].shape[1:])) 
    mask_generator.predictor.model.to('cuda')

    for i, (img, np_image_string) in tqdm(enumerate(zip(image_list, np_image_string_list)), desc="Embedding images"):
        timer += 1
        img_embed, seg_map = _embed_clip_sam_tiles(img.unsqueeze(0).to('cuda'), np_image_string, sam_encoder, embed_size)

        lengths = [len(v) for k, v in img_embed.items()]
        total_length = sum(lengths)
        total_lengths.append(total_length)
        
        if total_length > img_embeds.shape[1]:
            pad = total_length - img_embeds.shape[1]
            img_embeds = torch.cat([
                img_embeds,
                torch.zeros((len(image_list), pad, embed_size))
            ], dim=1)

        img_embed = torch.cat([v for k, v in img_embed.items()], dim=0)
        assert img_embed.shape[0] == total_length
        img_embeds[i, :total_length] = img_embed
        
        seg_map_tensor = []
        lengths_cumsum = lengths.copy()
        for j in range(1, len(lengths)):
            lengths_cumsum[j] += lengths_cumsum[j-1]
        for j, (k, v) in enumerate(seg_map.items()):
            if j == 0:
                seg_map_tensor.append(torch.from_numpy(v))
                continue
            assert v.max() == lengths[j] - 1, f"{j}, {v.max()}, {lengths[j]-1}"
            v[v != -1] += lengths_cumsum[j-1]
            seg_map_tensor.append(torch.from_numpy(v))
        seg_map = torch.stack(seg_map_tensor, dim=0)
        seg_maps[i] = seg_map

    mask_generator.predictor.model.to('cpu')
        
    for i in range(img_embeds.shape[0]):
        save_path = os.path.join(save_folder, data_list[i].split('.')[0])
        assert total_lengths[i] == int(seg_maps[i].max() + 1)
        curr = {
            'feature': img_embeds[i, :total_lengths[i]],
            'seg_maps': seg_maps[i]
        }
        sava_numpy(save_path, curr)

def sava_numpy(save_path, data):
    save_path_s = save_path + '_s.npy'
    save_path_f = save_path + '_f.npy'
    np.save(save_path_s, data['seg_maps'].numpy())
    np.save(save_path_f, data['feature'].numpy())

def _embed_clip_sam_tiles(aug_img, np_image_string, sam_encoder, embed_size):
    device = aug_img.device
    seg_image, seg_map = sam_encoder(aug_img)

    text_emb = tf.zeros([1, 1, embed_size])
    hcma_embed = extract_openseg_img_feature(aug_img, np_image_string, openseg_model, text_emb, img_size=(aug_img.shape[2],aug_img.shape[3])).to(device)
    hcma_embeds = {}
    for mode in ['default', 's', 'm', 'l']:
        unique_ids = np.unique(seg_map[mode])
        openseg_embeds_pooled = []
        for unique_id in unique_ids:
            if unique_id == -1:
                continue
            seg_map_mask = seg_map[mode] == unique_id
            openseg_embed_masked = hcma_embed.permute(1,2,0)[seg_map_mask]
            openseg_embed_pooled = F.adaptive_avg_pool1d(openseg_embed_masked.t().unsqueeze(0), 1)[0].t()
            openseg_embeds_pooled.append(openseg_embed_pooled)
        openseg_embeds_pooled = torch.cat(openseg_embeds_pooled)
        openseg_embeds_pooled /= openseg_embeds_pooled.norm(dim=-1, keepdim=True)
        hcma_embeds[mode] = openseg_embeds_pooled.detach().cpu().half()
    return hcma_embeds, seg_map

def get_seg_img(mask, image):
    image = image.copy()
    image[mask['segmentation']==0] = np.array([0, 0,  0], dtype=np.uint8)
    x,y,w,h = np.int32(mask['bbox'])
    seg_img = image[y:y+h, x:x+w, ...]
    return seg_img

def pad_img(img):
    h, w, _ = img.shape
    l = max(w,h)
    pad = np.zeros((l,l,3), dtype=np.uint8)
    if h > w:
        pad[:,(h-w)//2:(h-w)//2 + w, :] = img
    else:
        pad[(w-h)//2:(w-h)//2 + h, :, :] = img
    return pad

def filter(keep: torch.Tensor, masks_result) -> None:
    keep = keep.int().cpu().numpy()
    result_keep = []
    for i, m in enumerate(masks_result):
        if i in keep: result_keep.append(m)
    return result_keep

def mask_nms(masks, scores, iou_thr=0.7, score_thr=0.1, inner_thr=0.2, **kwargs):
    """
    Perform mask non-maximum suppression (NMS) on a set of masks based on their scores.
    
    Args:
        masks (torch.Tensor): has shape (num_masks, H, W)
        scores (torch.Tensor): The scores of the masks, has shape (num_masks,)
        iou_thr (float, optional): The threshold for IoU.
        score_thr (float, optional): The threshold for the mask scores.
        inner_thr (float, optional): The threshold for the overlap rate.
        **kwargs: Additional keyword arguments.
    Returns:
        selected_idx (torch.Tensor): A tensor representing the selected indices of the masks after NMS.
    """

    scores, idx = scores.sort(0, descending=True)
    num_masks = idx.shape[0]
    
    masks_ord = masks[idx.view(-1), :]
    masks_area = torch.sum(masks_ord, dim=(1, 2), dtype=torch.float)

    iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    inner_iou_matrix = torch.zeros((num_masks,) * 2, dtype=torch.float, device=masks.device)
    for i in range(num_masks):
        for j in range(i, num_masks):
            intersection = torch.sum(torch.logical_and(masks_ord[i], masks_ord[j]), dtype=torch.float)
            union = torch.sum(torch.logical_or(masks_ord[i], masks_ord[j]), dtype=torch.float)
            iou = intersection / union
            iou_matrix[i, j] = iou
            # select mask pairs that may have a severe internal relationship
            if intersection / masks_area[i] < 0.5 and intersection / masks_area[j] >= 0.85:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[i, j] = inner_iou
            if intersection / masks_area[i] >= 0.85 and intersection / masks_area[j] < 0.5:
                inner_iou = 1 - (intersection / masks_area[j]) * (intersection / masks_area[i])
                inner_iou_matrix[j, i] = inner_iou

    iou_matrix.triu_(diagonal=1)
    iou_max, _ = iou_matrix.max(dim=0)
    inner_iou_matrix_u = torch.triu(inner_iou_matrix, diagonal=1)
    inner_iou_max_u, _ = inner_iou_matrix_u.max(dim=0)
    inner_iou_matrix_l = torch.tril(inner_iou_matrix, diagonal=1)
    inner_iou_max_l, _ = inner_iou_matrix_l.max(dim=0)
    
    keep = iou_max <= iou_thr
    keep_conf = scores > score_thr
    keep_inner_u = inner_iou_max_u <= 1 - inner_thr
    keep_inner_l = inner_iou_max_l <= 1 - inner_thr
    
    # If there are no masks with scores above threshold, the top 3 masks are selected
    if keep_conf.sum() == 0:
        index = scores.topk(3).indices
        keep_conf[index, 0] = True
    if keep_inner_u.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_u[index, 0] = True
    if keep_inner_l.sum() == 0:
        index = scores.topk(3).indices
        keep_inner_l[index, 0] = True
    keep *= keep_conf
    keep *= keep_inner_u
    keep *= keep_inner_l

    selected_idx = idx[keep]
    return selected_idx

def masks_update(*args, **kwargs):
    # remove redundant masks based on the scores and overlap rate between masks
    masks_new = ()
    for masks_lvl in (args):
        seg_pred =  torch.from_numpy(np.stack([m['segmentation'] for m in masks_lvl], axis=0))
        iou_pred = torch.from_numpy(np.stack([m['predicted_iou'] for m in masks_lvl], axis=0))
        stability = torch.from_numpy(np.stack([m['stability_score'] for m in masks_lvl], axis=0))

        scores = stability * iou_pred
        keep_mask_nms = mask_nms(seg_pred, scores, **kwargs)
        masks_lvl = filter(keep_mask_nms, masks_lvl)

        masks_new += (masks_lvl,)
    return masks_new

def sam_encoder(image):
    if image.device != "cpu":
        image = image.cpu()
    image = cv2.cvtColor(image[0].permute(1,2,0).numpy().astype(np.uint8), cv2.COLOR_BGR2RGB)
    # pre-compute masks
    masks_default, masks_s, masks_m, masks_l = mask_generator.generate(image)
    # pre-compute postprocess
    masks_default, masks_s, masks_m, masks_l = \
        masks_update(masks_default, masks_s, masks_m, masks_l, iou_thr=0.8, score_thr=0.7, inner_thr=0.5)
    
    def mask2segmap(masks, image):
        seg_map = -np.ones(image.shape[:2], dtype=np.int32)
        for i in range(len(masks)):
            seg_map[masks[i]['segmentation']] = i

        return seg_map

    def continous_seg_map(mask):
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask)
        # Get unique values in the mask and sort them
        unique_values = mask.unique(sorted=True)  # Returns [1, 2, 3, 5, 9]

        # Create a mapping from non-continuous values to continuous values (starting from -1)
        mapping = {val.item(): idx - 1 for idx, val in enumerate(unique_values)}

        # Create a new mask and map non-continuous values to continuous values
        continuous_mask = mask.clone()  # Clone the original mask to preserve structure
        for old_value, new_value in mapping.items():
            continuous_mask[mask == old_value] = new_value

        return continuous_mask.numpy()

    seg_images, seg_maps = {}, {}
    seg_maps['default'] = continous_seg_map(mask2segmap(masks_default, image))
    if len(masks_s) != 0:
        seg_maps['s'] = continous_seg_map(mask2segmap(masks_s, image))
    if len(masks_m) != 0:
        seg_maps['m'] = continous_seg_map(mask2segmap(masks_m, image))
    if len(masks_l) != 0:
        seg_maps['l'] = continous_seg_map(mask2segmap(masks_l, image))
    
    # 0:default 1:s 2:m 3:l
    return seg_images, seg_maps

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


if __name__ == '__main__':
    seed_num = 42
    seed_everything(seed_num)

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument("--json_folder", type=str, default=None)
    parser.add_argument('--resolution', type=int, default=-1)
    parser.add_argument('--sam_ckpt_path', type=str, default="model_zoo/sam_vit_h_4b8939.pth")
    args = parser.parse_args()
    torch.set_default_dtype(torch.float32)

    dataset_path = args.dataset_path
    sam_ckpt_path = args.sam_ckpt_path

    if args.json_folder is not None:
        json_folder = os.path.join(args.json_folder, args.scene)
        _, _, image_paths = eval_gt_lerfdata(Path(json_folder))
        img_folder = os.path.join(dataset_path, 'images')
        data_lists = [image_path.split('/')[-1] for image_path in image_paths]
        # data_lists = data_lists[0:2]  # for debug
        data_lists.sort()
    else:
        if args.resolution > 1: 
            img_folder = os.path.join(dataset_path, f'images_{args.resolution}')
        else:
            img_folder = os.path.join(dataset_path, 'images')
        data_lists = os.listdir(img_folder)
        data_lists.sort()


    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to('cuda')
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        points_per_batch=64,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    saved_model_dir = './model_zoo/openseg_exported_clip' #@param {type:"string"}
    openseg_model = tf2.saved_model.load(saved_model_dir, tags=[tf.saved_model.tag_constants.SERVING],)
        
    img_list, np_image_string_list = [], []
    WARNED = False
    for data_path in data_lists:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)
        with tf.gfile.GFile(image_path, 'rb') as f:
            np_image_string = np.array([f.read()])

        orig_w, orig_h = image.shape[1], image.shape[0]
            
        scale = 1
        resolution = (int( orig_w  / scale), int(orig_h / scale))
        
        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
        np_image_string_list.append(np_image_string)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    save_folder = os.path.join(dataset_path, 'hcma_features')
    os.makedirs(save_folder, exist_ok=True)
    create(imgs, data_lists, np_image_string_list, save_folder)