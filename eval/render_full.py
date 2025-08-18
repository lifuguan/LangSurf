from __future__ import annotations

import cv2
import open3d as o3d
import os
import glob
import random
from pathlib import Path
from argparse import ArgumentParser
import numpy as np
import torch
import torchvision
from kneed import KneeLocator
from tqdm import tqdm
import copy
import clip
import colormaps
from autoencoder.model import Autoencoder
from eval_utils import colormap_saving, points_inside_convex_hull, filter_gaussians_and_save
from scene.gaussian_model import GaussianModel
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors


def save_gs_feature(xyz_params, feat_params, feature_path):
    pcd = o3d.geometry.PointCloud()
    feat_params = torch.clamp(feat_params, min=-1, max=2)
    min_value = torch.min(feat_params)
    max_value = torch.max(feat_params)
    feat_params = (feat_params - min_value) / (max_value - min_value)
    pcd.points = o3d.utility.Vector3dVector(xyz_params.detach().cpu().numpy())
    pcd.colors = o3d.utility.Vector3dVector(feat_params.detach().cpu().numpy())
    o3d.io.write_point_cloud(feature_path, pcd)


def activate_stream(valid_map,
                    CLASS_LABELS,
                    output_path,
                    index,
                    args,
                    gaussians_list=None,
                    colormap_options=None,
                    image_list=None):
    mask_thresh = args.mask_thresh
    n_head, n_prompt, h, w = valid_map.shape
    # positive prompts
    for k in range(n_prompt):
        for i in range(n_head):
            output = valid_map[i][k]
            output = output - torch.min(output)
            output = output / (torch.max(output) + 1e-9)
            output = output * (1.0 - (-1.0)) + (-1.0)
            output = torch.clip(output, 0, 1)
            rgb_language = colormap_saving(output.unsqueeze(-1), colormap_options, mask_thresh=mask_thresh)
            if args.generate_mask:
                mask = output > mask_thresh
                mask_path = os.path.join(output_path, "masks_" + str(mask_thresh), CLASS_LABELS[k], str(i),
                                         f'{index}.png')
                torchvision.utils.save_image(mask.float(), mask_path)
                # Composited(time-consuming)
                p_i = torch.clip(output - 0.5, 0, 1).unsqueeze(-1)
                valid_composited = colormaps.apply_colormap(p_i / (p_i.max() + 1e-6),
                                                            colormaps.ColormapOptions("turbo"))
                mask = (output < mask_thresh).squeeze()
                rgb_img = cv2.imread(image_list[i][index])[..., ::-1]
                rgb_img = (rgb_img / 255.0).astype(np.float32)
                rgb_img = torch.from_numpy(rgb_img).to('cuda')
                valid_composited[mask, :] = rgb_img[mask, :] * 0.3
                output_path_compo = os.path.join(output_path, 'composited_' + str(mask_thresh), f'{CLASS_LABELS[k]}',
                                                 str(i), os.path.splitext(os.path.basename(image_list[i][index]))[0])
                output_path_compo = Path(output_path_compo)
                output_path_compo.parent.mkdir(exist_ok=True, parents=True)
                colormap_saving(valid_composited, colormap_options, save_path=output_path_compo)
            elif args.remove_object:
                output_dir = os.path.join(output_path, 'remove_' + str(mask_thresh))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                gaussians = gaussians_list[index]
                mask = (output > mask_thresh).squeeze()
                if mask.size == 0:
                    continue
                mask = points_inside_convex_hull(gaussians_list[index].get_xyz.detach(), mask, convex_hull=False,
                                                 outlier_factor=1.0)
                points = gaussians_list[index].get_xyz.detach().cpu().numpy()[mask.detach().cpu().numpy()]
                # DBSCAN
                min_samples = 10
                neighbors = NearestNeighbors(n_neighbors=min_samples)
                neighbors.fit(points)
                distances, indices = neighbors.kneighbors(points)
                k_distances = np.sort(distances[:, -1])
                kneedle = KneeLocator(range(len(k_distances)), k_distances, curve="convex", direction="increasing")
                initial_eps = k_distances[kneedle.elbow]
                dbscan_initial = DBSCAN(eps=initial_eps, min_samples=min_samples)
                labels = dbscan_initial.fit_predict(points)
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts = dict(zip(unique_labels, counts))
                sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
                top_labels = sorted_labels[:1] if len(sorted_labels) > 1 else sorted_labels
                colors = np.zeros((len(labels), 3))
                color_map = {top_labels[i]: color for i, color in
                             enumerate([[1, 0, 0]]) if i < len(top_labels)}
                for i, label in enumerate(labels):
                    colors[i] = color_map.get(label, [0, 0, 0])
                pcd_mask = o3d.geometry.PointCloud()
                pcd_mask.points = o3d.utility.Vector3dVector(points)
                pcd_mask.colors = o3d.utility.Vector3dVector(colors)
                save_path = os.path.join(output_dir,
                                         output_path.split('/')[-1] + f'_{CLASS_LABELS[k]}_{index}.ply')
                o3d.io.write_point_cloud(save_path, pcd_mask)
                bool_mask = np.zeros(len(gaussians_list[index].get_xyz.detach().cpu().numpy()), dtype=bool)
                cluster_points = points[labels == top_labels[0]]
                matching_indices = \
                    np.where(np.isin(gaussians_list[index].get_xyz.detach().cpu().numpy(), cluster_points).all(axis=1))[
                        0]
                bool_mask[matching_indices] = True
                mask = torch.from_numpy(bool_mask).to('cuda')
                mask_convex = points_inside_convex_hull(gaussians.get_xyz.detach(), mask, remove_outliers=False,
                                                        outlier_factor=1.0)
                mask3d = ~mask_convex
                remove_rgb_language = rgb_language[mask3d.cpu().numpy()]
                save_gaussians_path = os.path.join(output_path, 'point_cloud',
                                                   output_path.split('/')[
                                                       -1] + f'_remove_{CLASS_LABELS[k]}_{index}.ply')
                filter_gaussians_and_save(gaussians, mask3d, save_gaussians_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(gaussians.get_xyz[mask3d].detach().cpu().numpy())

                pcd.colors = o3d.utility.Vector3dVector(remove_rgb_language.reshape(remove_rgb_language.shape[0], 3))
                save_path = os.path.join(output_dir,
                                         output_path.split('/')[-1] + f'_remove_{CLASS_LABELS[k]}_{index}.ply')
                o3d.io.write_point_cloud(save_path, pcd)
                mask3d = mask_convex
                rgb_language = rgb_language[mask3d.cpu().numpy()]
                save_gaussians_path = os.path.join(output_path, 'point_cloud',
                                                   output_path.split('/')[-1] + f'_{CLASS_LABELS[k]}_{index}.ply')
                filter_gaussians_and_save(gaussians, mask3d, save_gaussians_path)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(gaussians.get_xyz[mask3d].detach().cpu().numpy())

                pcd.colors = o3d.utility.Vector3dVector(rgb_language.reshape(rgb_language.shape[0], 3))
                save_path = os.path.join(output_dir,
                                         output_path.split('/')[-1] + f'_{CLASS_LABELS[k]}_{index}.ply')
                o3d.io.write_point_cloud(save_path, pcd)
            elif args.ins_seg:
                mask = (output > mask_thresh).squeeze()
                output_dir = os.path.join(output_path, 'ins_seg_' + str(mask_thresh))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                mask = points_inside_convex_hull(gaussians_list[index].get_xyz.detach(), mask, convex_hull=False,
                                                 outlier_factor=1.0)
                mask = mask.detach().cpu().numpy()
                points_xyz = gaussians_list[index].get_xyz.detach().cpu().numpy()[mask]
                points_instance = gaussians_list[index].get_instance_feature.detach().cpu().numpy()[mask]
                points = np.concatenate([points_xyz, points_instance], axis=1)
                min_samples = 10
                dbscan_initial = DBSCAN(eps=0.16, min_samples=min_samples)
                labels = dbscan_initial.fit_predict(points)
                unique_labels, counts = np.unique(labels, return_counts=True)
                label_counts = dict(zip(unique_labels, counts))

                sorted_labels = sorted(label_counts, key=label_counts.get, reverse=True)
                top_labels = sorted_labels[:5] if len(sorted_labels) > 3 else sorted_labels
                colors = np.zeros((len(labels), 3))
                color_map = {top_labels[i]: color for i, color in
                             enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]) if i < len(top_labels)}
                for i, label in enumerate(labels):
                    colors[i] = color_map.get(label, [0, 0, 0])
                pcd_mask = o3d.geometry.PointCloud()
                pcd_mask.points = o3d.utility.Vector3dVector(points_xyz)
                pcd_mask.colors = o3d.utility.Vector3dVector(colors)
                save_path = os.path.join(output_dir,
                                         output_path.split('/')[-1] + f'_{CLASS_LABELS[k]}_{index}.ply')
                o3d.io.write_point_cloud(save_path, pcd_mask)
                mask_points = []
                gaussians = gaussians_list[index]
                for ins_idx, label in enumerate(top_labels[:3]):
                    bool_mask = np.zeros(len(gaussians_list[index].get_xyz.detach().cpu().numpy()), dtype=bool)
                    cluster_points = points[labels == label][:, :3]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(cluster_points)

                    # Oriented Bounding Box（OBB）
                    obb = pcd.get_oriented_bounding_box(robust=True)
                    obb.color = (1, 0, 0)
                    obb_corners = np.asarray(obb.get_box_points())
                    obb_lines = [
                        [0, 1], [1, 6], [6, 3], [3, 0],
                        [4, 5], [5, 2], [2, 7], [7, 4],
                        [0, 2], [1, 7], [6, 4], [3, 5]
                    ]
                    line_set = o3d.geometry.LineSet()
                    line_set.points = o3d.utility.Vector3dVector(obb_corners)
                    line_set.lines = o3d.utility.Vector2iVector(obb_lines)
                    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0]] * len(obb_lines))

                    output_obb_path = os.path.join(output_dir,
                                                   output_path.split('/')[-1] +f'_ins_{CLASS_LABELS[k]}_{index}_{str(ins_idx)}_obb.ply')
                    o3d.io.write_line_set(output_obb_path, line_set)
                    matching_indices = \
                        np.where(
                            np.isin(gaussians_list[index].get_xyz.detach().cpu().numpy(), cluster_points).all(axis=1))[
                            0]
                    bool_mask[matching_indices] = True
                    mask3d = ~points_inside_convex_hull(gaussians.get_xyz.detach(), bool_mask, remove_outliers=False)
                    save_gaussians_path = os.path.join(output_path, 'point_cloud',
                                                       output_path.split('/')[-1] +f'_ins_{CLASS_LABELS[k]}_{index}_{str(ins_idx)}.ply')
                    filter_gaussians_and_save(gaussians, mask3d, save_gaussians_path)
                    mask_points.append(mask3d.detach().cpu().numpy())
                mask3d = np.logical_and.reduce(mask_points)
                save_gaussians_path = os.path.join(output_path, 'point_cloud',
                                                   output_path.split('/')[-1] +f'_ins_{CLASS_LABELS[k]}_{index}_all.ply')
                filter_gaussians_and_save(gaussians, mask3d, save_gaussians_path)

            elif args.heatmap_3d:
                mask = (output > mask_thresh).squeeze()
                output_dir = os.path.join(output_path, 'heatmap_' + str(mask_thresh))
                output_dir_mask = os.path.join(output_path, 'heatmap_mask_' + str(mask_thresh))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    os.makedirs(output_dir_mask)
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(gaussians_list[index].get_xyz.detach().cpu().numpy())

                pcd.colors = o3d.utility.Vector3dVector(rgb_language.reshape(rgb_language.shape[0], 3))
                save_path = os.path.join(output_dir,
                                         output_path.split('/')[-1] + f'_{CLASS_LABELS[k]}_{index}.ply')
                o3d.io.write_point_cloud(save_path, pcd)
                pcd_mask = o3d.geometry.PointCloud()
                pcd_mask.points = o3d.utility.Vector3dVector(
                    gaussians_list[index].get_xyz.detach().cpu().numpy()[mask.detach().cpu().numpy()])

                pcd_mask.colors = o3d.utility.Vector3dVector(
                    rgb_language.reshape(rgb_language.shape[0], 3)[mask.detach().cpu().numpy()])
                save_path = os.path.join(output_dir_mask,
                                         output_path.split('/')[-1] + f'_{CLASS_LABELS[k]}_{index}.ply')
                o3d.io.write_point_cloud(save_path, pcd_mask)


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


def evaluate(output_path, ae_ckpt_path, args, feat_dir=None, gaussians_list=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    colormap_options = colormaps.ColormapOptions(
        colormap="turbo",
        normalize=True,
        colormap_min=-1.0,
        colormap_max=1.0,
    )
    clip.available_models()
    clip_model, preprocess = clip.load("ViT-L/14@336px")
    checkpoint = torch.load(ae_ckpt_path, map_location=device)
    model = Autoencoder(args.encoder_dims, args.decoder_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()
    if os.path.basename(output_path) == 'ramen':
        CLASS_LABELS = ['bowl', 'chopsticks', 'egg', 'kamaboko', 'napkin', 'nori', 'onion segments', 'paper napkin',
                        'plate', 'sake cup', 'wavy noodles', 'glass of water', 'spoon', 'hand']
    elif os.path.basename(output_path) == 'waldo_kitchen':
        CLASS_LABELS = ['knife', 'ottolenghi', 'pour-over vessel', 'Stainless steel pots', 'toaster', 'yellow desk',
                        'plastic ladle', 'pot', 'refrigerator', 'spatula',
                        'cabinet', 'ketchup', 'dark cup', 'frog cup', 'plate', 'sink', 'spoon', 'red cup',
                        'glass container', 'glass', 'vessel', 'power outlet', 'socket', 'bowl']
    elif os.path.basename(output_path) == 'teatime':
        CLASS_LABELS = ['stuffed bear', 'coffee mug', 'bag of cookies', 'sheep', 'apple', 'paper napkin', 'plate',
                        'tea in a glass', 'bear nose', 'three cookies', 'coffee', 'dall-e brand', 'food bag',
                        'packaging bag',
                        'cookie', 'spoon', 'bear', 'chair', 'table', 'desk', 'hooves', 'glass', 'yellow pouf']
    else:
        CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                        'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                        'bathtub']
    use_neg = True if 'neg' in output_path else False
    if args.generate_mask:
        compressed_sem_feats = np.zeros((len(feat_dir), len(glob.glob(os.path.join(feat_dir[0], '*.npy'))),
                                         *np.load(glob.glob(os.path.join(feat_dir[0], '*.npy'))[0]).shape),
                                        dtype=np.float32)
        for i in range(len(feat_dir)):
            feat_paths_lvl = sorted(glob.glob(os.path.join(feat_dir[i], '*.npy')),
                                    key=lambda file_name: int(
                                        os.path.basename(file_name).split(".npy")[0].replace('frame_', '')))
            for j, idx in enumerate(feat_paths_lvl):
                compressed_sem_feats[i][j] = np.load(idx)
        for folder_name in CLASS_LABELS:
            for i in range(len(feat_dir)):
                folder_path = os.path.join(output_path, "masks_" + str(args.mask_thresh), folder_name, str(i))
                os.makedirs(folder_path, exist_ok=True)
        image_path = [
            os.path.join(str(os.path.join(*feat_dir[-1].split('/')[:4])), f"{dataset_name}_{i}",
                         "train/ours_None/renders_rgb")
            for i in range(1, 4)
        ]

        image_list = [
            sorted(
                glob.glob(os.path.join(path, '*.jpg')),
                key=lambda file_name: int(os.path.basename(file_name).split(".jpg")[0].replace('frame_', ''))
            )
            for path in image_path
        ]
        for j in tqdm(range(compressed_sem_feats.shape[1])):
            sem_feat = compressed_sem_feats[:, j, ...]
            sem_feat = torch.from_numpy(sem_feat).float().to(device)

            with torch.no_grad():
                lvl, h, w, _ = sem_feat.shape
                restored_feat = model.decode(sem_feat.flatten(0, 2))
                restored_feat = restored_feat.view(lvl, h, w, -1)  # 3x832x1264x512

            pred_masks_lvl = get_relevancy(clip_model, restored_feat, CLASS_LABELS, use_neg)
            activate_stream(pred_masks_lvl, CLASS_LABELS, output_path, j, args,
                            gaussians_list=gaussians_list,
                            colormap_options=colormap_options, image_list=image_list)
    else:
        compressed_sem_feats_list = []
        for i in range(0, len(gaussians_list)):
            compressed_sem_feats_list.append(
                gaussians_list[i].get_language_feature.detach().cpu().numpy().reshape(1,
                                                                                      gaussians_list[
                                                                                          i].get_language_feature.shape[
                                                                                          0], 1, 3))
        for j in tqdm(range(len(compressed_sem_feats_list))):
            sem_feat = torch.from_numpy(compressed_sem_feats_list[j]).float().to(device)
            with torch.no_grad():
                lvl, h, w, _ = sem_feat.shape
                restored_feat = model.decode(sem_feat.flatten(0, 2))
                restored_feat = restored_feat.view(lvl, h, w, -1)
            pred_masks_lvl = get_relevancy(clip_model, restored_feat, CLASS_LABELS, use_neg)
            activate_stream(pred_masks_lvl, CLASS_LABELS, output_path, j, args,
                            gaussians_list=gaussians_list,
                            colormap_options=colormap_options)


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
    parser.add_argument("--ae_ckpt_dir", type=str, default="autoencoder/ckpt")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--mask_thresh", type=float, default=0.6)
    parser.add_argument("--generate_mask", action="store_true", help='Generates 2d masks')
    parser.add_argument("--remove_object", action="store_true", help='Remove 3D object')
    parser.add_argument("--heatmap_3d", action="store_true", help='3D query')
    parser.add_argument("--ins_seg", action="store_true", help='3D instance segmentation')
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
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    if "scene" in dataset_name:
        base_feat_dir = f"data/scannet/{dataset_name}/output"
    else:
        base_feat_dir = f"data/lerf_ovs/{dataset_name}/output"
    # 2D query
    if args.generate_mask:
        feat_dir = [os.path.join(base_feat_dir, dataset_name + f"_{i}", "train/ours_None/renders_language_npy") for i in
                    range(1, 4)]
        evaluate(output_path, ae_ckpt_path, args, feat_dir=feat_dir)

    # 3D query
    else:
        pc_hierarchy_dir = [
            os.path.join(base_feat_dir, dataset_name + f"_{i}", "point_cloud", "iteration_40000", "point_cloud.ply") for
            i in range(1, 4)]
        gaussians = GaussianModel(sh_degree=3)
        gaussians_list = []
        for i in range(len(pc_hierarchy_dir)):
            gaussians.load_ply(pc_hierarchy_dir[i])
            gaussians_list.append(copy.deepcopy(gaussians))

        language_path = os.path.join(str(output_path), "language")
        os.makedirs(language_path, exist_ok=True)
        instance_path = os.path.join(str(output_path), "instance")
        os.makedirs(instance_path, exist_ok=True)
        for i in range(len(gaussians_list)):
            save_gs_feature(gaussians_list[i].get_xyz, gaussians_list[i].get_language_feature,
                            os.path.join(language_path, f"language_{i}.ply"))
            save_gs_feature(gaussians_list[i].get_xyz, gaussians_list[i].get_instance_feature,
                            os.path.join(instance_path, f"instance_{i}.ply"))
        evaluate(output_path, ae_ckpt_path, args, gaussians_list=gaussians_list)
