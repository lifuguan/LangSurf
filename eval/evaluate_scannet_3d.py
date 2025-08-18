from __future__ import annotations
import open3d as o3d
import os
import random
from argparse import ArgumentParser
import numpy as np
import torch
import time
import logging
from tqdm import tqdm
import copy
import clip

import colormaps
from scene.gaussian_model import GaussianModel
from autoencoder.model import Autoencoder
from eval_utils import colormap_saving, get_logger, get_relevancy

def activate_stream(valid_map,
                    CLASS_LABELS_ALL,
                    output_path,
                    index,
                    mask_thresh,
                    CLASS_LABELS,
                    gaussians_list=None,
                    colormap_options=None):
    n_head, n_prompt, h, w = valid_map.shape
    assert n_head == 1, "Something wrong here"
    score_lvl, i = torch.zeros((n_prompt,)), 0
    # positive prompts
    for k in tqdm(range(n_prompt), desc=f"Evaluating 3D point clouds in level {index}"):
        if CLASS_LABELS_ALL[k] not in CLASS_LABELS:
            continue
        output = valid_map[0][k]
        output = output - torch.min(output)
        output = output / (torch.max(output) + 1e-9)
        output = output * (1.0 - (-1.0)) + (-1.0)
        output = torch.clip(output, 0, 1)
        rgb_language = colormap_saving(output.unsqueeze(-1), colormap_options, mask_thresh=mask_thresh)

        output_dir = os.path.join(output_path, '3dseg_' + str(mask_thresh))
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        gaussians = gaussians_list[index]
        mask3d = (output > mask_thresh).squeeze()
        if mask3d.size == 0:
            continue
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(gaussians.get_xyz[mask3d].detach().cpu().numpy())
        pred_pcd.colors = o3d.utility.Vector3dVector(rgb_language[mask3d.cpu().numpy(), ...].reshape(-1, 3))
        save_path = os.path.join(output_dir, f'3dseg_{CLASS_LABELS_ALL[k]}_{index}.ply')
        o3d.io.write_point_cloud(save_path, pred_pcd)

        score_lvl[i] = valid_map[0][k].max()
        i += 1
    return score_lvl



def evaluate(output_path, ae_ckpt_path, encoder_hidden_dims,
             decoder_hidden_dims,
             mask_thresh, logger=None, gaussians_list=None, gt_label_path=None, dataset_name=None):
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
    model = Autoencoder(encoder_hidden_dims, decoder_hidden_dims).to(device)
    model.load_state_dict(checkpoint)
    model.eval()

    CLASS_LABELS_ALL = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf',
                    'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink',
                    'bathtub']
    with open(os.path.join(gt_label_path, 'classes.txt'), 'r') as f:
        lines = f.readlines()[0]
        CLASS_LABELS = lines.split(' ')
    ################## segment 3d point cloud with text prompts ##################
    cmp_lang_feat_lvls = []
    for i in range(0, len(gaussians_list)):
        cmp_lang_feat_lvls.append(
            gaussians_list[i].get_language_feature.detach().cpu().numpy().reshape(1,
                gaussians_list[i].get_language_feature.shape[0], 1, 3))

    use_neg = True if 'neg' in output_path else False
    score_lvls = []
    for i in range(len(cmp_lang_feat_lvls)):
        sem_feat = torch.from_numpy(cmp_lang_feat_lvls[i]).float().to(device)
        with torch.no_grad():
            lvl, h, w, _ = sem_feat.shape
            restored_feat = model.decode(sem_feat.flatten(0, 2))
            restored_feat = restored_feat.view(lvl, h, w, -1)
        pred_masks_lvl = get_relevancy(clip_model, restored_feat, CLASS_LABELS_ALL, use_neg)
        score_lvl = activate_stream(pred_masks_lvl, CLASS_LABELS_ALL, output_path, i, mask_thresh,CLASS_LABELS,
                                    gaussians_list=gaussians_list,
                                    colormap_options=colormap_options)
        score_lvls.append(score_lvl)

    chose_lvls = []
    for i in range(len(score_lvls[0])):
        chose_lvls.append(np.argmax([score_lvls[0][i], score_lvls[1][i], score_lvls[2][i]]))

    ################## perfrom Semantic F1-score ##################
    pred_dir = os.path.join(output_path, '3dseg_' + str(mask_thresh))
    sf_score = []
    for i in range(len(CLASS_LABELS)):
        pred_pc = o3d.io.read_point_cloud(os.path.join(pred_dir, f"3dseg_{CLASS_LABELS[i]}_{chose_lvls[i]}.ply"))
        gt_pc = o3d.io.read_point_cloud(os.path.join(gt_label_path, f"{dataset_name}_vh_clean_2.labels_{CLASS_LABELS[i]}.ply"))
        accuracy, Completion, precision, recall, fscore, _, _, _, _ = EvaluateHisto(
            pred_pc, gt_pc, threshold = 0.05, plot_stretch = 1)
        logger.info(f"{'Accuracy':<12} {'Completion':<15} {'Precision':<12} {'Recall':<8} {'F-Score':<8}")
        logger.info(f"{accuracy:<12.4f} {Completion:<15.4f} {precision:<12.4f} {recall:<8.4f} {fscore:<8.4f}")
        sf_score.append(fscore)
    logger.info(f"mean Semantic F-score: {np.mean(sf_score):.4f}")

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


def EvaluateHisto(
        source,
        target,
        threshold,
        plot_stretch,
):
    # print("[compute_point_cloud_to_point_cloud_distance]")
    distance1 = source.compute_point_cloud_distance(target)
    # print("[compute_point_cloud_to_point_cloud_distance]")
    distance2 = target.compute_point_cloud_distance(source)

    [
        accuracy,
        Completion,
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = get_f1_score_histo2(threshold, plot_stretch, distance1,
                            distance2)

    return [
        accuracy,
        Completion,
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ]


def get_f1_score_histo2(threshold,
                        plot_stretch,
                        distance1,
                        distance2,
                        ):
    # print("[get_f1_score_histo2]")
    dist_threshold = threshold
    accuracy = np.mean(distance1)
    Completion = np.mean(distance2)
    if len(distance1) and len(distance2):

        recall = float(sum(d < threshold for d in distance2)) / float(
            len(distance2))
        precision = float(sum(d < threshold for d in distance1)) / float(
            len(distance1))
        fscore = 2 * recall * precision / (recall + precision)
        num = len(distance1)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_source = np.histogram(distance1, bins)
        cum_source = np.cumsum(hist).astype(float) / num

        num = len(distance2)
        bins = np.arange(0, dist_threshold * plot_stretch, dist_threshold / 100)
        hist, edges_target = np.histogram(distance2, bins)
        cum_target = np.cumsum(hist).astype(float) / num

    else:
        precision = 0
        recall = 0
        fscore = 0
        edges_source = np.array([0])
        cum_source = np.array([0])
        edges_target = np.array([0])
        cum_target = np.array
    return accuracy, Completion, precision, recall, fscore, edges_source, cum_source, edges_target, cum_target


if __name__ == "__main__":
    seed_everything(seed_value = 42)

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
    feat_dir = [os.path.join(f"data/scannet/{dataset_name}/output", f"{dataset_name}_{i}/test/ours_None/renders_language_npy") for i in
                range(1, 4)]
    output_path = os.path.join(args.output_dir, dataset_name)
    ae_ckpt_path = os.path.join(args.ae_ckpt_dir, dataset_name, "best_ckpt.pth")
    gt_label_path = os.path.join(f"data/scannet/{dataset_name}", "gt_ply")
    mask_thresh = args.mask_thresh

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(str(output_path), f'{timestamp}.log')
    logger = get_logger(f'{dataset_name}', log_file=log_file, log_level=logging.INFO)

    ###### load Gaussian models ######
    max_iteration = os.listdir(os.path.join(f"data/scannet/{dataset_name}/output", f"{dataset_name}_1/point_cloud"))[-1].split("_")[-1]
    pc_hierarchy_dir = [
        os.path.join(f"data/scannet/{dataset_name}/output", f"{dataset_name}_{i}/point_cloud/iteration_{max_iteration}/point_cloud.ply") for
        i in range(1, 4)]
    gaussians = GaussianModel(sh_degree = 3)
    gaussians_list = []
    for i in range(len(pc_hierarchy_dir)):
        gaussians.load_ply(pc_hierarchy_dir[i])
        gaussians_list.append(copy.deepcopy(gaussians))

    evaluate(output_path, ae_ckpt_path, args.encoder_dims, args.decoder_dims, mask_thresh, 
             gaussians_list=gaussians_list, logger=logger, gt_label_path=gt_label_path, dataset_name=dataset_name)

