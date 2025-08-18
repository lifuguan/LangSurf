import numpy as np
import torch
import clip
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import mediapy as media
import cv2
import colormaps
import logging
import copy
from pathlib import Path
from scipy.spatial import ConvexHull, Delaunay, cKDTree

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


def filter_gaussians_and_save(gaussians, mask3d, save_gaussians_path):
    gaussians_temp = copy.deepcopy(gaussians)
    gaussians_temp._features_dc = gaussians_temp._features_dc[mask3d]
    gaussians_temp._features_rest = gaussians_temp._features_rest[mask3d]
    gaussians_temp._language_feature = gaussians_temp._language_feature[mask3d]
    gaussians_temp._instance_feature = gaussians_temp._instance_feature[mask3d]
    gaussians_temp._opacity = gaussians_temp._opacity[mask3d]
    gaussians_temp._rotation = gaussians_temp._rotation[mask3d]
    gaussians_temp._scaling = gaussians_temp._scaling[mask3d]
    gaussians_temp._xyz = gaussians_temp._xyz[mask3d]
    gaussians_temp.save_ply( save_gaussians_path, include_feature=True)

def expand_convex_hull(points):
    # Compute the convex hull of the given points
    hull = ConvexHull(points)

    # Get the minimum and maximum values along each axis
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    z_min, z_max = np.min(points[:, 2]), np.max(points[:, 2])

    # Calculate the expansion distance for each axis
    expansion_distance = np.array([
        (x_max - x_min) * 0.05,  # Expansion along the x-axis
        (y_max - y_min) * 0.05,  # Expansion along the y-axis
        (z_max - z_min) * 0.05  # Expansion along the z-axis
    ])  # The scaling factor can be adjusted

    # Get the vertices of the convex hull
    hull_points = points[hull.vertices]

    # Initialize a list to store the expanded points
    expanded_points = []

    # Iterate over each simplex (triangle) in the convex hull
    for simplex in hull.simplices:
        p1 = points[simplex[0]]
        p2 = points[simplex[1]]
        p3 = points[simplex[2]]

        # Calculate the normal vector of the simplex
        edge1 = p2 - p1
        edge2 = p3 - p1
        normal_vector = np.cross(edge1, edge2)
        normal_length = np.linalg.norm(normal_vector)

        if normal_length > 0:
            normal_vector /= normal_length  # Normalize the normal vector

            # Expand each vertex of the simplex along the normal direction
            expanded_p1 = p1 + normal_vector * np.linalg.norm(expansion_distance)
            expanded_p2 = p2 + normal_vector * np.linalg.norm(expansion_distance)
            expanded_p3 = p3 + normal_vector * np.linalg.norm(expansion_distance)

            expanded_points.append(expanded_p1)
            expanded_points.append(expanded_p2)
            expanded_points.append(expanded_p3)

    return np.array(expanded_points), hull

def points_inside_convex_hull(point_cloud, mask, remove_outliers=True, convex_hull=True, outlier_factor=1.0):
    """
    Given a point cloud and a mask indicating a subset of points, this function computes the convex hull of the
    subset of points and then identifies all points from the original point cloud that are inside this convex hull.

    Parameters:
    - point_cloud (torch.Tensor): A tensor of shape (N, 3) representing the point cloud.
    - mask (torch.Tensor): A tensor of shape (N,) indicating the subset of points to be used for constructing the convex hull.
    - remove_outliers (bool): Whether to remove outliers from the masked points before computing the convex hull. Default is True.
    - outlier_factor (float): The factor used to determine outliers based on the IQR method. Larger values will classify more points as outliers.

    Returns:
    - inside_hull_tensor_mask (torch.Tensor): A mask of shape (N,) with values set to True for the points inside the convex hull
                                              and False otherwise.
    """

    # Extract the masked points from the point cloud
    masked_points = point_cloud[mask].cpu().numpy()
    masked_points = masked_points[~np.isnan(masked_points).any(axis=1)]
    # Remove outliers if the option is selected
    if remove_outliers:
        Q1 = np.percentile(masked_points, 25, axis=0)
        Q3 = np.percentile(masked_points, 75, axis=0)
        IQR = Q3 - Q1
        outlier_mask = (masked_points < (Q1 - outlier_factor * IQR)) | (masked_points > (Q3 + outlier_factor * IQR))
        filtered_masked_points = masked_points[~np.any(outlier_mask, axis=1)]
    else:
        filtered_masked_points = masked_points
    if convex_hull:
        filtered_masked_points, _ = expand_convex_hull(filtered_masked_points)

        # Compute the Delaunay triangulation of the filtered masked points
        delaunay = Delaunay(filtered_masked_points)

        # Determine which points from the original point cloud are inside the convex hull
        points_inside_hull_mask = delaunay.find_simplex(point_cloud.cpu().numpy()) >= 0

        # Convert the numpy mask back to a torch tensor and return
        inside_hull_tensor_mask = torch.tensor(points_inside_hull_mask, device='cuda')
        return inside_hull_tensor_mask
    else:
        filtered_mask = np.zeros(len(point_cloud), dtype=bool)

        matching_indices = np.where(np.isin(point_cloud.cpu().numpy(), filtered_masked_points).all(axis=1))[0]

        filtered_mask[matching_indices] = True
        return torch.tensor(filtered_mask, dtype=torch.bool)



def show_points(coords, labels, ax, marker_size=100):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='firebrick', marker='o',
               s=marker_size, edgecolor='black', linewidth=2.5, alpha=1)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='o',
               s=marker_size, edgecolor='black', linewidth=1.5, alpha=1)


def show_box(boxes, ax, color=None):
    if type(color) == str and color == 'random':
        color = np.random.random(3)
    elif color is None:
        color = 'black'
    for box in boxes.reshape(-1, 4):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor=color, facecolor=(0, 0, 0, 0), lw=4,
                                   capstyle='round', joinstyle='round', linestyle='dotted'))


def show_result(image, point, bbox, save_path):
    plt.figure()
    plt.imshow(image)
    rect = patches.Rectangle((0, 0), image.shape[1] - 1, image.shape[0] - 1, linewidth=0, edgecolor='none',
                             facecolor='white', alpha=0.3)
    plt.gca().add_patch(rect)
    input_point = point.reshape(1, -1)
    input_label = np.array([1])
    show_points(input_point, input_label, plt.gca())
    show_box(bbox, plt.gca())
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.0, dpi=200)
    plt.close()


def smooth(mask):
    h, w = mask.shape[:2]
    im_smooth = mask.copy()
    scale = 3
    for i in range(h):
        for j in range(w):
            square = mask[max(0, i - scale): min(i + scale + 1, h - 1),
                     max(0, j - scale): min(j + scale + 1, w - 1)]
            im_smooth[i, j] = np.argmax(np.bincount(square.reshape(-1)))
    return im_smooth


def colormap_saving(image: torch.Tensor, colormap_options, mask_thresh=None, save_path=None):
    """
    if image's shape is (h, w, 1): draw colored relevance map;
    if image's shape is (h, w, 3): return directively;
    if image's shape is (h, w, c): execute PCA and transform it into (h, w, 3).
    """
    output_image = (
        colormaps.apply_colormap(
            image=image,
            mask_thresh=mask_thresh,
            colormap_options=colormap_options,
        ).cpu().numpy()
    )
    if save_path is not None:
        media.write_image(save_path.with_suffix(".png"), output_image, fmt="png")
    return output_image


def vis_mask_save(mask, save_path: Path = None):
    mask_save = mask.copy()
    mask_save[mask == 1] = 255
    save_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(save_path), mask_save)


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def stack_mask(mask_base, mask_add):
    mask = mask_base.copy()
    mask[mask_add != 0] = 1
    return mask

def create_color_palette():
    return [
        (0, 0, 0),
        (174, 199, 232),  # wall
        (152, 223, 138),  # floor
        (31, 119, 180),  # cabinet
        (255, 187, 120),  # bed
        (188, 189, 34),  # chair
        (140, 86, 75),  # sofa
        (255, 152, 150),  # table
        (214, 39, 40),  # door
        (197, 176, 213),  # window
        (148, 103, 189),  # bookshelf
        (196, 156, 148),  # picture
        (23, 190, 207),  # counter
        (178, 76, 76),
        (247, 182, 210),  # desk
        (66, 188, 102),
        (219, 219, 141),  # curtain
        (140, 57, 197),
        (202, 185, 52),
        (51, 176, 203),
        (200, 54, 131),
        (92, 193, 61),
        (78, 71, 183),
        (172, 114, 82),
        (255, 127, 14),  # refrigerator
        (91, 163, 138),
        (153, 98, 156),
        (140, 153, 101),
        (158, 218, 229),  # shower curtain
        (100, 125, 154),
        (178, 127, 135),
        (120, 185, 128),
        (146, 111, 194),
        (44, 160, 44),  # toilet
        (112, 128, 144),  # sink
        (96, 207, 209),
        (227, 119, 194),  # bathtub
        (213, 92, 176),
        (94, 106, 211),
        (82, 84, 163),  # otherfurn
        (100, 85, 144)
    ]

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
