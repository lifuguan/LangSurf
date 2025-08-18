import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
import open3d as o3d
import copy


def post_process_mesh(mesh, cluster_to_keep=3):
    """
    Post-process a mesh to filter out floaters and disconnected parts
    """
    print("post processing the mesh to have {} clusterscluster_to_kep".format(cluster_to_keep))
    mesh_0 = copy.deepcopy(mesh)
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, cluster_area = (mesh_0.cluster_connected_triangles())

    triangle_clusters = np.asarray(triangle_clusters)
    cluster_n_triangles = np.asarray(cluster_n_triangles)
    cluster_area = np.asarray(cluster_area)
    n_cluster = np.sort(cluster_n_triangles.copy())[-cluster_to_keep]
    n_cluster = max(n_cluster, 50)  # filter meshes smaller than 50
    triangles_to_remove = cluster_n_triangles[triangle_clusters] < n_cluster
    mesh_0.remove_triangles_by_mask(triangles_to_remove)
    mesh_0.remove_unreferenced_vertices()
    mesh_0.remove_degenerate_triangles()
    print("num vertices raw {}".format(len(mesh.vertices)))
    print("num vertices post {}".format(len(mesh_0.vertices)))
    return mesh_0


def render_set(model_path, name, iteration, views, scene, gaussians, pipeline, background, args,
               app_model=None, max_depth=5.0, volume=None, volume_feature=None, use_depth_filter=False, mask_path=None):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_rgb")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_depth_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth_npy")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_depth_npy_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    depths_tsdf_fusion = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        out = render(view, gaussians, pipeline, background, app_model=app_model)
        rendering = out["render"].clamp(0.0, 1.0)
        _, H, W = rendering.shape

        depth = out["plane_depth"].squeeze()
        depth_tsdf = depth.clone()
        depth = depth.detach().cpu().numpy()
        depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
        depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
        depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)

        normal = out["rendered_normal"].permute(1, 2, 0)
        normal = normal / (normal.norm(dim=-1, keepdim=True) + 1.0e-8)
        normal = normal.detach().cpu().numpy()
        normal = ((normal + 1) * 127.5).astype(np.uint8).clip(0, 255)
        if args.include_feature:
            render_language_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_language")
            render_instance_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_instance")
            gts_language_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_language")
            render_language_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                                                    "renders_language_npy")
            render_instance_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration),
                                                    "renders_instance_npy")
            gts_language_npy_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_language_npy")
            makedirs(render_language_path, exist_ok=True)
            makedirs(gts_language_path, exist_ok=True)
            makedirs(render_language_npy_path, exist_ok=True)
            makedirs(gts_language_npy_path, exist_ok=True)
            makedirs(render_instance_path, exist_ok=True)
            makedirs(render_instance_npy_path, exist_ok=True)
            language_feature = out["language_feature"]
            instance_feature = out["instance_feature"]
            gt_language, _, _ = view.get_language_feature(args.lf_path,
                                                          feature_level=args.feature_level)
            if args.normalized:
                # language_feature = torch.clamp(language_feature, min=-1, max=2)
                min_value = torch.min(language_feature)
                max_value = torch.max(language_feature)
                normalized_language_feature = (language_feature - min_value) / (max_value - min_value)
                torchvision.utils.save_image(normalized_language_feature,
                                             os.path.join(render_language_path, view.image_name + ".png"))
                # instance_feature = torch.clamp(instance_feature, min=-1, max=2)
                min_value = torch.min(instance_feature)
                max_value = torch.max(instance_feature)
                normalized_instance_feature = (instance_feature - min_value) / (max_value - min_value)
                torchvision.utils.save_image(normalized_instance_feature,
                                             os.path.join(render_instance_path, view.image_name + ".png"))
                # gt_language = torch.clamp(gt_language, min=-1, max=2)
                min_value = torch.min(gt_language)
                max_value = torch.max(gt_language)
                normalized_gt_language = (gt_language - min_value) / (max_value - min_value)
                torchvision.utils.save_image(normalized_gt_language,
                                             os.path.join(gts_language_path, view.image_name + ".png"))
            else:
                torchvision.utils.save_image(language_feature,
                                             os.path.join(render_language_path, view.image_name + ".png"))
                torchvision.utils.save_image(instance_feature,
                                             os.path.join(render_instance_path, view.image_name + ".png"))
                torchvision.utils.save_image(gt_language, os.path.join(gts_language_path, view.image_name + ".png"))
            np.save(os.path.join(render_language_npy_path, view.image_name + ".npy"),
                    language_feature.permute(1, 2, 0).cpu().numpy())
            np.save(os.path.join(gts_language_npy_path, view.image_name + ".npy"),
                    gt_language.permute(1, 2, 0).cpu().numpy())
            np.save(os.path.join(render_instance_npy_path, view.image_name + ".npy"),
                    instance_feature.permute(1, 2, 0).cpu().numpy())
            torchvision.utils.save_image(instance_feature, os.path.join(render_instance_path, view.image_name + ".png"))
        if name == 'test':
            gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_rgb")
            makedirs(gts_path, exist_ok=True)
            torchvision.utils.save_image(gt.clamp(0.0, 1.0), os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (
                    rendering.permute(1, 2, 0).clamp(0, 1)[:, :, [2, 1, 0]] * 255).detach().cpu().numpy().astype(
                np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)
        cv2.imwrite(os.path.join(render_depth_path, view.image_name + ".jpg"), depth_color)
        np.save(os.path.join(render_depth_npy_path, view.image_name + ".npy"), depth)
        cv2.imwrite(os.path.join(render_normal_path, view.image_name + ".jpg"), normal)

        if use_depth_filter:
            view_dir = torch.nn.functional.normalize(view.get_rays(), p=2, dim=-1)
            depth_normal = out["depth_normal"].permute(1, 2, 0)
            depth_normal = torch.nn.functional.normalize(depth_normal, p=2, dim=-1)
            dot = torch.sum(view_dir * depth_normal, dim=-1).abs()
            angle = torch.acos(dot)
            mask = angle > (80.0 / 180 * 3.14159)
            depth_tsdf[mask] = 0
        depths_tsdf_fusion.append(depth_tsdf.squeeze().cpu())

    if volume is not None:
        depths_tsdf_fusion = torch.stack(depths_tsdf_fusion, dim=0)
        for idx, view in enumerate(tqdm(views, desc="TSDF Fusion progress")):
            ref_depth = depths_tsdf_fusion[idx].cuda()

            if view.mask is not None:
                ref_depth[view.mask.squeeze() < 0.5] = 0
            ref_depth[ref_depth > max_depth] = 0
            ref_depth = ref_depth.detach().cpu().numpy()
            pose = np.identity(4)
            pose[:3, :3] = view.R.transpose(-1, -2)
            pose[:3, 3] = view.T
            color = o3d.io.read_image(os.path.join(render_path, view.image_name + ".jpg"))
            if mask_path is not None:
                mask = np.array(cv2.imread(os.path.join(mask_path, str(idx) + ".png"), cv2.IMREAD_GRAYSCALE)) > 0
                color_np = np.asarray(color)
                ref_depth = np.where(mask, ref_depth, 0)
                color_np = np.where(mask[:, :, None], color_np, 0)
                color = o3d.geometry.Image(color_np)
            depth = o3d.geometry.Image((ref_depth * 1000).astype(np.uint16))
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                color, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
            volume.integrate(
                rgbd,
                o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                pose)
            if args.include_feature:
                color_feature = o3d.io.read_image(os.path.join(render_language_path, view.image_name + ".png"))
                rgbd_feature = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    color_feature, depth, depth_scale=1000.0, depth_trunc=max_depth, convert_rgb_to_intensity=False)
                volume_feature.integrate(
                    rgbd_feature,
                    o3d.camera.PinholeCameraIntrinsic(W, H, view.Fx, view.Fy, view.Cx, view.Cy),
                    pose)


def render_sets(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool,
                max_depth: float, voxel_size: float, num_cluster: int, use_depth_filter: bool, args,
                mask_path=None, ply_path=None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        if ply_path is not None:
            gaussians.load_ply(ply_path)
        scene.loaded_iter = None
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
        volume_feature = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=voxel_size,
            sdf_trunc=4.0 * voxel_size,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        if not skip_train:
            name = "train"
            if ply_path is not None:
                name = "train_" + os.path.splitext(os.path.basename(ply_path))[0]
            render_set(dataset.model_path, name, scene.loaded_iter, scene.getTrainCameras(), scene, gaussians,
                       pipeline, background, args,
                       max_depth=max_depth, volume=volume, volume_feature=volume_feature,
                       use_depth_filter=use_depth_filter, mask_path=mask_path)
            print(f"extract_triangle_mesh")
            mesh = volume.extract_triangle_mesh()

            if mask_path is not None:
                path = os.path.join(dataset.model_path, "mesh_mask")
                os.makedirs(path, exist_ok=True)
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_mask_" + mask_path.split('/')[-2] + "_" +
                                                        mask_path.split('/')[-1] + ".ply"), mesh,
                                           write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                mesh = post_process_mesh(mesh, num_cluster)
                o3d.io.write_triangle_mesh(os.path.join(path,
                                                        "tsdf_fusion_post_mask_" + mask_path.split('/')[-2] + "_" +
                                                        mask_path.split('/')[-1] + ".ply"), mesh,
                                           write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                if args.include_feature:
                    mesh_feature = volume_feature.extract_triangle_mesh()
                    o3d.io.write_triangle_mesh(
                        os.path.join(path, "feature_tsdf_fusion_mask_" + mask_path.split('/')[-2] + "_" +
                                     mask_path.split('/')[-1] + ".ply"), mesh_feature,
                        write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                    mesh_feature = post_process_mesh(mesh_feature, num_cluster)
                    o3d.io.write_triangle_mesh(
                        os.path.join(path, "feature_tsdf_fusion_post_mask_" + mask_path.split('/')[-2] + "_" +
                                     mask_path.split('/')[-1] + ".ply"), mesh_feature,
                        write_triangle_uvs=True, write_vertex_colors=True,
                        write_vertex_normals=True)
            elif ply_path is not None:
                path = os.path.join(dataset.model_path, "mesh_ply")
                os.makedirs(path, exist_ok=True)
                o3d.io.write_triangle_mesh(
                    os.path.join(path, "tsdf_fusion_" + os.path.splitext(os.path.basename(ply_path))[0] + ".ply"), mesh,
                    write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                mesh = post_process_mesh(mesh, num_cluster)
                o3d.io.write_triangle_mesh(os.path.join(path,
                                                        "tsdf_fusion_post_" +
                                                        os.path.splitext(os.path.basename(ply_path))[0] + ".ply"), mesh,
                                           write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                if args.include_feature:
                    mesh_feature = volume_feature.extract_triangle_mesh()
                    o3d.io.write_triangle_mesh(os.path.join(path, "feature_tsdf_fusion_" +
                                                            os.path.splitext(os.path.basename(ply_path))[0] + ".ply"),
                                               mesh_feature,
                                               write_triangle_uvs=True, write_vertex_colors=True,
                                               write_vertex_normals=True)
                    mesh_feature = post_process_mesh(mesh_feature, num_cluster)
                    o3d.io.write_triangle_mesh(os.path.join(path, "feature_tsdf_fusion_post_" +
                                                            os.path.splitext(os.path.basename(ply_path))[0] + ".ply"),
                                               mesh_feature,
                                               write_triangle_uvs=True, write_vertex_colors=True,
                                               write_vertex_normals=True)
            else:
                path = os.path.join(dataset.model_path, "mesh")
                os.makedirs(path, exist_ok=True)
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion.ply"), mesh,
                                           write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)

                mesh = post_process_mesh(mesh, num_cluster)
                o3d.io.write_triangle_mesh(os.path.join(path, "tsdf_fusion_post.ply"), mesh,
                                           write_triangle_uvs=True, write_vertex_colors=True, write_vertex_normals=True)
                if args.include_feature:
                    mesh_feature = volume_feature.extract_triangle_mesh()
                    o3d.io.write_triangle_mesh(os.path.join(path, "feature_tsdf_fusion.ply"), mesh_feature,
                                               write_triangle_uvs=True, write_vertex_colors=True,
                                               write_vertex_normals=True)
                    mesh_feature = post_process_mesh(mesh_feature, num_cluster)
                    o3d.io.write_triangle_mesh(os.path.join(path, "feature_tsdf_fusion_post.ply"), mesh_feature,
                                               write_triangle_uvs=True, write_vertex_colors=True,
                                               write_vertex_normals=True)
        if not skip_test:
            name = "test"
            if ply_path is not None:
                name = "test_" + os.path.splitext(os.path.basename(ply_path))[0]
            render_set(dataset.model_path, name, scene.loaded_iter, scene.getTestCameras(), scene, gaussians,
                       pipeline, background, args)


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=5.0, type=float)
    parser.add_argument("--voxel_size", default=0.01, type=float)
    parser.add_argument("--num_cluster", default=3, type=int)
    parser.add_argument("--use_depth_filter", action="store_true")
    parser.add_argument("--include_feature", action="store_true")
    parser.add_argument("--ply_path", type=str, default=None, help='Render the specified point cloud')
    parser.add_argument("--mask_path", type=str, default=None, help='Generate a mesh of a individual object (2D query)')
    parser.add_argument("--normalized", action="store_true",
                        help='Whether to normalize features')
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    if hasattr(args, 'mask_path'):
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                    args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args,
                    mask_path=args.mask_path)
    elif hasattr(args, 'ply_path'):
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                    args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args,
                    ply_path=args.ply_path)
    else:
        render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test,
                    args.max_depth, args.voxel_size, args.num_cluster, args.use_depth_filter, args)
