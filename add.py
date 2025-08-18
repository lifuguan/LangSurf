import os
import torch
import argparse
from gaussian_renderer import GaussianModel

def process_and_save_gaussians(input_ply, input_object_ply, translation_vector):

    gaussians = GaussianModel(3)
    gaussians.load_ply(input_ply)
    gaussians_add = GaussianModel(3)
    gaussians_add.load_ply(input_object_ply)
    gaussians_add._xyz = gaussians_add._xyz + torch.tensor(translation_vector).to('cuda')
    gaussians._features_dc = torch.cat((gaussians._features_dc, gaussians_add._features_dc), dim=0)
    gaussians._features_rest = torch.cat((gaussians._features_rest, gaussians_add._features_rest), dim=0)
    gaussians._language_feature = torch.cat((gaussians._language_feature, gaussians_add._language_feature), dim=0)
    gaussians._opacity = torch.cat((gaussians._opacity, gaussians_add._opacity), dim=0)
    gaussians._rotation = torch.cat((gaussians._rotation, gaussians_add._rotation), dim=0)
    gaussians._scaling = torch.cat((gaussians._scaling, gaussians_add._scaling), dim=0)
    gaussians._xyz = torch.cat((gaussians._xyz, gaussians_add._xyz), dim=0)
    gaussians.max_radii2D = torch.cat((gaussians.max_radii2D, gaussians_add.max_radii2D), dim=0)
    gaussians._instance_feature = torch.cat((gaussians._instance_feature, gaussians_add._instance_feature), dim=0)

    base_name = os.path.basename(input_object_ply)
    output_plyname = f"add_{base_name}"
    output_path = os.path.join(os.path.dirname(input_ply), output_plyname)
    gaussians.save_ply(output_path, include_feature=True)
    print(f"Saved merged point cloud to {output_path}")

def main():

    parser = argparse.ArgumentParser(description="Process and merge 3D point clouds")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--feature_level", type=int, default=3)
    parser.add_argument("--input_object_ply", type=str, required=True, help="Path to the object ply")
    parser.add_argument("--translation", type=float, nargs=3, default=[1.8, 0.0, 1.9],
                        help="Translation vector to apply to the object ply")
    args = parser.parse_args()
    if "scene" in args.dataset_name:
        input_ply = f"data/scannet/{args.dataset_name}/output/{args.dataset_name}_{args.feature_level}/point_cloud/iteration_40000/point_cloud.ply"
    else:
        input_ply = f"data/lerf_ovs/{args.dataset_name}/output/{args.dataset_name}_{args.feature_level}/point_cloud/iteration_40000/point_cloud.ply"
    process_and_save_gaussians(input_ply, args.input_object_ply, args.translation)

if __name__ == "__main__":
    main()
