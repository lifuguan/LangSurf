import os
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
from scene.app_model import AppModel
from scene.cameras import Camera


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(22)


def training(dataset: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations, saving_iterations,
             checkpoint_iterations, checkpoint, debug_from, edit_images_path):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()

    (model_params, first_iter) = torch.load(checkpoint)
    gaussians.restore(model_params, opt)
    app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    include_feature = True
    progress_bar = tqdm(range(first_iter, first_iter + 3000), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    gaussians.change_reqiures_grad("finetune", iteration=first_iter, quiet=False)
    for key, value in scene.train_cameras.items():
        filtered_cameras = []
        for camrea in value:
            frame_id = int(camrea.image_name.split("_")[-1])
            if frame_id in args.edit_ids:
                camrea.image_path = os.path.join(edit_images_path, camrea.image_name + '.jpg')
                camrea.preload_img = False
                filtered_cameras.append(camrea)
            scene.train_cameras[key] = filtered_cameras
    scene.model_path = os.path.join(scene.model_path, "finetune")
    os.makedirs(scene.model_path, exist_ok=True)
    for iteration in range(first_iter, first_iter + 3000):

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam: Camera = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration > opt.single_view_weight_from_iter,
                            return_depth_normal=iteration > opt.single_view_weight_from_iter,
                            include_feature=include_feature)
        image, viewspace_point_tensor, visibility_filter, radii, language_feature, instance_feature = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], \
                render_pkg["language_feature"], render_pkg["instance_feature"]

        # Loss
        overall_loss = 0;
        image_loss = None
        if iteration < first_iter + 3000:
            # image loss
            ssim_loss = (1.0 - ssim(image, gt_image))
            if 'app_image' in render_pkg and ssim_loss < 0.5:
                app_image = render_pkg['app_image']
                Ll1 = l1_loss(app_image, gt_image)
            else:
                Ll1 = l1_loss(image, gt_image)
            image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
            overall_loss += image_loss.clone()

        overall_loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log if image_loss is not None else 0.0 + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, include_feature=include_feature, finetune=True)

            # Optimizer step
            if iteration < first_iter + 3000:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                app_model.optimizer.zero_grad(set_to_none=True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(include_feature=include_feature), iteration),
                           scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                app_model.save_weights(scene.model_path, iteration)

    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000, 43_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000, 43_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000, 40_000, 43_000])
    parser.add_argument("--start_checkpoint", type=str,
                        default=None)
    parser.add_argument("--edit_ids",
                        default='6 12 18 24 30 36 42 48 54 66 72 78 84 90 96 102 108 114',
                        type=str)
    args = get_combined_args(parser)
    args.save_iterations.append(args.iterations)
    args.start_checkpoint = os.path.join(args.model_path, 'chkpnt40000.pth')
    edit_images_path = os.path.join(args.model_path, 'train/ours_None/edited_rgb')
    args.edit_ids = [int(x) for x in args.edit_ids.split(" ")]
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations,
             args.checkpoint_iterations, args.start_checkpoint, args.debug_from, edit_images_path)

    # All done
    print("\nTraining complete.")
