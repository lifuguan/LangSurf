from copy import deepcopy
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing_extensions import Literal
import os
from rich.progress import Console

CONSOLE = Console(width=120)

import torch, random
from torchvision.transforms import Resize, InterpolationMode
from torchvision.utils import save_image
from diffusers.models.attention_processor import AttnProcessor
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.schedulers import DDIMScheduler, DDIMInverseScheduler
import utils.edit_utils as utils


def process_mask(image_folder, kernel_size=(7, 7), area_threshold=2000):
    kernel = np.ones(kernel_size, np.uint8)

    for filename in os.listdir(image_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(image_folder, filename)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed_image, connectivity=8)
            filtered_image = np.zeros_like(closed_image)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= area_threshold:
                    filtered_image[labels == i] = 255
            cv2.imwrite(image_path, filtered_image)
    print("Process masks.")


@dataclass
class GaussEditorConfig():
    render_rate: int = 500
    """how many gauss steps for gauss training"""
    edit_prompt: str = ""
    """Positive Prompt"""
    reverse_prompt: str = ""
    """DDIM Inversion Prompt"""
    langsam_obj: str = ""
    """The object to be edited"""
    guidance_scale: float = 5
    """Classifier Free Guidance"""
    num_inference_steps: int = 20
    """Inference steps"""
    chunk_size: int = 4
    """Batch size for image editing, feel free to reduce to fit your GPU"""
    ref_view_num: int = 4
    """Number of reference frames"""
    diffusion_ckpt: str = 'CompVis/stable-diffusion-v1-4'
    """Diffusion checkpoints"""
    rendered_depth_path = ""
    """Path to rendered depth images"""
    rendered_rgb_path: str = ""
    """Path to rendered rgb images"""
    rendered_mask_path: str = ""
    """Path to rendered mask (2D query)"""

    mask_ratio: float = 0.4
    edit_ids: str = ""

    """Path to rendered rgb images"""
    edited_rgb_path: str = ""

    """Path to edited rgb images"""


class GaussEditor():
    config: GaussEditorConfig

    def __init__(self,
                 config: GaussEditorConfig,
                 device: str,
                 test_mode: Literal["test", "val", "inference"] = "val",
                 ) -> None:
        self.config = config
        self.test_mode = test_mode
        self.edit_prompt = self.config.edit_prompt
        self.reverse_prompt = self.config.reverse_prompt
        self.pipe_device = device
        self.ddim_scheduler = DDIMScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")
        self.ddim_inverser = DDIMInverseScheduler.from_pretrained(self.config.diffusion_ckpt, subfolder="scheduler")

        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-depth")
        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(self.config.diffusion_ckpt,
                                                                      controlnet=controlnet).to(self.pipe_device).to(
            torch.float16)
        self.pipe.to(self.pipe_device)

        added_prompt = 'best quality, extremely detailed'
        self.positive_prompt = self.edit_prompt + ', ' + added_prompt
        self.positive_reverse_prompt = self.reverse_prompt + ', ' + added_prompt
        self.negative_prompts = 'longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality'

        self.mask_ratio = self.config.mask_ratio
        self.edit_ids = [int(x) for x in self.config.edit_ids.split(" ")]

        self.data, (self.width, self.height) = self.load_rendered_data(self.config.rendered_depth_path,
                                                                       self.config.rendered_rgb_path,
                                                                       self.config.rendered_mask_path)

        view_num = len(self.data)
        anchors = [(view_num * i) // self.config.ref_view_num for i in range(self.config.ref_view_num)] + [view_num]

        random.seed(13789)
        self.ref_indices = [random.randint(anchor, anchors[idx + 1]) for idx, anchor in enumerate(anchors[:-1])]
        self.num_ref_views = len(self.ref_indices)

        self.num_inference_steps = self.config.num_inference_steps
        self.guidance_scale = self.config.guidance_scale
        self.controlnet_conditioning_scale = 1.0
        self.eta = 0.0
        self.chunk_size = self.config.chunk_size

    def update_latents(self, cam_idx, latent, mask):
        """Save mid results"""
        self.data[cam_idx]["z_0_image"] = latent.cpu().to(torch.float32).numpy()
        if mask is not None:
            self.data[cam_idx]["mask_image"] = mask

    def check_mask_ratio(self, mask):
        mask = mask[0][0]
        image_shape = mask.shape
        total_area = image_shape[0] * image_shape[1]  # area of images
        mask_area = int(mask.sum().numpy())  # area of masks

        thres_masks = self.mask_ratio * total_area  # ratio of the masks
        return mask_area > thres_masks

    def load_rendered_data(self, rendered_depth_path, rendered_rgb_path, rendered_mask_path):
        disparities = utils.read_depth2disparity(rendered_depth_path)
        rgbs, (width, height), frames = utils.read_rgb(rendered_rgb_path)
        if rendered_mask_path == '':
            image_pairs = []
            for i, (disparity, rgb) in enumerate(zip(disparities, rgbs)):
                if frames[i] not in self.edit_ids:
                    continue
                image_pairs.append({
                    'image_idx': frames[i],
                    'disparity_image': disparity,
                    'z_0_image': None,
                    'mask_image': None,
                    'unedited_image': rgb,
                })
            return image_pairs, (width, height)
        else:
            masks = utils.read_mask(rendered_mask_path)
            image_pairs = []
            idx = 0
            # assert len(disparities) == len(rgbs) and len(rgbs) == len(masks)
            for i, (disparity, rgb, mask) in enumerate(zip(disparities, rgbs, masks)):
                if frames[i] not in self.edit_ids:
                    continue
                if self.check_mask_ratio(mask) == True:
                    image_pairs.append({
                        'image_idx': frames[i],
                        'disparity_image': disparity,
                        'z_0_image': None,
                        'mask_image': mask,
                        'unedited_image': rgb,
                    })
                    idx += 1
            return image_pairs, (width, height)

    @torch.no_grad()
    def image2latent(self, image):
        """Encode images to latents"""
        image = image * 2 - 1
        latents = self.pipe.vae.encode(image.to(torch.float16).to(self.pipe_device))['latent_dist'].mean
        latents = latents * 0.18215
        return latents

    def render_reverse(self):
        '''Render rgb, depth and reverse rgb images back to latents'''

        print("#############################")
        CONSOLE.print("Start Reversing: ", style="bold yellow")
        print("#############################")
        for cam_idx in range(len(self.data)):
            CONSOLE.print(f"Rendering view {cam_idx}", style="bold yellow")
            rendered_image = self.data[cam_idx]
            rendered_rgb = rendered_image['unedited_image'].to(torch.float16).to(self.pipe_device)  # [512 512 3] 0-1
            disparity = rendered_image['disparity_image'].to(torch.float16).to(self.pipe_device)  # [512 512 1]

            # reverse the images to noises
            self.pipe.unet.set_attn_processor(processor=AttnProcessor())
            self.pipe.controlnet.set_attn_processor(processor=AttnProcessor())
            init_latent = self.image2latent(rendered_rgb)

            self.pipe.scheduler = self.ddim_inverser
            latent, _ = self.pipe(prompt=self.positive_reverse_prompt,  # placeholder here, since cfg=0
                                  num_inference_steps=self.num_inference_steps,
                                  latents=init_latent,
                                  image=disparity, return_dict=False, guidance_scale=0, output_type='latent')

            # LangSAM is optional
            if self.config.langsam_obj != "":
                langsam_obj = self.config.langsam_obj
                langsam_rgb_pil = Image.fromarray((rendered_rgb.cpu().numpy() * 255).astype(np.uint8))
                masks, _, _, _ = self.langsam.predict(langsam_rgb_pil, langsam_obj)
                mask_npy = masks.clone().cpu().numpy()[0] * 1

            if self.config.langsam_obj != "":
                self.update_latents(cam_idx, latent, mask_npy)
            else:
                self.update_latents(cam_idx, latent, None)
        print("#############################")
        CONSOLE.print("Done Reversing", style="bold yellow")
        print("#############################")

    def edit_images(self):
        '''Edit images with ControlNet and AttnAlign'''
        # Set up ControlNet and AttnAlign
        self.pipe.scheduler = self.ddim_scheduler
        self.pipe.unet.set_attn_processor(
            processor=utils.CrossViewAttnProcessor(self_attn_coeff=0.6,
                                                   unet_chunk_size=2))
        self.pipe.controlnet.set_attn_processor(
            processor=utils.CrossViewAttnProcessor(self_attn_coeff=0,
                                                   unet_chunk_size=2))
        CONSOLE.print("Done Resetting Attention Processor", style="bold blue")

        print("#############################")
        CONSOLE.print("Start Editing: ", style="bold yellow")
        CONSOLE.print(f"Reference views are {[j + 1 for j in self.ref_indices]}", style="bold yellow")
        print("#############################")
        ref_disparity_list = []
        ref_z0_list = []
        for ref_idx in self.ref_indices:
            ref_data = deepcopy(self.data[ref_idx])
            ref_disparity = ref_data['disparity_image']
            ref_z0 = ref_data['z_0_image']
            ref_disparity_list.append(ref_disparity)
            ref_z0_list.append(ref_z0)

        ref_disparities = np.concatenate(ref_disparity_list, axis=0)
        ref_disparity_torch = torch.from_numpy(ref_disparities.copy()).to(torch.float16).to(self.pipe_device)
        ref_z0s = np.concatenate(ref_z0_list, axis=0)
        ref_z0_torch = torch.from_numpy(ref_z0s.copy()).to(torch.float16).to(self.pipe_device)

        # Edit images in chunk
        for idx in range(0, len(self.data), self.config.chunk_size):
            chunked_data = self.data[idx: idx + self.config.chunk_size]

            indices = [current_data['image_idx'] for current_data in chunked_data]
            mask_images = [current_data['mask_image'] for current_data in chunked_data if
                           'mask_image' in current_data.keys()]
            unedited_images = [current_data['unedited_image'] for current_data in chunked_data]
            CONSOLE.print(f"Generating view: {indices}", style="bold yellow")

            depth_images = [current_data['disparity_image'] for current_data in chunked_data]
            disparities = np.concatenate(depth_images, axis=0)
            disparities_torch = torch.from_numpy(disparities.copy()).to(torch.float16).to(self.pipe_device)

            z_0_images = [current_data['z_0_image'] for current_data in chunked_data]  # list of np array
            z0s = np.concatenate(z_0_images, axis=0)
            latents_torch = torch.from_numpy(z0s.copy()).to(torch.float16).to(self.pipe_device)

            disp_ctrl_chunk = torch.concatenate((ref_disparity_torch, disparities_torch), dim=0)
            latents_chunk = torch.concatenate((ref_z0_torch, latents_torch), dim=0)

            chunk_edited = self.pipe(
                prompt=[self.positive_prompt] * (self.num_ref_views + len(chunked_data)),
                negative_prompt=[self.negative_prompts] * (self.num_ref_views + len(chunked_data)),
                latents=latents_chunk,
                image=disp_ctrl_chunk,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                eta=self.eta,
                output_type='pt',
            ).images[self.num_ref_views:]
            chunk_edited = chunk_edited.cpu()

            # Insert edited images back to train data for training
            for local_idx, edited_image in enumerate(chunk_edited):
                global_idx = indices[local_idx]

                bg_cntrl_edited_image = edited_image

                if not any(x is None for x in mask_images):
                    mask = mask_images[local_idx].to(torch.uint8)
                    bg_mask = 1 - mask

                    unedited_image = unedited_images[local_idx]
                    bg_cntrl_edited_image = (edited_image * mask + unedited_image * bg_mask)[0]
                data_idx = -1
                for index, d in enumerate(self.data):
                    if d.get('image_idx') == global_idx:
                        data_idx = index
                        break
                self.data[data_idx]["edited_image"] = bg_cntrl_edited_image.permute(1, 2, 0).to(
                    torch.float32)  # [512 512 3]

        print("#############################")
        CONSOLE.print("Done Editing", style="bold yellow")
        print("#############################")
        os.makedirs(self.config.edited_rgb_path, exist_ok=True)
        for idx in range(len(self.data)):
            resize_edited_image = Resize((self.height, self.width), interpolation=InterpolationMode.BILINEAR,
                                         antialias=True)(self.data[idx]["edited_image"].cpu().permute(2, 0, 1))
            save_image(resize_edited_image,
                       os.path.join(self.config.edited_rgb_path, f"frame_{self.data[idx]['image_idx']:05d}.jpg"))


from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--edit_prompt", default="a photo of a red floor with gray carpets ", type=str)
    parser.add_argument("--reverse_prompt", default="a photo of a brown floor with gray carpets", type=str)
    parser.add_argument("--edit_object",
                        default='floor', type=str)
    parser.add_argument("--guidance_scale", default=5.0, type=float)
    parser.add_argument("--mask_ratio", default=0.0, type=float)
    parser.add_argument("--edit_ids",
                        default='6 12 18 24 30 36 42 48 54 66 72 78 84 90 96 102 108 114',
                        help="Image ids, such as frame_00001.jpg,frame_00002.jpg is 1 2",
                        type=str)
    parser.add_argument("--num_inference_steps", default=20, type=int)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--feature_level", type=int, default=3)
    args = parser.parse_args()
    if "scene" in args.dataset_name:
        base_dir = f"data/scannet/{args.dataset_name}/output/{args.dataset_name}_{args.feature_level}/train/ours_None"
    else:
        base_dir = f"data/lerf_ovs/{args.dataset_name}/output/{args.dataset_name}_{args.feature_level}/train/ours_None"
    config = GaussEditorConfig()
    config.edit_prompt = args.edit_prompt
    config.reverse_prompt = args.reverse_prompt
    config.guidance_scale = args.guidance_scale
    config.rendered_mask_path=f"eval_result/{args.dataset_name}/masks_0.6/{args.edit_object}/{args.feature_level-1}"
    config.rendered_rgb_path = os.path.join(base_dir, 'renders_rgb')
    config.rendered_depth_path = os.path.join(base_dir, 'renders_depth_npy')
    config.edited_rgb_path = os.path.join(base_dir, 'edited_rgb')
    config.mask_ratio = args.mask_ratio
    config.num_inference_steps = args.num_inference_steps
    config.edit_ids = args.edit_ids
    process_mask(config.rendered_mask_path)
    gseditor = GaussEditor(config, device='cuda:0')
    gseditor.render_reverse()
    gseditor.edit_images()
