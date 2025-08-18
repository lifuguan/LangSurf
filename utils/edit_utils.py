import numpy as np
import torch
import cv2
from torchvision.transforms import Resize, InterpolationMode, ToTensor, Compose, CenterCrop
from einops import rearrange
import glob
from diffusers.utils import USE_PEFT_BACKEND
from diffusers.utils import load_image
from natsort import natsorted


def read_mask(mask_dir):
    transform = Compose([
        Resize((512, 512), interpolation=InterpolationMode.BILINEAR, antialias=True),
        # CenterCrop((512, 512)),
        ToTensor()])
    mask_paths = glob.glob(mask_dir + '/*.png')
    mask_paths = natsorted(mask_paths)
    mask_list = []
    for mask_path in mask_paths:
        mask = load_image(mask_path)
        mask_torch = transform(mask).bool().unsqueeze(0)  # torch.Size([1, 3, 512, 512]) -1~1
        mask_list.append(mask_torch)
    return mask_list


def read_rgb(rgb_dir):
    transform = Compose([
        Resize((512, 512), interpolation=InterpolationMode.BILINEAR, antialias=True),
        # CenterCrop((512, 512)),
        ToTensor()])
    rgb_paths = sorted(glob.glob(rgb_dir + '/*.jpg'))
    rgb_list = []
    rgb_frame = []
    for rgb_path in rgb_paths:
        rgb = load_image(rgb_path);
        width, height = rgb.size
        file_name = rgb_path.split('/')[-1]
        frame_number = int(file_name.split('_')[1].split('.')[0].lstrip('0') or '0')
        rgb_frame.append(frame_number)
        rgb_torch = transform(rgb).unsqueeze(0)  # torch.Size([1, 3, 512, 512])
        rgb_list.append(rgb_torch)
    return rgb_list, (width, height), rgb_frame


def read_depth2disparity(depth_dir):
    depth_paths = sorted(glob.glob(depth_dir + '/*.npy'))
    disparity_list = []
    for depth_path in depth_paths:
        depth = np.load(depth_path)
        depth = cv2.resize(depth, (512, 512)).reshape((512, 512, 1))  # [512,512,1]
        # depth = CenterCrop((512, 512))(torch.from_numpy(depth))[..., None].numpy() # [512,512,1]

        disparity = 1 / (depth + 1e-5)
        disparity_map = disparity / np.max(disparity)  # 0.00233~1
        # disparity_map = disparity_map.astype(np.uint8)[:,:,0]
        disparity_map = np.concatenate([disparity_map, disparity_map, disparity_map], axis=2)
        disparity_list.append(torch.from_numpy(disparity_map[None]).permute(0, 3, 1, 2).float())  # [1,512,512,3]
    return disparity_list


def compute_attn(attn, query, key, value, video_length, ref_frame_index, attention_mask):
    key_ref_cross = rearrange(key, "(b f) d c -> b f d c", f=video_length)
    key_ref_cross = key_ref_cross[:, ref_frame_index]
    key_ref_cross = rearrange(key_ref_cross, "b f d c -> (b f) d c")
    value_ref_cross = rearrange(value, "(b f) d c -> b f d c", f=video_length)
    value_ref_cross = value_ref_cross[:, ref_frame_index]
    value_ref_cross = rearrange(value_ref_cross, "b f d c -> (b f) d c")

    key_ref_cross = attn.head_to_batch_dim(key_ref_cross)
    value_ref_cross = attn.head_to_batch_dim(value_ref_cross)
    attention_probs = attn.get_attention_scores(query, key_ref_cross, attention_mask)
    hidden_states_ref_cross = torch.bmm(attention_probs, value_ref_cross)
    return hidden_states_ref_cross


class CrossViewAttnProcessor:
    def __init__(self, self_attn_coeff, unet_chunk_size=2):
        self.unet_chunk_size = unet_chunk_size
        self.self_attn_coeff = self_attn_coeff

    def __call__(
            self,
            attn,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
            scale=1.0, ):

        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states, *args)

        is_cross_attention = encoder_hidden_states is not None
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)
        query = attn.head_to_batch_dim(query)
        # Sparse Attention
        if not is_cross_attention:
            ################## Perform self attention
            key_self = attn.head_to_batch_dim(key)
            value_self = attn.head_to_batch_dim(value)
            attention_probs = attn.get_attention_scores(query, key_self, attention_mask)
            hidden_states_self = torch.bmm(attention_probs, value_self)
            #######################################

            video_length = key.size()[0] // self.unet_chunk_size
            ref0_frame_index = [0] * video_length
            ref1_frame_index = [1] * video_length
            ref2_frame_index = [2] * video_length
            ref3_frame_index = [3] * video_length

            hidden_states_ref0 = compute_attn(attn, query, key, value, video_length, ref0_frame_index, attention_mask)
            hidden_states_ref1 = compute_attn(attn, query, key, value, video_length, ref1_frame_index, attention_mask)
            hidden_states_ref2 = compute_attn(attn, query, key, value, video_length, ref2_frame_index, attention_mask)

            key = rearrange(key, "(b f) d c -> b f d c", f=video_length)
            key = key[:, ref3_frame_index]
            key = rearrange(key, "b f d c -> (b f) d c")
            value = rearrange(value, "(b f) d c -> b f d c", f=video_length)
            value = value[:, ref3_frame_index]
            value = rearrange(value, "b f d c -> (b f) d c")

        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)
        hidden_states_ref3 = torch.bmm(attention_probs, value)

        hidden_states = self.self_attn_coeff * hidden_states_self + (1 - self.self_attn_coeff) * torch.mean(
            torch.stack([hidden_states_ref0, hidden_states_ref1, hidden_states_ref2, hidden_states_ref3]),
            dim=0) if not is_cross_attention else hidden_states_ref3
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
