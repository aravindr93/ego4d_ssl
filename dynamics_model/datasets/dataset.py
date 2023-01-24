from omegaconf import DictConfig
from PIL import Image
from typing import Any, Union, Callable
from tqdm import tqdm
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import random
import numpy as np
import os
import gc
import re
import bisect
import pickle

def compute_embeddings(
    processed_images: torch.Tensor, 
    model: nn.Module, 
    embedding_dim: int, 
    device: str = 'cpu', 
    batch_size: int = 32, 
) -> torch.Tensor:
    model.to(device=device)
    model.eval()
    input_len = len(processed_images)
    latent_states = torch.zeros(input_len, embedding_dim)
    with torch.no_grad():
        for idx in range(input_len // batch_size + 1):
            start = idx * batch_size
            end = min((idx + 1) * batch_size, input_len)
            
            batch = processed_images[start:end]
            embs = model(batch.to(device))
            latent_states[start:end] = embs
    return latent_states.detach()


def compute_embeddings_from_paths(
    paths: list[dict[str, Any]], 
    img_keys: list[str], 
    model: nn.Module, 
    embedding_dim: int, 
    transforms: Callable[[Union[np.ndarray, torch.Tensor]], torch.Tensor],
    device: str = 'cpu', 
    batch_size: int = 32, 
    keep_images: int = 25, 
    stack_or_concat: str = 'stack'
) -> list[dict[str, Any]]:
    for idx, path in enumerate(tqdm(paths)):
        latent_states = []
        for img_key in img_keys:
            processed_images = torch.stack([transforms(frame) for frame in path[img_key]])
            if idx < len(paths) - keep_images:
                del(path[img_key])   # no longer need the images, free up RAM
                gc.collect()
            latent_states.append(compute_embeddings(processed_images, model, embedding_dim, device, batch_size))
        if stack_or_concat == 'stack':
            path['latent_states'] = torch.stack(latent_states, dim=1)
        else:
            path['latent_states'] = torch.cat(latent_states, dim=1)
        del latent_states
        gc.collect()
    return paths


def retrieve_proprioception(
    paths: list[dict[str, Any]],
    prop_key: str
) -> torch.Tensor:
    # assume at most 1 prop key for now
    prop = []
    for path in paths:
        prop.append(torch.from_numpy(path['env_infos'][prop_key]))
    return torch.cat(prop)


class RandomRotateFrames:
    def __init__(self) -> None:
        self.angles = [0, 90, 180, 270]

    def __call__(self, frames: list) -> list:
        angle = random.choice(self.angles)
        frames = [TF.rotate(frame, angle) for frame in frames]
        return frames
    
class RandomShiftFrames:
    def __init__(self, pad_pixels=12) -> None:
        self.pad_pixels = pad_pixels

    def __call__(self, frames: list) -> list:
        _, h, w = TF.get_dimensions(frames[0])
        h_new = np.random.randint(0, 2 * self.pad_pixels + 1) 
        w_new = np.random.randint(0, 2 * self.pad_pixels + 1) 
        for i in range(len(frames)):
            frame = F.pad(frames[i], (self.pad_pixels, self.pad_pixels, self.pad_pixels, self.pad_pixels), mode='constant', value=0)
            frames[i] = frame[:, h_new:h_new + h, w_new:w_new + w]
        return frames

class ColorJitterFrames:
    def __init__(self, brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5) -> None:
        self.transform = T.ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, frames: list) -> list:
        fn_idx, brightness_factor, contrast_factor, saturation_factor, hue_factor = self.transform.get_params(
            self.transform.brightness, self.transform.contrast, self.transform.saturation, self.transform.hue
        )

        for i in range(len(frames)):
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    frames[i] = TF.adjust_brightness(frames[i], brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    frames[i] = TF.adjust_contrast(frames[i], contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    frames[i] = TF.adjust_saturation(frames[i], saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    frames[i] = TF.adjust_hue(frames[i], hue_factor)
        return frames

class RandomCropFrames:
    def __init__(self, size=224) -> None:
        self.transform = T.RandomCrop(size)

    def __call__(self, frames: list) -> list:
        i, j, h, w = self.transform.get_params(frames[0], self.transform.size)
        frames = [TF.crop(frame, i, j, h, w) for frame in frames]
        return frames

class GrayscaleFrames:
    def __call__(self, frames: list) -> list:
        num_output_channels, _, _ = TF.get_dimensions(frames[0])
        frames =  [TF.rgb_to_grayscale(frame, num_output_channels=num_output_channels) for frame in frames]
        return frames
    
class NoAugmentationFrames:
    def __call__(self, frames: list) -> list:
            return frames

AUGMENTATIONS_LIST = [
    ColorJitterFrames(),
    RandomCropFrames(),
    RandomRotateFrames(),
    RandomShiftFrames(),
    GrayscaleFrames(),
    NoAugmentationFrames()
]


# new function to sort list of trajectory subdirectories or frame files
def natural_keys(text):
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', text)]


class FrameDataset(Dataset):
    def __init__(
        self, 
        model: nn.Module, 
        embedding_dim: int, 
        transforms: Callable[[Union[np.ndarray, torch.Tensor]], torch.Tensor],
        args: DictConfig,
        device: str
    ) -> None:
        self.paths = []
        self.frame_buffer = []
        self.action_dim = 0
        pickle_root = os.path.join(args.pickle_dir, args.suite.name)
        frames_root = os.path.join(args.frames_dir, args.suite.name)
        for env in args.envs:
            env_id = args.suite.prefix + env + args.suite.suffix
            pickle_loc = os.path.join(pickle_root, env_id + ".pickle")
            env_subdir = os.path.join(frames_root, env_id)
            try:
                paths = pickle.load(open(pickle_loc, 'rb'))
            except:
                print("Unable to load the data. Check the data path.")
                print(pickle_loc)
                quit()
            paths = compute_embeddings_from_paths(paths, args.suite.img_keys, model, embedding_dim, transforms, device, keep_images=0)
            self.paths.extend(paths)
            self.action_dim = max(self.action_dim, paths[0]['actions'].shape[-1])
            for traj in sorted(os.listdir(env_subdir), key=natural_keys):
                traj_subdir = os.path.join(env_subdir, traj)
                self.frame_buffer.append([])
                for frame in sorted(os.listdir(traj_subdir), key=natural_keys):
                    self.frame_buffer[-1].append(os.path.join(traj_subdir, frame))
        for path in self.paths:
            path['actions'] = np.pad(path['actions'], ((0, 0), (0, self.action_dim - path['actions'].shape[-1])), constant_values=0)
        
        self.total_timesteps = sum([len(path['actions']) - 1 for path in self.paths])
        self.timestep_cumsum = np.cumsum([len(path['actions']) - 1 for path in self.paths])
        self.history_window = args.history_window
        
        self.img_keys = args.suite.img_keys
        self.transforms = transforms
        self.to_tensor = T.ToTensor()
        self.augmentations = args.augmentations
        
        self.actions = np.concatenate([path['actions'][:-1] for path in self.paths])
        self.proprioception = retrieve_proprioception(paths, args.suite.prop_key) if args.suite.prop_key else torch.Tensor([])
        self.ret_prop = len(self.proprioception > 0)
        self.proprioception_dim = self.proprioception.shape[-1] if self.ret_prop else 0

    def __len__(self) -> int:
        return self.total_timesteps

    def __getitem__(self, index: int) -> Any:
        path_index = bisect.bisect(self.timestep_cumsum, index)
        timestep = index - self.timestep_cumsum[path_index - 1] if path_index else index
        frames, embeddings = self.retrieve_frames(path_index, timestep)
        curr_prop = self.proprioception[index] if self.ret_prop else torch.Tensor([])
        next_prop = self.proprioception[index + 1] if self.ret_prop else torch.Tensor([])
        return *frames, embeddings, curr_prop, next_prop, self.actions[index]

    def retrieve_frames(self, path_index: int, timestep: int) -> list[tuple[torch.Tensor]]:
        frames, embeddings = [], []
        for k in range(self.history_window - 1, -2, -1):
            frames.append(self.to_tensor(Image.open(self.frame_buffer[path_index][max(timestep - k, 0)])))
            embeddings.append(self.paths[path_index]['latent_states'][max(timestep - k, 0)])
        if self.augmentations:
            aug = random.choice(AUGMENTATIONS_LIST)
            frames = aug(frames)
        return tuple([self.transforms(frame) for frame in frames]), torch.cat(embeddings, dim=-2)
