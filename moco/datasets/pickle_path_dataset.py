from __future__ import annotations
import os, pickle, gc
import torchvision.datasets as datasets
import torch.utils.data as data
import torch, torchvision
import numpy as np
from typing import Union, Tuple, List, Dict
from PIL import Image


class PicklePathsDataset(data.Dataset):
    """Dataset that reads frames generated from trajectories"""

    def __init__(
        self,
        root_dir: str,
        frameskip: int = 1,
        transforms: Union[torchvision.transforms.Compose, None] = None,
    ) -> None:
        """
        Creates a dataset using frames from sub-directories in root directory.
        Frames are expected to be organized as:
        root
        | -- folder_1 (name of task)
            | -- subfolder_1 (traj_i)
                | -- image_1 (frame_j.jpg)
                | -- image_2
        """
        data.Dataset.__init__(self)
        assert os.path.isdir(root_dir)
        self.root_dir = root_dir
        # get a list of all pickle files in root directory
        tasks = next(os.walk(root_dir))[1]

        # store the transforms
        self.transforms = transforms

        # maintain a frame buffer in memory and populate from dataset
        self.frame_buffer = []
        for task in tasks:
            print("Currently loading task: %s" % task)
            task_root = os.path.join(root_dir, task)
            trajectories = next(os.walk(task_root))[1]
            for traj in trajectories:
                traj_root = os.path.join(task_root, traj)
                frames = os.listdir(traj_root)
                for timestep, frame in enumerate(frames):
                    if timestep % frameskip == 0:
                        frame_meta_data = {
                            "path": os.path.join(traj_root, frame),
                            "task": task,
                            "traj": int(traj.split("_")[-1]),
                            "time": timestep,
                        }
                        self.frame_buffer.append(frame_meta_data)

        # print messages
        print("\n Successfully loaded dataset from root_dir: %s" % root_dir)
        print("\n Dataset size is: %i" % len(self.frame_buffer))

    def __getitem__(self, index: int) -> Dict:
        frame = self.frame_buffer[index]
        frame = Image.open(frame["path"])
        # compute two different views/augmentations of the same image
        if self.transforms is not None:
            im1 = self.transforms(frame)
            im2 = self.transforms(frame)
        out = {
            "input1": im1,
            "input2": im2,
            "meta": dict(),
        }
        return out

    def __len__(self) -> int:
        return len(self.frame_buffer)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.transforms import RandomResizedCrop, ToTensor
    from tqdm import tqdm
    import time

    batch_size, num_workers = 16, 16

    dataset = PicklePathsDataset(
        root_dir="/checkpoint/yixinlin/eaif/datasets/dmc-expert-v0.1/",
        # image_key="images",
        frameskip=5,
        transforms=torchvision.transforms.Compose(
            [
                RandomResizedCrop(size=(224, 224), scale=(0.2, 1.0)),
                ToTensor(),
            ]
        ),
    )
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    t0 = time.time()
    for idx, batch in tqdm(enumerate(data_loader)):
        q, k = batch["input1"], batch["input2"]
    print(
        "\n PicklePathsDataset \n"
        "Generated %i mini-batches of size %i | took time = %2.3f seconds \n"
        % (idx + 1, batch_size, time.time() - t0)
    )
