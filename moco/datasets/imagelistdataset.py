from ctypes import resize
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.utils import make_grid, save_image
from PIL import ImageFilter
import torch.utils.data as data
import numpy as np
import torch
import os
import random
from copy import deepcopy
from scipy.stats import gamma
from collections import defaultdict
import sys
sys.path.append('../')
import moco.loader
import moco.builder


def longtail_class_distrib(list_fname=None, seed=0, num_classes=-1):
    """TODO: Docstring for longtail_class_distrib.

    :list_fname: TODO
    :seed: TODO
    :num_classes: TODO
    :returns: TODO

    """
    # Generate Gamma distribution
    rv = gamma(3, loc=-4, scale=2.0)
    if list_fname is not None:
        lab_to_fnames = defaultdict(list)
        with open(list_fname, 'r') as f:
            filedata = f.read().splitlines()
            for line in filedata:
                lab_to_fnames[int(line.split(' ')[1])].append(
                    line.split(' ')[0])
        labels = list(lab_to_fnames.keys())
    else:
        labels = list(range(num_classes))
    # Seed controls which classes are in tail
    np.random.seed(seed)
    np.random.shuffle(labels)
    distrib = np.array(
        [rv.pdf(li * 18 / 1000.0) / rv.pdf(0) for li in range(len(labels))])
    class_distrib = distrib[np.argsort(labels)]
    return class_distrib


class ImageListDataset(data.Dataset):
    """Dataset that reads videos"""

    def __init__(self, list_fname, transforms=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert (
            os.path.exists(list_fname)), '{} does not exist'.format(list_fname)
        with open(list_fname, 'r') as f:
            filedata = f.read().splitlines()
            self.pair_filelist = [(d.split(' ')[0], d.split(' ')[0])
                                  for d in filedata]
            print(self.pair_filelist[:10])

        if not isinstance(transforms, list):
            transforms = [transforms]
        self.transforms = transforms
        self.std = torch.Tensor(self.transforms[0].transforms[-1].std).view(
            3, 1, 1)
        self.mean = torch.Tensor(self.transforms[0].transforms[-1].mean).view(
            3, 1, 1)

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        fname1, fname2 = self.pair_filelist[index]
        im1 = datasets.folder.pil_loader(fname1)
        im2 = datasets.folder.pil_loader(fname2)
        meta = {}
        i = 0
        transform = None
        if self.transforms is not None:
            i = np.random.randint(len(self.transforms))
            transform = self.transforms[i]
        if transform is not None:
            im1 = transform(im1)
            im2 = transform(im2)
        meta['transind1'] = i
        meta['transind2'] = i

        out = {
            'input1': im1,
            'input2': im2,
            'meta': meta,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.pair_filelist)


class DMControlDataset(data.Dataset):
    """Dataset that reads DMControl videos"""

    # transforms:

    #  1. random resized crop
    def resize_crop_all(self, ims):
        params = transforms.RandomResizedCrop.get_params(ims[0], scale=(0.3, 1.2), ratio=(0.75, 1.3333333333333333))
        return [TF.resized_crop(im, *params, size=(84, 84)) for im in ims]

    #  2. random apply (color jitter)
    def color_jitter_all(self, ims):
        p = 0.8
        if np.random.rand() < p:
            brightness = 0.4
            contrast = 0.4
            saturation = 0.4
            hue = 0.1
            params = transforms.ColorJitter.get_params(
                [max(0, 1 - brightness), 1 + brightness],
                [max(0, 1 - contrast), 1 + contrast],
                [max(0, 1 - saturation), 1 + saturation],
                [-hue, hue])
            for fn_id in params[0]:
                if fn_id == 0:
                    ims = [TF.adjust_brightness(im, params[1]) for im in ims]
                elif fn_id == 1:
                    ims = [TF.adjust_contrast(im, params[2]) for im in ims]
                elif fn_id == 2:
                    ims = [TF.adjust_saturation(im, params[3]) for im in ims]
                elif fn_id == 3:
                    ims = [TF.adjust_hue(im, params[4]) for im in ims]
        return ims

    #  3. random grayscale
    def grayscale_all(self, ims):
        p = 0.2
        if np.random.rand() < p:
            ims = [TF.to_grayscale(im, num_output_channels=3) for im in ims]
        return ims

    #  4. random Gaussian blur
    def gaussian_blur_all(self, ims):
        p = 0.5
        if np.random.rand() < p:
            sigma = random.uniform(.1, 2.)
            ims = [im.filter(ImageFilter.GaussianBlur(radius=sigma)) for im in ims]
        return ims

    # 5. to tensor
    def to_tensor_all(self, ims):
        return [TF.to_tensor(im) for im in ims]

    # 6. normalize
    def normalize_all(self, ims):
        return [TF.normalize(im, self.mean, self.std) for im in ims]

    def __init__(self, list_fname, input_sec=3):
        """TODO: to be defined.

        :filelist: TODO

        """
        data.Dataset.__init__(self)
        assert (
            os.path.exists(list_fname)), '{} does not exist'.format(list_fname)
        with open(list_fname, 'r') as f:
            filedata = f.read().splitlines()
            self.filelist = [d.split(' ')[0] for d in filedata]
        print('dmcontrol dataset with {} input_sec'.format(input_sec))

        for i,f in enumerate(self.filelist):
            # extract frame id from filename
            base = os.path.basename(f)
            frame_id = base.split('.')[0].split('_')[-1]
            frame_id = int(frame_id)

            # adjust frame id if at the start/end of video
            padding = input_sec // 2
            if frame_id < padding:
                frame_id = padding
            elif frame_id > 500 - padding:
                frame_id = 500 - padding
            self.filelist[i] = [
                os.path.join(os.path.dirname(f), \
                    '_'.join(base.split('.')[0].split('_')[:-1]) + f'_{idx:03d}.png') \
                        for idx in range(frame_id - padding, frame_id + padding + 1)
            ]
        print(self.filelist[:4])
        
        self.transforms = [
            self.resize_crop_all,
            self.color_jitter_all,
            self.grayscale_all,
            self.gaussian_blur_all,
            self.to_tensor_all,
            self.normalize_all,
        ]
        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        self.input_sec = input_sec

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        frames = self.filelist[index]
        assert len(frames) == self.input_sec, '{} frames'.format(len(frames))

        ims = [datasets.folder.pil_loader(f) for f in frames]
        ims1, ims2 = ims, deepcopy(ims)

        meta = {}
        i = 0

        for transform in self.transforms:
            ims1 = transform(ims1)
            ims2 = transform(ims2)

        meta['transind1'] = i
        meta['transind2'] = i

        out = {
            'input1': torch.cat(ims1, dim=0),
            'input2': torch.cat(ims2, dim=0),
            'meta': meta,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.filelist)


class LongTailImageListDataset(ImageListDataset):
    """Docstring for LongTailImageListDataset. """

    def __init__(self, list_fname, transforms=None, seed=1992):
        """TODO: to be defined. """
        ImageListDataset.__init__(self, list_fname, transforms=transforms)

        # Generate Gamma distribution
        rv = gamma(3, loc=-4, scale=2.0)

        lab_to_fnames = defaultdict(list)
        with open(list_fname, 'r') as f:
            filedata = f.read().splitlines()
            for line in filedata:
                lab_to_fnames[int(line.split(' ')[1])].append(
                    line.split(' ')[0])

        # Seed controls which classes are in tail
        np.random.seed(seed)
        labels = list(lab_to_fnames.keys())
        np.random.shuffle(labels)

        # Sample images for each class
        max_im_per_lab = max([len(v) for v in lab_to_fnames.values()])
        for li, lab in enumerate(labels):
            # Magic numbers
            num = int(rv.pdf(li * 18 / 1000.0) * max_im_per_lab / rv.pdf(0))
            replace = (len(lab_to_fnames[lab]) < num)
            lab_to_fnames[lab] = np.random.choice(lab_to_fnames[lab],
                                                  size=num,
                                                  replace=replace)

        self.pair_filelist = [(v, v) for lab_fnames in lab_to_fnames.values()
                              for v in lab_fnames]
        print(self.pair_filelist[:10])


class UniformImageListDataset(ImageListDataset):
    """Docstring for UniformImageListDataset. """

    def __init__(self, list_fname, transforms=None, seed=1992,
                 num_images=1000):
        """TODO: to be defined. """
        ImageListDataset.__init__(self, list_fname, transforms=transforms)

        lab_to_fnames = defaultdict(list)
        with open(list_fname, 'r') as f:
            filedata = f.read().splitlines()
            for line in filedata:
                lab_to_fnames[int(line.split(' ')[1])].append(
                    line.split(' ')[0])

        # Seed controls which images are sampled
        np.random.seed(seed)
        labels = list(lab_to_fnames.keys())
        num_per_lab = num_images // len(labels)

        # Sample images for each class
        for li, lab in enumerate(labels):
            replace = (len(lab_to_fnames[lab]) < num_per_lab)
            lab_to_fnames[lab] = np.random.choice(lab_to_fnames[lab],
                                                  size=num_per_lab,
                                                  replace=replace)

        self.pair_filelist = [(v, v) for lab_fnames in lab_to_fnames.values()
                              for v in lab_fnames]
        print(self.pair_filelist[:10])


class ImageListStandardDataset(data.Dataset):
    """Dataset that reads videos"""

    def __init__(self, list_fname, transform=None):
        """TODO: to be defined.

        :pair_filelist: TODO

        """
        data.Dataset.__init__(self)
        assert (
            os.path.exists(list_fname)), '{} does not exist'.format(list_fname)
        with open(list_fname, 'r') as f:
            filedata = f.read().splitlines()
            self.pair_filelist = [(d.split(' ')[0], int(d.split(' ')[1]))
                                  for d in filedata]

        self.transform = transform

    def __getitem__(self, index):
        """TODO: Docstring for __getitem__.

        :index: TODO
        :returns: TODO

        """
        fname1, target = self.pair_filelist[index]
        im1 = datasets.folder.pil_loader(fname1)
        if self.transform is not None:
            im1 = self.transform(im1)

        out = {
            'input': im1,
            'target': torch.tensor(target),
            'fname': fname1,
        }
        return out

    def __len__(self):
        """TODO: Docstring for __len__.

        :f: TODO
        :returns: TODO

        """
        return len(self.pair_filelist)



if __name__ == '__main__':
    list_fname = '/checkpoint/nihansen/data/tdmpc2/dmcontrol.txt'

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(84, scale=(0.3, 1.2)),
        transforms.RandomApply(
            [
                transforms.ColorJitter(0.4, 0.4, 0.4,
                                        0.1)  # not strengthened
            ],
            p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([moco.loader.GaussianBlur([.1, 2.])],
                                p=0.5),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    dataset = DMControlDataset(list_fname, input_sec=3)
    
    # rows = []
    # for i in range(16):
    #     _data = dataset[i]
    #     im0, im1 = [_data['input1'], _data['input2']]
    #     row = torch.stack([im0, im1], dim=0)
    #     rows.append(row)
    
    # # save as grid
    # grid = make_grid(torch.cat(rows, dim=0), nrow=2)
    # save_image(grid, 'grid_ours.png')

    # dataset_prev = ImageListDataset(list_fname, transforms=augmentation)
    # rows = []

    # for i in range(16):
    #     _data = dataset_prev[i]
    #     im0, im1 = [_data['input1'], _data['input2']]
    #     row = torch.stack([im0, im1], dim=0)
    #     rows.append(row)

    # # save as grid
    # grid = make_grid(torch.cat(rows, dim=0), nrow=2)
    # save_image(grid, 'grid_prev.png')

    # build model
    import torchvision.models as models
    model = moco.builder.MoCo(models.__dict__['resnet50'],
                              128, 65536,
                              0.999, 0.07,
                              True, 3).cuda()

    _data = dataset[0]
    im0, im1 = _data['input1'].unsqueeze(0).cuda(), _data['input2'].unsqueeze(0).cuda()
    print(im0.shape, im1.shape)
