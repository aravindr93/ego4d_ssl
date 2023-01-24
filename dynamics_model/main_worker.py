#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import builtins
import os
import time
import sys
import shutil

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

from mjrl.utils.logger import DataLog
from utils.fusion import fuse_preprocess, fuse_base
from utils.model_loading import load_pvr_model
from utils.models import InverseDynamicsModel, ForwardDynamicsModel
from utils.optimizer import LARS

from omegaconf import DictConfig, OmegaConf

def main_worker(gpu, ngpus_per_node: int, args: DictConfig, train_dataset: torch.utils.data.Dataset):
    # args = OmegaConf.to_container(args, resolve=True, throw_on_missing=True)
    cudnn.benchmark = True
    args.environment.gpu = gpu

    # suppress printing if not master
    if args.environment.multiprocessing_distributed and args.environment.gpu != 0:

        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.environment.gpu is not None:
        print("Use GPU: {} for training".format(args.environment.gpu))

    # load the pvr backbone with default weights
    print("=> creating {} model with default pre-trained weights".format(args.model.embedding))
    pvr_model, embedding_dim, transforms = load_pvr_model(args.model.embedding, None)

    # load separate weights for the pvr backbone if needed
    if os.path.exists(args.environment.load_path):
        print("=> loading checkpoint '{}'".format(args.environment.load_path))
        if args.environment.gpu == "":
            checkpoint = torch.load(args.environment.load_path)
        else:
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(args.environment.gpu)
            checkpoint = torch.load(args.environment.load_path, map_location=loc)
        pvr_model.load_state_dict(checkpoint["state_dict"])
        print("=> loaded {} model from '{}'".format(args.model.embedding, args.environment.load_path))

    # prepare fusion functions for combining embeddings across multiple views and timesteps (for partial observability)
    fusion_preprocess = fuse_preprocess[args.data.fuse_embeddings]
    fusion_base = fuse_base[args.data.fuse_embeddings]
    fusion_base = fusion_base(embedding_dim, (args.data.history_window + 1), num_views=len(args.data.suite.img_keys))
    fused_embedding_dim = fusion_base.latent_dim

    # summarize all dimensions
    action_dim = train_dataset.action_dim
    proprioception_dim = train_dataset.proprioception_dim
    latent_state_dim = args.data.latent_state_dim
    print(f"Representation Embedding dim: {embedding_dim}")
    print(f"Fused Embedding dim: {fused_embedding_dim}")
    print(f"Proprioception dim: {proprioception_dim}")
    print(f"Latent State dim: {latent_state_dim}")
    print(f"Action dim: {action_dim}")

    if args.environment.distributed:
        if args.environment.dist_url == "env://" and args.environment.rank == -1:
            args.environment.rank = int(os.environ["RANK"])
        if args.environment.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.environment.rank = args.environment.rank * ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.environment.dist_backend,
            init_method=args.environment.dist_url,
            world_size=args.environment.world_size,
            rank=args.environment.rank,
        )
        torch.distributed.barrier()

    # create dynamics model
    if args.dynamics == 'inverse':
        model = InverseDynamicsModel(fused_embedding_dim, proprioception_dim, latent_state_dim, action_dim, pvr_model, args.model, fusion_preprocess, fusion_base)
    elif args.dynamics == 'forward':
        model = ForwardDynamicsModel(fused_embedding_dim, proprioception_dim, latent_state_dim, action_dim, pvr_model, args.model, fusion_preprocess, fusion_base)
    else:
        raise NotImplementedError("Dynamics must either be forward or inverse")

    # infer learning rate before changing batch size
    args.optim.lr = args.optim.lr * args.optim.batch_size / 256

    if args.environment.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.model.sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.environment.gpu is not None:
            torch.cuda.set_device(args.environment.gpu)
            model.cuda(args.environment.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.optim.batch_size = int(
                args.optim.batch_size / args.environment.world_size
            )
            args.environment.workers = int(
                (args.environment.workers + ngpus_per_node - 1) / ngpus_per_node
            )
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[args.environment.gpu],
            )
        else:
            model.cuda()
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.environment.gpu is not None:
        torch.cuda.set_device(args.environment.gpu)
        model = model.cuda(args.environment.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")

    # optimizer setup
    assert args.optim.optimizer in ["lars", "adamw"]
    if args.optim.optimizer == "lars":
        optimizer = LARS(
            model.module.parameters(),
            args.optim.lr,
            weight_decay=args.optim.weight_decay,
            momentum=args.optim.momentum,
        )
    elif args.optim.optimizer == "adamw":
        optimizer = torch.optim.AdamW(
            model.module.parameters(), 
            args.optim.lr, 
            weight_decay=args.optim.weight_decay
        )

    scaler = torch.cuda.amp.GradScaler()

    # optionally resume from a checkpoint
    os.makedirs(os.path.join(args.logging.ckpt_dir, args.logging.name), exist_ok=True)
    ckpt_fname = os.path.join(
        args.logging.ckpt_dir, args.logging.name, "checkpoint_{:04d}.pth"
    )
    if args.environment.resume:
        for i in range(args.optim.epochs, -1, -1):
            if os.path.exists(ckpt_fname.format(i)):
                print("=> loading checkpoint '{}'".format(ckpt_fname.format(i)))
                if args.environment.gpu == "":
                    checkpoint = torch.load(ckpt_fname.format(i))
                else:
                    # Map model to be loaded to specified single gpu.
                    loc = "cuda:{}".format(args.environment.gpu)
                    checkpoint = torch.load(ckpt_fname.format(i), map_location=loc)
                args.optim.start_epoch = checkpoint["epoch"]
                model.load_state_dict(checkpoint["state_dict"])
                pvr_model.load_state_dict(checkpoint["pvr_state_dict"])
                optimizer.load_state_dict(checkpoint["optimizer"])
                print(
                    "=> loaded checkpoint '{}' (epoch {})".format(
                        ckpt_fname.format(i), checkpoint["epoch"]
                    )
                )
                break

    # Create logger
    logger = None
    if args.logging.log_tb and args.environment.rank == 0:
        logger = DataLog(
            wandb_user=args.logging.wandb_user,
            wandb_project=args.logging.wandb_project,
            wandb_config=OmegaConf.to_container(
                args, resolve=True, throw_on_missing=True
            ),
        )

    cudnn.benchmark = True

    if args.environment.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.optim.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.environment.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )

    for epoch in range(args.optim.start_epoch, args.optim.epochs):

        if args.environment.distributed:
            train_sampler.set_epoch(epoch)

        sys.stdout.flush()
        # adjust_learning_rate(optimizer, epoch, args)
        print("Train Epoch {}".format(epoch))

        train(train_loader, model, optimizer, scaler, logger, epoch, args)

        if not args.environment.multiprocessing_distributed or (
            args.environment.multiprocessing_distributed and args.environment.rank == 0
        ):
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.model.embedding,
                    "state_dict": model.state_dict(),
                    "pvr_state_dict": model.module.pvr_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                is_best=False,
                filename=ckpt_fname.format(epoch),
            )

            # save the log
            if logger is not None:
                logger.save_log(save_path=args.logging.ckpt_dir)

            # remove previous checkpoint if necessary to save space
            # if os.path.exists(ckpt_fname.format(epoch - 1)):
            # os.remove(ckpt_fname.format(epoch - 1))

    if logger is not None:
        logger.run.finish()


def train(
    train_loader: torch.utils.data.DataLoader, 
    model: nn.parallel.DistributedDataParallel, 
    optimizer: torch.optim.Optimizer, 
    scaler: torch.cuda.amp.GradScaler, 
    logger: DataLog, 
    epoch: int, 
    args: DictConfig
) -> None:
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    dynamics_losses = AverageMeter("Model Loss", ":.4e")
    embedding_losses = AverageMeter("Embedding Loss", ":.4e")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, dynamics_losses, embedding_losses],
        prefix="Epoch: [{}]".format(epoch),
        logger=logger,
    )

    # switch to train mode
    model.train()

    end = time.time()

    for batch_i, data in enumerate(train_loader):
        # measure data loading time
        *obs_window, embeddings, curr_prop, next_prop, action = data
        data_time.update(time.time() - end)

        if args.environment.gpu is not None:
            obs_window = list(obs_window)
            for j in range(len(obs_window)):
                obs_window[j] = obs_window[j].cuda(args.environment.gpu, non_blocking=True)
            embeddings = embeddings.detach().cuda(args.environment.gpu, non_blocking=True)
            curr_prop = curr_prop.cuda(args.environment.gpu, non_blocking=True)
            next_prop = next_prop.cuda(args.environment.gpu, non_blocking=True)
            action = action.cuda(args.environment.gpu, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast(True):
            dynamics_loss, embedding_loss = model(obs_window, embeddings, curr_prop, next_prop, action)
        dynamics_losses.update(dynamics_loss.item(), action.size(0))
        embedding_losses.update(embedding_loss.item(), action.size(0))

        # compute gradient and take step
        optimizer.zero_grad()
        scaler.scale(dynamics_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if batch_i % args.logging.print_freq == 0:
            log_step = int(
                epoch * len(train_loader.dataset) // args.optim.batch_size
                + batch_i * torch.distributed.get_world_size()
            )
            progress.display(batch_i)
            progress.log(log_step)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", tbname=""):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", logger=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))
        sys.stdout.flush()

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"

    def update_global_step(self, global_step):
        if self.logger is not None:
            self.logger.global_step = global_step
        else:
            pass

    def log(self, batch=None):
        self.update_global_step(batch)
        if self.logger is not None:
            scalar_dict = self.get_scalar_dict()
            for k, v in scalar_dict.items():
                self.logger.log_kv(k, v)

    def get_scalar_dict(self):
        out = {}
        for meter in self.meters:
            val = meter.avg
            tag = meter.name
            out[tag] = val
        return out

    def save_log(self, fname):
        if self.logger is not None:
            self.logger.save_log(save_path=fname)
