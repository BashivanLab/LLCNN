# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import random
import shutil
import time
import warnings
import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter

from utils import get_model, get_data_loaders, get_train_args


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

from pathlib import Path
import json
import signal
import subprocess
import sys

from PIL import Image, ImageOps, ImageFilter
from torch import optim
import torchvision
import torchvision.transforms as transforms
from train_progressive_imagenet import validate

parser = argparse.ArgumentParser(description='Barlow Twins Training')

#parser.add_argument('--workers', default=8, type=int, metavar='N',
 #                   help='number of data loader workers')


parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                    help='weight on off-diagonal terms')
parser.add_argument('--projector', default='8192-8192-8192', type=str,
                    metavar='MLP', help='projector MLP')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', # 100
                    help='print frequency')
parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                    help='base learning rate for weights')
parser.add_argument('--learning-rate-biases', default=0.0048, type=float, metavar='LR',
                    help='base learning rate for biases and batch norm parameters')


parser.add_argument('--save_path', default='./chkpts', type=str, help='path to where to save checkpoints')
parser.add_argument('--data_path', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', default='resnet18', type=str)                    
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--dataset', default='imagenet', type=str, help='imagenet')                    
parser.add_argument('-p', '--log-interval', default=50, type=int,
                    metavar='N', help='logging frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--pool_type', default=None, type=str)
parser.add_argument('--max_num_pools', default=1, type=int, help='# of kernel pools to apply')
parser.add_argument('--noise_std', default=0.2, type=float, help='Noise STD.')

parser.add_argument('--kap_kernelsize', default=3, type=int, help='KAP kernel size')
parser.add_argument('--exp_name', default='', type=str, help='experiment name')

parser.add_argument('--sigma_factor', default=1, type=float, help='sigma multiplier')
parser.add_argument('--decay', default=4, type=float, help='decay multiplier')
#parser.add_argument('--decayvalue', default=0.159, type=float, help='decay to value')
parser.add_argument('--continuous', default=False, type=bool, help='sigma multiplier')
parser.add_argument('--prog', default=False, type=bool, help='progressive training')
parser.add_argument('--training_tune', default=0, type=int, help='tuning training')
parser.add_argument('--local_conv', default=False, type=bool, help='local conn')


def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))\
      
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    
    global best_acc1
    args.gpu = gpu

    ROOT_PATH = args.save_path
    EXP_NAME = f'cln_{args.arch}_{args.pool_type}_{args.max_num_pools}_{args.noise_std}{args.exp_name}'
    TRAINED_MODEL_PATH = os.path.join(ROOT_PATH, f'trained_models/imagenet', EXP_NAME)
    
    if not os.path.exists(TRAINED_MODEL_PATH):
        os.makedirs(TRAINED_MODEL_PATH)
    
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu

        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
        writer = SummaryWriter(TRAINED_MODEL_PATH)
        
        
    

    
    torch.backends.cudnn.benchmark = True
    
    
    # params for model
    args.kap_kernelsize = 1 * args.sigma_factor
    if args.prog == True:
        args.kap_kernelsize = max(math.exp(-args.decay*args.start_epoch/90), args.sigma_factor)
        if args.pool_type=="mexicanhat":
            args.kap_kernelsize = max(math.exp(-0.4*4*args.start_epoch/90)*2*args.sigma_factor, args.sigma_factor)
            
            
    print(args.kap_kernelsize)
    
    model = get_model(args)
    trainargs = get_train_args(args)
    print(args.exp_name)
    print(args.continuous)
            



    #batch_size = int(trainargs['train_batch_size'] / ngpus_per_node)
    args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)#not sure
    
    model = BarlowTwins(args, model).cuda(args.gpu)
    #model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    param_weights = []
    param_biases = []
    for param in model.parameters():
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]
    #model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    # create model
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        batch_size = trainargs['train_batch_size']
        args.batch_size = batch_size
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            batch_size = int(trainargs['train_batch_size'] / ngpus_per_node)
            args.batch_size = batch_size
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            batch_size = trainargs['train_batch_size']
            args.batch_size = batch_size
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        batch_size = trainargs['train_batch_size']
        args.batch_size = batch_size
    else:
        model = torch.nn.DataParallel(model).cuda()
        batch_size = trainargs['train_batch_size']
        args.batch_size = batch_size
    
    
    
    optimizer = LARS(parameters, trainargs['lr'], momentum=trainargs['momentum'],
                     weight_decay=trainargs['weight_decay'],
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            #best_acc1 = checkpoint['best_acc1']
            
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError()





    dataset = torchvision.datasets.ImageFolder(f"{args.data_path}/train", Transform())#/train
    #sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #assert batch_size % args.world_size == 0
    #per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=args.workers,
        pin_memory=True)#, sampler=sampler)
    
    #save first
    epoch = args.start_epoch
    if args.distributed:
        loader.sampler.set_epoch(epoch)
        # train_sampler.set_epoch(epoch)
    if epoch==0:
        # acc1 = validate(val_loader, model, criterion, writer, epoch, args)
        # is_best=False
        chkpt_name = os.path.join(TRAINED_MODEL_PATH, f'{args.arch}_{epoch}.pt')
        torch.save({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': 0.1, #acc1,
            'acc1': 0.1, #acc1,
            'optimizer' : optimizer.state_dict(),
        }, chkpt_name)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    
    _, val_loader = get_data_loaders(args.dataset,
                                    batch_size, batch_size,
                                    args.data_path,
                                    norm=False,
                                    noise_std=args.noise_std,
                                    args=args)
   
    for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)):
        y1 = y1.cuda(gpu, non_blocking=True)
        y2 = y2.cuda(gpu, non_blocking=True)
        adjust_learning_rate(args, optimizer, loader, step)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            loss = model.forward(y1, y2)
            #  not sure if mean works here
        scaler.scale(loss.sum()).backward()
        scaler.step(optimizer)
        scaler.update()
       
        print(epoch, step, loss.sum())
        
                
                
        
      

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        chkpt_name = os.path.join(TRAINED_MODEL_PATH, f'{args.arch}_{epoch+1}.pt')                
        torch.save({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': 0.1, #acc1,
            'acc1': 0.1, #acc1,
            'optimizer' : optimizer.state_dict(),
        }, chkpt_name)


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = 100 * len(loader) # set to 100 tempo
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args, model):
        super().__init__()
        self.args = args
        self.backbone = model #torchvision.models.resnet50(zero_init_residual=True)
        #model = get_model(args)
        self.backbone.fc = nn.Identity()

        # projector
        sizes = [529] + list(map(int, args.projector.split('-')))
        layers = []
        for i in range(len(sizes) - 2):
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))
        self.projector = nn.Sequential(*layers)

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        z1 = self.projector(self.backbone(y1))
        z2 = self.projector(self.backbone(y2))
        #print(z1.shape,z2.shape)
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        #torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag

        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)


    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])



class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


def get_imagenet_val_loader(batch_size, data_path, norm=False, noise_std=0., shuffle=True, args=None):
  valdir = data_path
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
  ts = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
  ]
  if noise_std > 0.: 
    ts.append(AddGaussianNoise(0., noise_std))
  if norm:
    ts.append(normalize)
  val_transforms = transforms.Compose(ts)
  ds = torchvision.datasets.ImageFolder(valdir, val_transforms)
  loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
  shuffle=shuffle, num_workers=1, pin_memory=True)
  loader.name = "imagenet_validation"
  return loader


if __name__ == '__main__':
    main()