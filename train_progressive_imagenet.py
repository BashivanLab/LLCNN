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

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
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



best_acc1 = 0


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
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
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
    
    
    
    
    # set the kap_kernelsize according to the decay factor
    #decay_factor = args.decay #3
    #init_kap_kernelsize = 1.0
    #last_decay_epoch = 90
    #decay_factor = max(math.exp(-decay_factor*args.start_epoch/last_decay_epoch), args.decayvalue)#0.159)
    
    

    args.kap_kernelsize = int(1 * args.sigma_factor)
    if args.prog == True:
        args.kap_kernelsize = max(math.exp(-args.decay*args.start_epoch/90), args.sigma_factor)
        if args.pool_type=="mexicanhat":
            args.kap_kernelsize = max(math.exp(-0.4*4*args.start_epoch/90)*10, args.sigma_factor)
            
    
    print(args.kap_kernelsize)
    
    model = get_model(args)
    trainargs = get_train_args(args)
    print(args.exp_name)
    print(args.continuous)


    # create model
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
        batch_size = trainargs['train_batch_size']
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
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            batch_size = trainargs['train_batch_size']
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        batch_size = trainargs['train_batch_size']
    else:
        model = torch.nn.DataParallel(model).cuda()
        batch_size = trainargs['train_batch_size']

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), trainargs['lr'],
                                momentum=trainargs['momentum'],
                                weight_decay=trainargs['weight_decay'])

    # optimizer = torch.optim.Adam(model.parameters(), trainargs['lr'],
    #                             betas=(0.9, 0.95),
    #                             weight_decay=trainargs['weight_decay'])
                                

    # optionally resume from a checkpoint
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
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            raise ValueError()

    cudnn.benchmark = True

    # Data loading code
    train_loader, val_loader = get_data_loaders(args.dataset,
                                               batch_size, batch_size,
                                               args.data_path,
                                               norm=False,
                                               noise_std=args.noise_std,
                                               args=args)
    

    if args.evaluate:
        validate(val_loader, model, criterion, writer, 0, args)
        return

    # Training         
    epoch = args.start_epoch
    if args.distributed:
        train_loader.sampler.set_epoch(epoch)
        # train_sampler.set_epoch(epoch)
    if epoch==0:
        # acc1 = validate(val_loader, model, criterion, writer, epoch, args)
        # is_best=False
        chkpt_name = os.path.join(TRAINED_MODEL_PATH, f'{args.arch}_{epoch}.pt')
        save_checkpoint({
            'epoch': epoch,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': 0.1, #acc1,
            'acc1': 0.1, #acc1,
            'optimizer' : optimizer.state_dict(),
        }, filename=chkpt_name)
        
        
    adjust_learning_rate(optimizer, epoch, trainargs)

    # train for one epoch
    train(train_loader, model, criterion, optimizer, epoch, writer, args)

    # evaluate on validation set
    acc1 = validate(val_loader, model, criterion, writer, epoch, args)
    
    # print the kernel


    # remember best acc@1 and save checkpoint
    is_best = acc1 > best_acc1
    best_acc1 = max(acc1, best_acc1)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed
            and args.rank % ngpus_per_node == 0):
        chkpt_name = os.path.join(TRAINED_MODEL_PATH, f'{args.arch}_{epoch+1}.pt')                
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'acc1': acc1,
            'optimizer' : optimizer.state_dict(),
        }, filename=chkpt_name)


def train(train_loader, model, criterion, optimizer, epoch, writer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        current_iteration = epoch*len(train_loader)+i
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        if torch.cuda.is_available():
            target = target.cuda(args.gpu, non_blocking=True)
            
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.log_interval == 0:
            progress.display(i)
            writer.add_scalar('Train Loss', loss.item(), current_iteration)
            writer.add_scalar('Train Top1', top1.avg.item(), current_iteration)


def validate(val_loader, model, criterion, writer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.log_interval == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        writer.add_scalar('Val Top1', top1.avg.item(), epoch+1)              
        writer.add_scalar('Val Top5', top5.avg.item(), epoch+1)              

    return top1.avg


def save_checkpoint(state, filename):
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
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
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, trainargs):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = trainargs['lr'] * (trainargs['scheduler_gamma'] ** (epoch // trainargs['schedule_rate']))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res





if __name__ == '__main__':
    main()