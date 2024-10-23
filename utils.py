import os
# from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torchvision.datasets as datasets
import torchvision.transforms as transforms
#from advertorch.attacks import LinfPGDAttack, GradientSignAttack, L2PGDAttack
import models.preactresnet_cifar as preact_resnet_cifar
import models.mnist_models as mnist_models
"""
import models.cifar_models as cifar_models
from   models.cifar_contopo_models import SmoothConv
import models.resnet_cifar as resnet_cifar
from models.cifar_vit import VitTiny
"""
import models.resnet_imagenet as resnet_imagenet
import models.resnet_imagenet_continuoustopo as resnet_imagenet_contopo
import models.resnet_imagenet_continuoustopo_ as resnet_imagenet_contopo_
import models.resnet_imagenet_continuoustopo_LLC as resnet_imagenet_contopo_LLC
import models.resnet_imagenet_continuoustopo_LLC_car as resnet_imagenet_contopo_LLC_car
import models.resnet_imagenet_continuoustopo_test as resnet_imagenet_contopo_test
#import models.resnet_imagenet_postactkap as resnet_imagenet_postactkap
import models.wideresnet_cifar as wideresnet_cifar
#from autoattack.autopgd_base import APGDAttack



class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def get_data_loaders(dataset, train_batch_size, test_batch_size, data_path, norm=False, noise_std=0., args=None):
  if dataset == 'mnist':
    train_loader = get_mnist_train_loader(
      batch_size=train_batch_size, data_path=data_path, shuffle=True, noise_std=noise_std)
    test_loader = get_mnist_test_loader(
      batch_size=test_batch_size, data_path=data_path, shuffle=False)
  elif dataset == 'cifar10':
    train_loader = get_cifar10_train_loader(
      batch_size=train_batch_size, data_path=data_path, shuffle=True, norm=norm, noise_std=noise_std)
    test_loader = get_cifar10_test_loader(
      batch_size=test_batch_size, data_path=data_path, shuffle=False, norm=norm)
  elif dataset == 'cifar100':
    train_loader = get_cifar100_train_loader(
      batch_size=train_batch_size, data_path=data_path, shuffle=True, norm=norm, noise_std=noise_std)
    test_loader = get_cifar100_test_loader(
      batch_size=test_batch_size, data_path=data_path, shuffle=False, norm=norm)
  elif dataset == 'tiny-imagenet-200':
    train_loader, test_loader = get_tiny_imagenet_dataset(data_path=data_path,
                                                          img_size=128,
                                                          train_batch_size=train_batch_size,
                                                          test_batch_size=test_batch_size
                                                          )
  elif dataset == 'imagenet':
    train_loader = get_imagenet_train_loader(
      batch_size=train_batch_size, data_path=data_path, norm=norm, noise_std=noise_std, args=args)
    test_loader = get_imagenet_val_loader(
      batch_size=test_batch_size, data_path=data_path, norm=norm, args=args)
  else:
    raise ValueError(f'Dataset not recognized ({dataset})')
  return train_loader, test_loader


def get_model(args):
  if args.dataset == 'mnist':
    num_classes = 10
    if args.arch == 'linear': 
      return mnist_models.LinearModel(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride)
    elif args.arch == 'singleconv': 
      return mnist_models.SingleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, activation=F.relu)  
    elif args.arch == 'doubleconv': 
      return mnist_models.DoubleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, activation=F.relu)  
    elif args.arch == 'tripleconv': 
      return mnist_models.TripleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, activation=F.relu)    
    elif args.arch == 'singleconv_linear': 
      return mnist_models.SingleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride)  
    elif args.arch == 'doubleconv_linear': 
      return mnist_models.DoubleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride)  
    elif args.arch == 'tripleconv_linear': 
      return mnist_models.TripleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride)    
    elif args.arch == 'cnn': 
      return mnist_models.ConvEncoder(num_classes, args.pool_type, 
      args.noise_std, args.kap_kernelsize, args.kap_stride, args.expansion, args.do_prob)  

  elif 'cifar' in args.dataset:
    if args.dataset == 'cifar10':
      num_classes = 10
    else:
      num_classes = 100
    if args.arch == 'resnet18':
      return resnet_cifar.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.kap_stride)
      # return resnet_cifar.ResNet18(num_classes, args.pool_type, 
      # args.noise_std, args.kap_kernelsize, args.kap_stride)
    elif args.arch == 'resnet50':
      return resnet_cifar.ResNet50(num_classes, args.pool_type, 
                                   args.max_num_pools, args.noise_std, args.kap_kernelsize, args.kap_stride)
    elif args.arch == 'vit':
      return VitTiny(num_classes, args.pool_type, 
                     args.noise_std, args.kap_kernelsize, args.kap_stride)
    elif args.arch == 'cnn':
      return cifar_models.ConvEncoder(num_classes, args.pool_type, 
      args.noise_std, args.kap_kernelsize, args.kap_stride, args.expansion, args.do_prob)
    elif args.arch == 'smoothcnn':
      return SmoothConv(num_classes, args.pool_type, 
      args.noise_std, args.kap_kernelsize, args.kap_stride, args.expansion, args.do_prob)
    elif args.arch == 'singleconv': 
      return cifar_models.SingleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion, activation=F.relu)  
    elif args.arch == 'doubleconv': 
      return cifar_models.DoubleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion, activation=F.relu)  
    elif args.arch == 'tripleconv': 
      return cifar_models.TripleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion, activation=F.relu)    
    elif args.arch == 'tripleconv1kap': 
      return cifar_models.TripleConv1KAP(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion, activation=F.relu)    
    elif args.arch == 'tripleconv2kap': 
      return cifar_models.TripleConv2KAP(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion, activation=F.relu)    
    elif args.arch == 'singleconv_linear': 
      return cifar_models.SingleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion)  
    elif args.arch == 'doubleconv_linear': 
      return cifar_models.DoubleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion)  
    elif args.arch == 'tripleconv_linear': 
      return cifar_models.TripleConv(pool_type=args.pool_type, noise_std=args.noise_std, kap_kernelsize=args.kap_kernelsize, kap_stride=args.kap_stride, expansion=args.expansion)    
    elif args.arch == 'wrn34':
      return wideresnet_cifar.WideResnet34(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std)
    elif args.arch == 'preact_resnet18': 
      return preact_resnet_cifar.PreActResNet18(num_classes)
    elif args.arch == 'ensemblecnn': 
      return cifar_models.EnsembleCNN(num_classes=num_classes, ensemble_size=args.expansion**2)
    else:
      raise ValueError(f'Model name not recognized ({args.arch})')

  elif args.dataset == 'tiny-imagenet-200':
    num_classes = 200
    if args.arch == 'resnet18':
      return resnet_imagenet.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std)
    elif args.arch == 'resnet18widex4':
      return resnet_imagenet.ResNet18WideX4(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std)
    elif args.arch == 'resnet18widex9':
      return resnet_imagenet.ResNet18WideX9(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std)      
    else:
      raise ValueError(f'Model name not recognized ({args.arch})')

  elif args.dataset == 'imagenet':
    num_classes = 1000
    if args.arch == 'resnet18':
      return resnet_imagenet.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize)
    elif args.arch == 'resnet18contopo':
      return resnet_imagenet_contopo.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)  
    elif args.arch == 'resnet18contopo_':
      return resnet_imagenet_contopo_.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)
      
        
    elif args.arch == 'resnet18contopo_LLC':
      return resnet_imagenet_contopo_LLC.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)
    
    elif args.arch == 'resnet18contopo_LLC_car':
      return resnet_imagenet_contopo_LLC_car.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)
    
    elif args.arch == 'resnet18contopo_LLC2':
      return resnet_imagenet_contopo_LLC.ResNet18_2(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)
    elif args.arch == 'resnet18contopo_LLC3':
      return resnet_imagenet_contopo_LLC.ResNet18_3(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)
    elif args.arch == 'resnet18contopo_LLC4':
      return resnet_imagenet_contopo_LLC.ResNet18_4(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)
      
          
      
    elif args.arch == 'resnet18contopo_test':
      return resnet_imagenet_contopo_test.ResNet18(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize, args.continuous, args.local_conv)    
        
    elif args.arch == 'resnet18widex4':
      return resnet_imagenet.ResNet18WideX4(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize)
    elif args.arch == 'resnet18widex4contopo':
      return resnet_imagenet_contopo.ResNet18WideX4(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize)    
    elif args.arch == 'resnet18widex4postactkap':
      return resnet_imagenet_postactkap.ResNet18WideX4(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize)   
    elif args.arch == 'resnet18widex9':
      return resnet_imagenet.ResNet18WideX9(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize)      
    elif args.arch == 'resnet50widex4':
      return resnet_imagenet.ResNet50WideX4(num_classes, args.pool_type, 
      args.max_num_pools, args.noise_std, args.kap_kernelsize)      
    else:
      raise ValueError(f'Model name not recognized ({args.arch})')
  else:
    raise ValueError(f'Dataset not recognized ({args.dataset})')


def get_train_args(args):
  dataset = args.dataset
  trainargs = {}
  if dataset == 'mnist': 
    trainargs['num_classes'] = 10
    trainargs['train_batch_size'] = 128
    trainargs['test_batch_size'] = 128
    trainargs['log_interval'] = 200
    trainargs['nb_epoch'] = 11
    trainargs['lr'] = 0.1
    trainargs['weight_decay'] = 1e-4
    trainargs['schedule_milestones'] = [25]
    trainargs['scheduler_gamma'] = 0.1
    trainargs['save_interval'] = 1
  elif 'cifar' in dataset:
    if dataset == 'cifar10':
      trainargs['num_classes'] = 10
    else:
      trainargs['num_classes'] = 100
    trainargs['train_batch_size'] = 128
    trainargs['test_batch_size'] = 128
    trainargs['log_interval'] = 200
    trainargs['nb_epoch'] = 200
    trainargs['lr'] = 0.1
    trainargs['weight_decay'] = 1e-4
    trainargs['schedule_milestones'] = [150, 250]
    trainargs['scheduler_gamma'] = 0.1
    trainargs['save_interval'] = 10
  elif 'tiny-imagenet-200' in dataset:
    trainargs['num_classes'] = 200
    trainargs['train_batch_size'] = 128
    trainargs['test_batch_size'] = 128
    trainargs['nb_epoch'] = 90
    trainargs['lr'] = 0.1
    trainargs['momentum'] = 0.9
    trainargs['weight_decay'] = 1e-4
    trainargs['schedule_milestones'] = [30, 60]
    trainargs['scheduler_gamma'] = 0.1
    trainargs['save_interval'] = 5
  elif 'imagenet' in dataset:
    trainargs['num_classes'] = 1000
    trainargs['train_batch_size'] = 100 # RN18 256
    trainargs['test_batch_size'] = 100 # RN18 256
    trainargs['nb_epoch'] = 100
    trainargs['lr'] = 0.1 
    trainargs['momentum'] = 0.9   # 0.9
    trainargs['weight_decay'] =  1e-5 # 1e-4: NT, best: 1e-5
    trainargs['schedule_rate'] = 40
    trainargs['scheduler_gamma'] = 0.1
    
    if args.training_tune==1:
      
      trainargs['schedule_rate'] = 30
      
    if args.training_tune==2:
      
      trainargs['schedule_rate'] = 50
      
    if args.local_conv:
      trainargs['train_batch_size'] = 50 # RN18 256
      trainargs['test_batch_size'] = 50
      

      
      
      
    
      
  else:
    raise ValueError(f'Dataset not recognized ({dataset})')
  if args.arch == 'wrn34': 
    trainargs['nb_epoch'] = 200
    trainargs['schedule_milestones'] = [60, 120, 160]
    # trainargs['weight_decay'] = 5e-4
  return trainargs


def get_mnist_train_loader(batch_size, data_path, shuffle=True, noise_std=0.):
  ts = [
    transforms.ToTensor()
  ]
  if noise_std > 0.: 
    ts.append(AddGaussianNoise(0., noise_std))
  train_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train=True, download=True,
                   transform=train_transforms),
    batch_size=batch_size, shuffle=shuffle)
  loader.name = "mnist_train"
  return loader


def get_mnist_test_loader(batch_size, data_path, shuffle=False):
  loader = torch.utils.data.DataLoader(
    datasets.MNIST(data_path, train=False, download=True,
                   transform=transforms.ToTensor()),
    batch_size=batch_size, shuffle=shuffle)
  loader.name = "mnist_test"
  return loader


def get_cifar10_train_loader(batch_size, data_path, shuffle=True, norm=False, noise_std=0.):
  ts = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
  ]
  if noise_std > 0.: 
    ts.append(AddGaussianNoise(0., noise_std))
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  train_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(data_path, train=True, download=True,
                     transform=train_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=8)
  loader.name = "cifar10_train"
  return loader


def get_cifar10_test_loader(batch_size, data_path, shuffle=False, norm=False):
  ts = [transforms.ToTensor()]
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  test_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(data_path, train=False, download=True,
                     transform=test_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=2)
  loader.name = "cifar10_test"
  return loader


def get_cifar100_train_loader(batch_size, data_path, shuffle=True, norm=False, noise_std=0.):
  ts = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
  ]
  if noise_std > 0.: 
    ts.append(AddGaussianNoise(0., noise_std))
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  train_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_path, train=True, download=True,
                     transform=train_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=8)
  loader.name = "cifar10_train"
  return loader


def get_cifar100_test_loader(batch_size, data_path, shuffle=False, norm=False):
  ts = [transforms.ToTensor()]
  if norm:
    ts.append(transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)))
  test_transforms = transforms.Compose(ts)
  loader = torch.utils.data.DataLoader(
    datasets.CIFAR100(data_path, train=False, download=True,
                     transform=test_transforms),
    batch_size=batch_size, shuffle=shuffle, num_workers=2)
  loader.name = "cifar10_test"
  return loader


def get_tiny_imagenet_dataset(data_path, img_size=32, train_batch_size=128, test_batch_size=128, shuffle_test=True):
    # preprocess data with https://gist.github.com/moskomule/2e6a9a463f50447beca4e64ab4699ac4
    print(data_path)
    train_root = os.path.join(data_path, 'train')
    test_root = os.path.join(data_path, 'val')
    # mean = [x / 255 for x in [127.5, 127.5, 127.5]]
    # std = [x / 255 for x in [127.5, 127.5, 127.5]]
    train_transform = transforms.Compose(
      [
      #  transforms.Resize((img_size, img_size)),
       transforms.RandomResizedCrop(img_size),
       transforms.RandomHorizontalFlip(),
      #  transforms.RandomCrop(img_size, padding=4),
       transforms.ToTensor(),
      #  transforms.Normalize(mean, std)
      ])
    test_transform = transforms.Compose(
      [transforms.Resize((img_size, img_size)), transforms.ToTensor(), 
      # transforms.Normalize(mean, std)
      ])
    train_data = datasets.ImageFolder(train_root, transform=train_transform)
    test_data = datasets.ImageFolder(test_root, transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, shuffle=shuffle_test)

    return train_loader, test_loader


def get_imagenet_train_loader(batch_size, data_path, norm=False, noise_std=0., args=None):
  traindir = os.path.join(data_path, 'train')
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  ts = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
  ]
  if noise_std > 0.: 
    ts.append(AddGaussianNoise(0., noise_std))
  if norm:
    ts.append(normalize)
  train_transforms = transforms.Compose(ts)

  ds = datasets.ImageFolder(traindir, train_transforms)
  if args.distributed:
    train_sampler = torch.utils.data.distributed.DistributedSampler(ds)
  else:
    train_sampler = None
  loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
  shuffle=(train_sampler is None), num_workers=args.workers, 
  pin_memory=True, sampler=train_sampler)
  loader.name = "imagenet_train"
  return loader


def get_imagenet_val_loader(batch_size, data_path, norm=False, noise_std=0., shuffle=False, args=None):
  valdir = os.path.join(data_path, 'val')
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
  ds = datasets.ImageFolder(valdir, val_transforms)
  loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
  shuffle=shuffle, num_workers=args.workers, pin_memory=True)
  loader.name = "imagenet_validation"
  return loader


def get_attack(model, attack_name, dataset):
  if dataset == 'mnist':
    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=0.3,
        nb_iter=40, eps_iter=0.01, rand_init=True, clip_min=0.0,
        clip_max=1.0, targeted=False)
  elif 'cifar' in dataset:
    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
        nb_iter=20, eps_iter=2. / 255, rand_init=True, clip_min=0., clip_max=1.0,
        targeted=False)
    elif attack_name == 'l2_pgd':
      return L2PGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=1.,
        nb_iter=20, eps_iter=0.01, rand_init=True, clip_min=0., clip_max=1.0,
        targeted=False)    
    elif attack_name == 'fgsm':
      return GradientSignAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=8. / 255,
        clip_min=0., clip_max=1.0,
        targeted=False)
    elif attack_name == 'l2_apgdce':
      aa = AA(model, norm='L2', eps=1., n_iter=20, verbose=False)
      return aa
    elif attack_name == 'linf_apgdce':
      aa = AA(model, norm='Linf', eps=0.031, n_iter=20, verbose=False)
      return aa
    else:
      raise NotImplementedError(f'Attack name not recognized ({attack_name})')
  elif dataset == 'tiny-imagenet-200':
    if attack_name == 'linf_pgd':
      return LinfPGDAttack(
        model, loss_fn=nn.CrossEntropyLoss(reduction="sum"), eps=4. / 255,
        nb_iter=10, eps_iter=2. / 255, rand_init=True, clip_min=0., clip_max=1.0,
        targeted=False)
    elif attack_name == 'l2_apgdce':
      aa = AA(model, norm='L2', eps=1., n_iter=10, verbose=False)
      return aa    
    else:
      raise NotImplementedError(f'Attack name not recognized ({attack_name})')

  else:
    raise NotImplementedError(f'Dataset not recognized ({dataset})')

"""
class AA(APGDAttack): 
  def __init__(self, model, norm='L2', eps=1., n_iter=20, verbose=False):
    super(AA, self).__init__(model, norm=norm, eps=eps, n_iter=n_iter, verbose=verbose)
"""