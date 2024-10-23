# source $HOME/jupyter_py3/bin/activate
import torch
import torch.nn as nn
import torchvision
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
from utils import *

import numpy as np
import math
from collections import namedtuple
from torch.utils.data import Dataset, DataLoader
from spacetorch.datasets import DatasetRegistry

#import spacetorch.analyses.rsa as rsa

from utils import get_model
import argparse


def load_pickle(filename):
    import pickle
    with open(f'{filename}.pkl', 'rb') as f:
        return pickle.load(f)


def save_pickle(avg_features, filename):
    import pickle
    with open(f'{filename}.pkl', 'wb') as f:
        pickle.dump(avg_features, f)

def detach(tensor):
    return tensor.detach().cpu().numpy()

def load_model(pool_type, kap_kernelsize, continuous, local_conv, expname, epoch, sel_range=10):
    
    print(f"======loading!=========")
    
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 'continuous', 'local_conv'])
    args = Args(dataset="imagenet", arch="resnet18contopo", pool_type=pool_type, max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    model = get_model(args) #arch change
    
    #gaussin needs _?
    #LLC needs to change two lines

    #/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/old_scripts/cln_resnet18contopo_gaussian_1_0.0progtrain_decay3_init0.5_30epochs_Noncontinuous_Gaussian_D100_V0200/resnet18contopo_58.pt')
    state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_{pool_type}_1_0.0{expname}/resnet18contopo_{epoch}.pt')
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt')
    state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.to(device)
    model.eval();
    
    print(f"======{expname} loading is finished! Now test it!===========")
    
    imagenet = get_imagenet_val_loader(
      batch_size=100, data_path="/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/imagenet/val", norm=False, args=args)
    clean_accuracy(imagenet, model, wanted_samples=2000)
    
    return model
  
  
def load_model_LC(pool_type, kap_kernelsize, continuous, local_conv, expname, epoch, sel_range=10):
    
    print(f"======loading!=========")
    
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 'continuous', 'local_conv'])
    args = Args(dataset="imagenet", arch="resnet18contopo_LLC", pool_type=pool_type, max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    model = get_model(args) #arch change
    

    #/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/old_scripts/cln_resnet18contopo_gaussian_1_0.0progtrain_decay3_init0.5_30epochs_Noncontinuous_Gaussian_D100_V0200/resnet18contopo_58.pt')
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_{pool_type}_1_0.0{expname}/resnet18contopo_{epoch}.pt')
    state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0{expname}/resnet18contopo_LLC_{epoch}.pt')
    state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.to(device)
    model.eval();
    
    print(f"======{expname} loading is finished! Now test it!===========")
    
    imagenet = get_imagenet_val_loader(
      batch_size=100, data_path="/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/imagenet/val", norm=False, args=args)
    clean_accuracy(imagenet, model, wanted_samples=2000)
    
    return model
  
  
def load_model_LC_car(pool_type, kap_kernelsize, continuous, local_conv, expname, epoch, sel_range=10):
    
    print(f"======loading!=========")
    
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 'continuous', 'local_conv'])
    args = Args(dataset="imagenet", arch="resnet18contopo_LLC_car", pool_type=pool_type, max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    model = get_model(args) #arch change
    

    #/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/old_scripts/cln_resnet18contopo_gaussian_1_0.0progtrain_decay3_init0.5_30epochs_Noncontinuous_Gaussian_D100_V0200/resnet18contopo_58.pt')
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_{pool_type}_1_0.0{expname}/resnet18contopo_{epoch}.pt')
    state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_car_gaussian_1_0.0{expname}/resnet18contopo_LLC_car_{epoch}.pt')
    state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.to(device)
    model.eval();
    
    print(f"======{expname} loading is finished! Now test it!===========")
    
    imagenet = get_imagenet_val_loader(
      batch_size=100, data_path="/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/imagenet/val", norm=False, args=args)
    clean_accuracy(imagenet, model, wanted_samples=2000)
    
    return model
  

def load_model_imagenetc(pool_type, kap_kernelsize, continuous, local_conv, expname, epoch, sel_range=10, datapath=''):
    
    print(f"======loading!=========")
    
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 'continuous', 'local_conv'])
    args = Args(dataset="imagenet", arch="resnet18contopo", pool_type=pool_type, max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    model = get_model(args) #arch change
    
    #LLC needs to change two lines

    #/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/old_scripts/cln_resnet18contopo_gaussian_1_0.0progtrain_decay3_init0.5_30epochs_Noncontinuous_Gaussian_D100_V0200/resnet18contopo_58.pt')
    state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_{pool_type}_1_0.0{expname}/resnet18contopo_{epoch}.pt')
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt')
    state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.to(device)
    model.eval();
    
    print(f"======{expname} loading is finished! Now test it!===========")
    
    imagenet = get_imagenet_val_loader(
      batch_size=100, data_path=datapath, norm=False, args=args)
    #imagenet = get_imagenet_val_loader(
      #batch_size=100, data_path="/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/imagenet/val", norm=False, args=args)
    clean_accuracy(imagenet, model, wanted_samples=2000)
    
    return model

  
def load_model_unsupervised(pool_type, kap_kernelsize, continuous, local_conv, expname, epoch, sel_range=10):
    
    print(f"======loading!=========")
    
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 'continuous', 'local_conv', 'projector'])
    args = Args(dataset="imagenet", arch="resnet18contopo", pool_type=pool_type, max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv, projector='8192-8192-8192')
    model = get_model(args) #arch change
    
    #LLC needs to change two lines
    
    from train_unsupervised import BarlowTwins
    model = BarlowTwins(args, model)

    #/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/old_scripts/cln_resnet18contopo_gaussian_1_0.0progtrain_decay3_init0.5_30epochs_Noncontinuous_Gaussian_D100_V0200/resnet18contopo_58.pt')
    state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_{pool_type}_1_0.0{expname}/resnet18contopo_{epoch}.pt')
   
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_LLC_gaussian_1_0.0gaussian_01_LLC/resnet18contopo_LLC_100.pt')
    state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model = model.backbone
    model.to(device)
    model.eval();
    
    print(f"======{expname} loading is finished! Now test it!===========")
    
    imagenet = get_imagenet_val_loader(
      batch_size=100, data_path="/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/imagenet/val", norm=False, args=args)
    clean_accuracy(imagenet, model, wanted_samples=2000)
    
    return model

def load_model_shape_bias(pool_type, kap_kernelsize, continuous, local_conv, expname, epoch, sel_range=10):
    
    print(f"======loading!=========")
    
    Args = namedtuple('nt', ['dataset', 'arch', 'pool_type', 'max_num_pools', 'noise_std', 'kap_kernelsize', 'kap_stride', 'expansion', 'do_prob', 'continuous', 'local_conv'])
    args = Args(dataset="imagenet", arch="resnet18contopo", pool_type=pool_type, max_num_pools=1, noise_std=0., kap_kernelsize=kap_kernelsize, kap_stride=1, expansion=1, do_prob=0., continuous=continuous, local_conv=local_conv)
    model = get_model(args)
    
    #state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/old_scripts/cln_resnet18contopo_gaussian_1_0.0progtrain_decay3_init0.5_30epochs_Noncontinuous_Gaussian_D100_V0200/resnet18contopo_58.pt')
    state_dict = torch.load(f'/home/xinyuq/scratch/results/kernel_avepool/trained_models/imagenet/cln_resnet18contopo_{pool_type}_1_0.0{expname}/resnet18contopo_{epoch}.pt')
    state_dict['state_dict'] = {k.replace('module.', ''): state_dict['state_dict'][k] for k in state_dict['state_dict'].keys()}
    model.load_state_dict(state_dict['state_dict'], strict=True)
    model.to(device)
    model.eval();
    
    print(f"======{expname} loading is finished! Now test it!===========")
    
    imagenet = get_imagenet_val_loader(
      batch_size=1, data_path="/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/texture-vs-shape/stimuli/style-transfer-preprocessed-512", norm=False, args=args)
    acc1 = clean_accuracy_shape_bias(imagenet, model, wanted_samples=2000)
    
    imagenet = get_imagenet_val_loader(
      batch_size=1, data_path="/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/texture-vs-shape/stimuli/style-transfer-preprocessed-512_texture", norm=False, args=args)
    acc2 = clean_accuracy_shape_bias(imagenet, model, wanted_samples=2000)
    
    acc = acc1/(acc1+acc2)
    print(f'the shape bias is: {100*acc} %')
    
    return model




def load_layers_names1(model):
  #[64, 144, 256, 529]
  #bn = nn.functional.batch_norm(256)
  layers = [
    

        model.layer1[0].conv1,
        model.layer1[0].conv2,


        model.layer1[1].conv1,
        model.layer1[1].conv2,

    

        model.layer2[0].conv1,
        model.layer2[0].conv2,

    

        model.layer2[1].conv1,
        model.layer2[1].conv2,
    
    
        model.layer3[0].conv1,
        model.layer3[0].conv2,
    

        model.layer3[1].conv1,
        model.layer3[1].conv2,

    

        model.layer4[0].conv1,
        model.layer4[0].conv2,


        model.layer4[1].conv1,
        model.layer4[1].conv2,
 
         ]

  layers_names_raw =[]
  for name, layer in model.named_modules():
    if isinstance(layer, type(model.layer1[0])):
        layers_names_raw.append(name)
    if isinstance(layer, type(model.layer1[0].conv1)):
        layers_names_raw.append(name)
    if isinstance(layer, type(model.layer1[0].conv1.conv)):
        layers_names_raw.append(name)
  layers_names = layers_names_raw[2:]



  #layers=[model.layer4[1].relu]
  #layers_names=['relu'] 
  return layers, layers_names  



def load_layers_names(model):
  #[64, 144, 256, 529]
  #bn = nn.functional.batch_norm(256)
  layers = [
    
        model.layer1[0],
        model.layer1[0].conv1,
        model.layer1[0].conv1.conv,
        model.layer1[0].conv2,
        model.layer1[0].conv2.conv,
    
        model.layer1[1],
        model.layer1[1].conv1,
        model.layer1[1].conv1.conv,
        model.layer1[1].conv2,
        model.layer1[1].conv2.conv,
    
        model.layer2[0],
        model.layer2[0].conv1,
        model.layer2[0].conv1.conv,
        model.layer2[0].conv2,
        model.layer2[0].conv2.conv,
    
        model.layer2[0].downsample[0],
        model.layer2[0].downsample[0].conv,
        
        model.layer2[1],
        model.layer2[1].conv1,
        model.layer2[1].conv1.conv,
        model.layer2[1].conv2,
        model.layer2[1].conv2.conv,
    
    
        model.layer3[0],
        model.layer3[0].conv1,
        model.layer3[0].conv1.conv,
        model.layer3[0].conv2,
        model.layer3[0].conv2.conv,
    
        model.layer3[0].downsample[0],
        model.layer3[0].downsample[0].conv,
    
        model.layer3[1],
        model.layer3[1].conv1,
        model.layer3[1].conv1.conv,
        model.layer3[1].conv2,
        model.layer3[1].conv2.conv,
    
        model.layer4[0],
        model.layer4[0].conv1,
        model.layer4[0].conv1.conv,
        model.layer4[0].conv2,
        model.layer4[0].conv2.conv,
    
        model.layer4[0].downsample[0],
        model.layer4[0].downsample[0].conv,
    
        model.layer4[1],
        model.layer4[1].conv1,
        model.layer4[1].conv1.conv,
        model.layer4[1].conv2,
        model.layer4[1].conv2.conv,
         ]

  layers_names_raw =[]
  for name, layer in model.named_modules():
    if isinstance(layer, type(model.layer1[0])):
        layers_names_raw.append(name)
    if isinstance(layer, type(model.layer1[0].conv1)):
        layers_names_raw.append(name)
    if isinstance(layer, type(model.layer1[0].conv1.conv)):
        layers_names_raw.append(name)
  layers_names = layers_names_raw[2:]

  # for test
  layers = layers[:20]
  layers_names = layers_names[:20]

  #layers=[model.layer4[1].relu]
  #layers_names=['relu'] 
  return layers, layers_names  

def load_layers_names_forcontinuous(model):
  #[64, 144, 256, 529]
  #bn = nn.functional.batch_norm(256)
  layers = [
    
        model.layer1[0].conv1,
        model.layer1[0].conv2,
        model.layer1[1].conv1,
        model.layer1[1].conv2,

    

        model.layer2[0].conv1,
        model.layer2[0].conv2,
        model.layer2[1].conv1,
        model.layer2[1].conv2,

    
    

        model.layer3[0].conv1,
        model.layer3[0].conv2,
        model.layer3[1].conv1,
        model.layer3[1].conv2,
  
    

        model.layer4[0].conv1,
        model.layer4[0].conv2,
        model.layer4[1].conv1,
        model.layer4[1].conv2,

         ]

  #layers_names_raw =[ 'layer1.0', 'layer1.1', 'layer2.0',  'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0',  'layer4.1']
  layers_names_raw =[ 'layer1', 'layer2', 'layer3', 'layer4']

  for name, layer in model.named_modules():
    
    if isinstance(layer, type(model.layer1[0].conv1)):
        layers_names_raw.append(name)

  #layers_names = layers_names_raw[2:]

  # for test


  #layers=[model.layer4[1].relu]
  #layers_names=['relu'] 
  return layers, layers_names_raw



def load_layers_names_forcontinuous_norm(model):
  #[64, 144, 256, 529]
  #bn = nn.functional.batch_norm(256)
  layers = [
    
        model.layer1[0].bn1,
        model.layer1[0].bn2,
        model.layer1[1].bn1,
        model.layer1[1].bn2,

    

        model.layer2[0].bn1,
        model.layer2[0].bn2,
        model.layer2[1].bn1,
        model.layer2[1].bn2,

    
    

        model.layer3[0].bn1,
        model.layer3[0].bn2,
        model.layer3[1].bn1,
        model.layer3[1].bn2,
  
    

        model.layer4[0].bn1,
        model.layer4[0].bn2,
        model.layer4[1].bn1,
        model.layer4[1].bn2,

         ]

  #layers_names_raw =[ 'layer1.0', 'layer1.1', 'layer2.0',  'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0',  'layer4.1']
  layers_names_raw =[ 'layer1', 'layer2', 'layer3', 'layer4']

  for name, layer in model.named_modules():
    
    if isinstance(layer, type(model.layer1[0].conv1)):
        layers_names_raw.append(name)

  #layers_names = layers_names_raw[2:]

  # for test


  #layers=[model.layer4[1].relu]
  #layers_names=['relu'] 
  return layers, layers_names_raw



def load_layers_names_forcontinuous1(model):
  #[64, 144, 256, 529]
  #bn = nn.functional.batch_norm(256)
  layers = [
    

    
    

        model.layer4[0].conv1,
        model.layer4[0].conv2,
        model.layer4[1].conv1,
        model.layer4[1].conv2,
  
    



         ]

  #layers_names_raw =[ 'layer1.0', 'layer1.1', 'layer2.0',  'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0',  'layer4.1']
  layers_names_raw =['layer4']



  #layers_names = layers_names_raw[2:]

  # for test


  #layers=[model.layer4[1].relu]
  #layers_names=['relu'] 
  return layers, layers_names_raw

def load_layers_names_forcontinuous3(model):
  #[64, 144, 256, 529]
  #bn = nn.functional.batch_norm(256)
  layers = [
    

    
    

        model.layer3[0].conv1,
        model.layer3[0].conv2,
        model.layer3[1].conv1,
        model.layer3[1].conv2,
  
    



         ]

  #layers_names_raw =[ 'layer1.0', 'layer1.1', 'layer2.0',  'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0',  'layer4.1']
  layers_names_raw =['layer3']



  #layers_names = layers_names_raw[2:]

  # for test


  #layers=[model.layer4[1].relu]
  #layers_names=['relu'] 
  return layers, layers_names_raw

def load_layers_names_forcontinuous2(model):
  #[64, 144, 256, 529]
  #bn = nn.functional.batch_norm(256)
  layers = [
    

    
    

        model.layer1[0].conv1,
        model.layer1[0].conv2,
        model.layer1[1].conv1,
        model.layer1[1].conv2,
  
    



         ]

  #layers_names_raw =[ 'layer1.0', 'layer1.1', 'layer2.0',  'layer2.1', 'layer3.0', 'layer3.1', 'layer4.0',  'layer4.1']
  layers_names_raw =['layer1']



  #layers_names = layers_names_raw[2:]

  # for test


  #layers=[model.layer4[1].relu]
  #layers_names=['relu'] 
  return layers, layers_names_raw



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
  ds = datasets.ImageFolder(valdir, val_transforms)
  loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, 
  shuffle=shuffle, num_workers=1, pin_memory=True)
  loader.name = "imagenet_validation"
  return loader





        
        
def clean_accuracy(test_loader, model, wanted_samples=2000):       
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(1000)]
        n_class_samples = [0 for i in range(1000)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            print(n_correct/n_samples*100, n_correct, n_samples)
            break
            for i in range(len(labels)):
              label = labels[i]
              pred = predicted[i]
              if (label== pred):
                n_class_correct[label] += 1
              n_class_samples[label] += 1
            
            if n_samples >= 5000:
              break
              
              
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        
        #for i in range(1000):
         # acc = 100.0 * n_class_correct[i] / n_class_samples[i]
         # print(f'Accuracy of {i}: {acc} %')
        
        
def clean_accuracy_shape_bias(test_loader, model, wanted_samples=2000):   
  
        label_list = sorted(["knife", "keyboard", "elephant", "bicycle", "airplane",
            "clock", "oven", "chair", "bear", "boat", "cat",
            "bottle", "truck", "car", "bird", "dog"])
         
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10000)]
        n_class_samples = [0 for i in range(10000)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        # max returns (value ,index)
            from temp import probabilities_to_decision
            mapping = probabilities_to_decision.ImageNetProbabilitiesTo16ClassesMapping()
            import torch.nn.functional as F
            #print(outputs.shape)
            outputs =   F.softmax(outputs, dim=1).squeeze()
            #print(outputs.shape)
            
            decision_from_16_classes = mapping.probabilities_to_decision(outputs.cpu().detach().numpy())
            #print(decision_from_16_classes)
            #print(labels)
            n_samples += labels.size(0)
            n_correct += (decision_from_16_classes == label_list[labels.item()])
            

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        return acc
        
def clean_accuracy_c(test_loader, model, wanted_samples=2000):       
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(1000)]
        n_class_samples = [0 for i in range(1000)]
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
        # max returns (value ,index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            print(n_correct/n_samples*100, n_correct, n_samples)
            for i in range(len(labels)):
              label = labels[i]
              pred = predicted[i]
              if (label== pred):
                n_class_correct[label] += 1
              n_class_samples[label] += 1
            
            if n_samples >= 5000:
              break
              
              
        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')
        
        #for i in range(1000):
         # acc = 100.0 * n_class_correct[i] / n_class_samples[i]
         # print(f'Accuracy of {i}: {acc} %')
               

def get_coord1(kw, kh):
  
    coord = []
    for x in range(kh): #i dont fucking understand
        for y in range(kw):
            coord.append([x,y]+ np.random.normal(0,0.1,2))
    
    coord = np.array(coord)
    
    return coord    
        
    
def get_coord(kw, kh):
  
    coord = []
    for y in range(kh): #i dont fucking understand
        for x in range(kw):
            coord.append([x,y]+ np.random.normal(0,0.1,2))
    
    coord = np.array(coord)
    
    return coord
  
def get_coord0(kw, kh):
  
    coord = []
    for y in range(kh): #i dont fucking understand
        for x in range(kw):
            coord.append([x,y])
    
    coord = np.array(coord)
    
    return coord

def get_closest_factors(num): 
    num_root = int(math.sqrt(num))
    while num % num_root != 0: 
        num_root -= 1
    return num_root, int(num / num_root)   
        
