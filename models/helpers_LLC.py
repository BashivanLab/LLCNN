import torch 
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from typing import List
from torch.nn.modules.utils import _pair


import math
import numpy as np

def car_to_polar(s, hw):
    phis =[]
    rhos =[]
    ind = []
    for i in range(-s*hw//2, s*hw//2):
      for j in range(-s*hw//2, s*hw//2):
        rho = np.sqrt(i**2 + j**2)
        phi = np.arctan2(j, i)
        phis.append(phi)
        rhos.append(rho)
    
    ind = range(0, len(phis))
    
    return phis, rhos, ind
  
phis, rhos, ind = car_to_polar(56, 3)

polar_coord = sorted(zip(phis, rhos, ind), key=lambda pair: pair[1])
polar_coord = [p for _, _, p in sorted(polar_coord, key=lambda pair: pair[0])]
      


def get_closest_factors(num): 
    num_root = int(math.sqrt(num))
    while num % num_root != 0: 
        num_root -= 1
    return num_root, int(num / num_root)


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-
    0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d
  
  
  
  
  
  
  
def evaluate(x, y, sigma):
        """Two dimensional Ricker Wavelet model function."""
        rr_ww = (x ** 2 + y ** 2) / (2 * sigma**2)
        return (1 - rr_ww) * torch.exp(-rr_ww)



def _get_mexican_hat_kernel2d(
    kernel_size: List[int], sigma: float, dtype: torch.dtype, device: torch.device
) -> Tensor:
  
    ksize_half = (kernel_size[0] - 1) * 0.5 # should be change if not symmetric
    xs = torch.linspace(-ksize_half, ksize_half, steps= kernel_size[0])
    ys = torch.linspace(-ksize_half, ksize_half, steps= kernel_size[1])
    
    pdf = torch.zeros([kernel_size[0], kernel_size[1]])
    for i_y, y in enumerate(ys):
        for i_x, x in enumerate(xs):
            w = evaluate(xs[i_x], ys[i_y], sigma)  # sigma too big too many negative correlation noooo too small??? 1.5 for 7 negtive up no// 1.2 for 7 great
            pdf[i_x, i_y] = w #+ np.random.normal(0,1)
        
    kernel2d = pdf / abs(pdf).sum()
    
    return kernel2d
  
  


class KAP2D(nn.Module):
  def __init__ (self, pool_type='mean', noise_std=0.2, kernelsize=3, stride=1, continuous=False, local_conv=False):
    super(KAP2D, self).__init__()
    self.pool_type = pool_type
    self.noise_std = noise_std
    self.kernelsize = kernelsize
    self.stride = stride
    self.continuous = continuous
    self.local_conv = local_conv
    

  def reshape_xin(self, xin, x): 
    if xin.shape == x.shape: 
      return self.reshape_x(xin)
    else: 
      xin_now = torch.squeeze(torch.nn.functional.interpolate(torch.unsqueeze(xin,1), size=x.shape[1:]), 1)
      return self.reshape_x(xin_now)


  def reshape_x(self, x): 
    x_shape = x.shape
    if self.local_conv: 
      #print("polar:", x.shape, self.kh, self.kw, x.permute(0, 2, 3, 1).reshape(x_shape[0], x_shape[2], x_shape[3], self.kh, self.kw).permute(0, 1, 3, 2, 4).reshape(x_shape[0], 1, x_shape[-2]*self.kh, x_shape[-2]*self.kw).shape)
      patches =  x.permute(0, 2, 3, 1).reshape(x_shape[0], x_shape[2], x_shape[3], self.kh, self.kw).permute(0, 1, 3, 2, 4).reshape(x_shape[0], 1, x_shape[-2]*self.kh*x_shape[-2]*self.kw)#.permute(0, 1, 3, 2, 4).reshape(x_shape[0], 1, x_shape[-2]*self.kh * x_shape[-2]*self.kw)

      
      temp = patches # create the tensor
      
      #print(len(polar_coord))
      for i in range(len(polar_coord)):
        temp[ :, :, i] = patches[ :, :, polar_coord[i]]
      
      return temp.reshape(x_shape[0], 1, x_shape[-2]*self.kh, x_shape[-2]*self.kw)
      #return patches
    else: 
      return x.permute(0, 2, 3, 1).reshape(x_shape[0], x_shape[2] * x_shape[3], self.kh, self.kw)

  def undo_reshape_x(self, x, orig_shape): #(batch, 1, h*c1, w*c2) _> (batch, c1*c2, h, w)
    if self.local_conv:
      
      #raise ValueError(f'Pool type {self.pool_type} not recognized.')
      
      patches = x.reshape(orig_shape[0], 1, orig_shape[2] * self.kh *orig_shape[3] * self.kw)#.permute(0, 1, 3, 2, 4).reshape(orig_shape[0], orig_shape[2]*orig_shape[3], self.kh, self.kw)
      
      temp = patches # create the tensor
      
      for i in range(len(polar_coord)):
        temp[ :, :, polar_coord[i]] = patches[ :, :, i]
      
      
      return temp.reshape(orig_shape[0], orig_shape[2], self.kh, orig_shape[3], self.kw).permute(0, 2, 4, 1, 3).reshape(orig_shape[0], self.kh*self.kw, orig_shape[2], orig_shape[3])
      
       
    else: 
       return x.reshape(orig_shape[0], orig_shape[2], orig_shape[3], -1).permute(0, 3, 1, 2)

  def forward(self,x,x_in=None):
    self.kh, self.kw = get_closest_factors(x.shape[1])

    if self.kernelsize <= 1.: 
      self.kernelsize = round(min(self.kw, self.kh) * self.kernelsize)
      if self.kernelsize % 2 == 0: 
          self.kernelsize = self.kernelsize-1
      self.kernelsize = max(self.kernelsize, 3)  # restrict minimum kernelsize to 3

    # if x_in is not empty, permute and reshape appropriately and use for left padding
    if x_in is not None: 
      assert self.continuous

    if self.pool_type is None:
      out = x
    else: 
      x_shape = x.shape
      reshaped_x = self.reshape_x(x)
      # WORKS ONLY IF KAP_STRIDE=1
      if self.continuous and x_in is not None: 
        reshaped_xin = self.reshape_xin(x_in, x)
        reshaped_out = torch.cat((reshaped_xin[:,:,-2*(self.kernelsize-1)//2:,:], reshaped_x), dim=2)
        padding = (0, (self.kernelsize-1)//2)
      else: 
        reshaped_out = reshaped_x
        padding = (self.kernelsize-1)//2
      if self.pool_type == 'mean':
        out = F.avg_pool2d(reshaped_out, kernel_size=self.kernelsize, stride=self.stride, padding=padding, count_include_pad=False)
      elif self.pool_type == 'max':
        out = F.max_pool2d(reshaped_out, kernel_size=self.kernelsize, stride=1, padding=(self.kernelsize-1)//2)
      else: 
        raise ValueError(f'Pool type {self.pool_type} not recognized.')
      out = self.undo_reshape_x(out, x_shape)
    if self.noise_std > 0.: 
      out = out + torch.randn(out.size(), device=out.device) * self.noise_std 
    return out


class GAP2D(KAP2D): 
  def __init__ (self, planes, output_size, pool_type='gaussian', sigma=1., noise_std=0.1, kernelsize=3, stride=1, continuous=False, local_conv=False, device='cpu'):
    super(GAP2D, self).__init__(pool_type=pool_type, noise_std=noise_std, kernelsize=kernelsize, stride=stride, continuous=continuous, local_conv=local_conv)
    self.kh, self.kw = get_closest_factors(planes)
    
    
    self.kernelsize = int(kernelsize) #odd number 

    self.sigma = sigma  * min(self.kw, self.kh)

    self.output_size = output_size
    self.kernel = _get_gaussian_kernel2d([self.kernelsize]*2, [self.sigma]*2, dtype=torch.float32, device=device)
    if local_conv:
      self.kernel = self.kernel.expand(1, 1, self.kernel.shape[0], self.kernel.shape[1])
    else:
      self.kernel = self.kernel.expand(output_size, 1, self.kernel.shape[0], self.kernel.shape[1])
    self.local_conv=local_conv

  def forward(self,x,x_in=None):
  
  
    # if x_in is not empty, permute and reshape appropriately and use for left padding
    if x_in is not None: 
      assert self.continuous

    if self.pool_type is None:
      out = x
    else: 
      x_shape = x.shape
      #print("kap input shape:",self.local_conv, x_shape)
      reshaped_x = self.reshape_x(x)
      #print("kap after reshape:",self.local_conv, reshaped_x.shape)
      # WORKS ONLY IF KAP_STRIDE=1
      if self.continuous and x_in is not None: # for downsample we have to check x_in
        #reshaped_xin = self.reshape_xin(x_in, x)
        #reshaped_out = torch.cat((reshaped_xin[:,:,-2*(self.kernelsize-1)//2:,:], reshaped_x), dim=2)
        #padding = (0, (self.kernelsize-1)//2)
        reshaped_out = reshaped_x
        padding = (self.kernelsize-1)//2
      else: 
        reshaped_out = reshaped_x
        padding = (self.kernelsize-1)//2
        
      #print("continuous",self.local_conv, reshaped_out.shape)
      if self.pool_type == 'gaussian':
        #localize the bug
        #print("kernel", self.kernel.shape, self.kernelsize)
        out = F.conv2d(reshaped_out, self.kernel.to(reshaped_out.device), groups=reshaped_out.shape[1], padding=padding)
        #print("reshape_out:", reshaped_out.shape[1])
        #print("after kap:",self.local_conv, out.shape)
      else: 
        raise ValueError(f'Pool type {self.pool_type} not recognized.')
      out = self.undo_reshape_x(out, x_shape)
      #print("unshape",out.shape)

    return out




class MAP2D(KAP2D): 
  def __init__ (self, planes, output_size, pool_type='mexicanhat', sigma=1., noise_std=0.1, kernelsize=3, stride=1, continuous=False, local_conv=False, device='cpu'):
    super(MAP2D, self).__init__(pool_type=pool_type, noise_std=noise_std, kernelsize=kernelsize, stride=stride, continuous=continuous, local_conv=local_conv)
    self.kh, self.kw = get_closest_factors(planes)
    if self.kernelsize <= 1.: 
      self.orig_kernelsize = self.kernelsize
      k = round(min(self.kw, self.kh) * self.kernelsize)
      if k % 2 == 0: 
          k = k-1
      self.kernelsize = max(k, 3)  # restrict minimum kernelsize to 3

    #if sigma is None: 
       # exponential decay of sigma
       #sigma = max(self.orig_kernelsize * min(self.kw, self.kh), 1)

    self.sigma = sigma 
    self.output_size = output_size
    self.kernel = _get_mexican_hat_kernel2d([self.kernelsize]*2, sigma, dtype=torch.float32, device=device)
    self.kernel = self.kernel.expand(output_size, 1, self.kernel.shape[0], self.kernel.shape[1])

  def forward(self,x,x_in=None):
    # if x_in is not empty, permute and reshape appropriately and use for left padding
    if x_in is not None: 
      assert self.continuous

    if self.pool_type is None:
      out = x
    else: 
      x_shape = x.shape
      reshaped_x = self.reshape_x(x)
      # WORKS ONLY IF KAP_STRIDE=1
      if self.continuous and x_in is not None: 
        reshaped_xin = self.reshape_xin(x_in, x)
        reshaped_out = torch.cat((reshaped_xin[:,:,-2*(self.kernelsize-1)//2:,:], reshaped_x), dim=2)
        padding = (0, (self.kernelsize-1)//2)
        con_pad=0
      else: 
        reshaped_out = reshaped_x
        padding = (self.kernelsize-1)//2  
        con_pad=(self.kernelsize-1)//2
      
      if self.pool_type == 'mexicanhat':
        pad_ = (self.kernelsize-1)//2
        reshaped_out_pad = F.pad(reshaped_out, pad=(pad_,pad_,con_pad,con_pad), mode="reflect")
        out = F.conv2d(reshaped_out_pad, self.kernel.to(reshaped_out.device), groups=reshaped_out_pad.shape[1])
        #out = F.conv2d(reshaped_out, self.kernel.to(reshaped_out.device),padding=pad_, groups=reshaped_out.shape[1])
        #print(self.continuous)
        #print(reshaped_out_pad.shape)
      else: 
        raise ValueError(f'Pool type {self.pool_type} not recognized.')
      out = self.undo_reshape_x(out, x_shape)
    if self.noise_std > 0.: 
      out = out + torch.randn(out.size(), device=out.device) * self.noise_std 
    return out  




# replace the first convolutional layer
class LocallyConnected2d(nn.Module):
    def __init__(self, in_channels, out_channels, output_size, kernel_size, stride, bias=False):
        super(LocallyConnected2d, self).__init__()
        self.output_size = _pair(output_size)
        self.weight = nn.Parameter(
            torch.randn(1, out_channels, in_channels, self.output_size[0], self.output_size[1], kernel_size**2)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.randn(1, out_channels, self.output_size[0], self.output_size[1])
            )
        else:
            self.register_parameter('bias', None)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        
    def forward(self,x):
        #print('archor:', self.output_size)
        #print("Input to conv: ", x.shape)
        
        
        _, c, h, w = x.size()
        kh, kw = self.kernel_size
        dh, dw = self.stride
        x = x.unfold(2, kh, dh).unfold(3, kw, dw)
       # print("After unfold:", x.shape)
        
        
        x = x.contiguous().view(*x.size()[:-2], -1)
        #print("After contiguous", x.shape)

        #if (self.weight.shape[3] - x.shape[3] == 1):
          #x = F.pad(x, pad=(0,0,0,1,0,1))
        ##elif (self.weight.shape[3] - x.shape[3] == 2):
        x = F.pad(x, pad=(0,0,1,1,1,1))
        #else:
          #x = x
       
        
        
        out = (x.unsqueeze(1) * self.weight).sum([2, -1])
        
        #print("Out", x.shape)
        
        if self.bias is not None:
            out += self.bias
        return out



  

class PooledConv(nn.Module):
  def __init__ (self, in_planes, out_planes, kernel_size, stride, padding=0, bias=False, 
  groups=1, dilation=1, pool_type='mean', max_num_pools=1, noise_std=0.2, kap_kernelsize=3, kap_stride=1, 
  continuous=False, local_conv=False, output_size=None):
    super(PooledConv, self).__init__()
    self.in_planes = in_planes
    self.planes = out_planes
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.bias = bias 
    self.groups = groups
    self.dilation = dilation
    self.pool_type = pool_type
    self.max_num_pools = max_num_pools
    self.noise_std = noise_std
    self.kap_kernelsize = kap_kernelsize
    self.kap_stride = kap_stride
    self.continuous = continuous
    self.local_conv = local_conv
    #print(local_conv)
    self.output_size=output_size
    if self.local_conv:
       self.conv = LocallyConnected2d(in_channels=self.in_planes, out_channels=self.planes,
                                      output_size=output_size, kernel_size=kernel_size, stride=stride,
                                      bias=True)
    else: 
       self.conv = nn.Conv2d(in_channels=self.in_planes, out_channels=self.planes, 
                             kernel_size=kernel_size, stride=self.stride, padding=self.padding, 
                             bias=self.bias, groups=groups, dilation=dilation)
       
    self.bn = nn.BatchNorm2d(self.planes)
    self.relu = nn.ReLU(inplace=True)
    
    if self.pool_type in ['gaussian']: 
        self.kap = GAP2D(out_planes, output_size**2, pool_type=pool_type, sigma=kap_kernelsize, noise_std=noise_std, kernelsize=kap_kernelsize, stride=kap_stride, 
                     continuous=continuous, local_conv=local_conv, device=self.conv.weight.device)
    elif self.pool_type in ['mexicanhat']:
        self.kap = MAP2D(out_planes, output_size**2, pool_type=pool_type, sigma=kap_kernelsize, noise_std=noise_std, kernelsize=1, stride=kap_stride, 
                     continuous=continuous, local_conv=local_conv, device=self.conv.weight.device)
        
    elif self.pool_type in ['mean']:
        #print(self.pool_type)
        self.kap = KAP2D(pool_type=pool_type, noise_std=noise_std, kernelsize=kap_kernelsize, stride=kap_stride, 
                         continuous=continuous, local_conv=local_conv)

  def forward(self,x, a= None): 
    

    if self.continuous and a is not None:
      out = self.kap(self.bn(self.conv(x)), a)
    
    else: 
      out = self.kap(self.bn(self.conv(x)))

    return out

