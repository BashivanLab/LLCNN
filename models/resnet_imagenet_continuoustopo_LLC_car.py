import torch
import torch.nn as nn
from .helpers_LLC_car import *


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, pool_type='mean', max_num_pools=1, noise_std=0.2, 
            kap_kernelsize=3, continuous=False, local_conv=False, output_size=None):
    """3x3 convolution with padding"""
    return PooledConv(in_planes, out_planes, kernel_size=3, stride= stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation,
                     pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, 
                     continuous=continuous, local_conv=local_conv, output_size=output_size)


def conv1x1(in_planes, out_planes, stride=1, pool_type='mean', max_num_pools=2, noise_std=0.2, kap_kernelsize=3, continuous=False, local_conv=False, output_size=None):
    """1x1 convolution"""
    return PooledConv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False, 
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, 
    continuous=continuous, local_conv=local_conv, output_size=output_size)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, local_conv=False, output_size=None, continuous=False, a=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, 
        pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, local_conv=local_conv, output_size=output_size, continuous=continuous)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 
        pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, local_conv=local_conv, output_size=output_size, continuous=continuous)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.a =a

    def forward(self, x, a):
        identity = x
        
       

        out = self.conv1(x, a)
        out = self.bn1(out)
        t = out
        out = self.relu(out)

        out = self.conv2(out, t)
        out = self.bn2(out)
        

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        t = out
        
        out = self.relu(out)

        return out, t

class sequentialMultiInput(nn.Sequential):
	def forward(self, *inputs):
		for module in self._modules.values():
			if type(inputs) == tuple:
				inputs = module(*inputs)
			else:
				inputs = module(inputs)
		return inputs



class ResNet(nn.Module):

    def __init__(self, block, layers, planes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, num_classes=1000, 
                 zero_init_residual=False, groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, continuous=False, local_conv=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = planes[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = PooledConv(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False, 
        pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, output_size=112, continuous=continuous, local_conv=False)   #, local_conv=True, output_size=109
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, planes[0], layers[0], #attention
                                       pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, output_size=56, continuous=continuous, local_conv=local_conv[0], a=None)  # , local_conv=True, output_size=54
        self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], 
                                       pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, output_size=28, continuous=continuous, local_conv=local_conv[1])  # , local_conv=True, output_size=27
        self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], 
                                       pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, output_size=14, continuous=continuous, local_conv=local_conv[2])
        self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], 
                                       pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, output_size=7, continuous=continuous, local_conv=local_conv[3])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(planes[3] * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, pool_type, max_num_pools, noise_std, kap_kernelsize=3, stride=1, dilate=False, local_conv=False, output_size=None, continuous=False, a=None):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride=stride, 
                pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, output_size=output_size, continuous=continuous, local_conv=local_conv),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, pool_type=pool_type, 
                      max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize,
                      stride=stride, downsample=downsample, groups=self.groups,
                      base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer, local_conv=local_conv, output_size=output_size, continuous=continuous, a=None))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, pool_type=pool_type, 
                                max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize,
                                groups=self.groups, base_width=self.base_width, 
                                dilation=self.dilation, norm_layer=norm_layer, local_conv=local_conv, output_size=output_size, continuous=continuous, a=None))

        return sequentialMultiInput(*layers) # only takes one input and one output???
        

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        t = x
        x = self.relu(x)
        x = self.maxpool(x)
        
        x,t  = self.layer1(x, t)
        x,t = self.layer2(x, t)
        x,t = self.layer3(x, t)
        x,_ = self.layer4(x, t)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x


def ResNet18(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False, local_conv=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], planes=[9, 144, 256, 529], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous, local_conv=[True, False, False, False])

def ResNet18_2(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False, local_conv=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], planes=[64, 9, 256, 529], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous, local_conv=[False, True, False, False])
    
def ResNet18_3(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False, local_conv=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], planes=[64, 144,9, 529], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous, local_conv=[False, False, True, False])
    
def ResNet18_4(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False, local_conv=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], planes=[64, 144, 256, 9], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous, local_conv=[False, False, False, True])
    
    
    

def ResNet18WideX4(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False, local_conv=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], planes=[64*4, 144*4, 256*4, 529*4], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous, local_conv=local_conv)


def ResNet18WideX9(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False, local_conv=False):
    return ResNet(BasicBlock, [2, 2, 2, 2], planes=[64*9, 144*9, 256*9, 529*9], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous, local_conv=local_conv)


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], planes=[64, 144, 256, 529], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous)


def ResNet50WideX4(num_classes, pool_type, max_num_pools, noise_std, kap_kernelsize=3, continuous=False):
    return ResNet(Bottleneck, [3, 4, 6, 3], planes=[64*4, 144*4, 256*4, 529*4], num_classes=num_classes,
    pool_type=pool_type, max_num_pools=max_num_pools, noise_std=noise_std, kap_kernelsize=kap_kernelsize, continuous=continuous)


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])