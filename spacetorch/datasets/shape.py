from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Type

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
import torchvision
from skimage import io
from torch.utils.data import Dataset
import xarray as xr

from spacetorch.utils.array_utils import flatten, tstat
ComposedTransforms = Type[torchvision.transforms.transforms.Compose]
import h5py

SHAPE_TRANSFORMS = torchvision.transforms.Compose(
    [
        #torchvision.transforms.CenterCrop(90),
        torchvision.transforms.Resize(224),
        torchvision.transforms.ToTensor()
    ]
)

class ShapeData(Dataset):
    """VPNL fLoc Dataset"""

    def __init__(self, floc_dir: str, transforms: Optional[ComposedTransforms] = SHAPE_TRANSFORMS):
        self.transforms = transforms

        file_dir = Path('/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/Contour')#'/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/shape'
        #file_dir = Path('/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/shape')
        #file_dir = Path('/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/MEI')
        
        #file_dir = Path("/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/algonauts_2023_tutorial_data/subj04/test_split/test_images")
        #file_dir = Path('/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/MEI_cluster')
        
        
        self.file_list = sorted(file_dir.glob("*.png"))
    
        raw = [f"{f}" for f in self.file_list]
       
        
        self.labels = raw #[CATEGORIES.index(category) for category in raw_categories]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img = io.imread(self.file_list[idx])
        # expand third dim
        img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)#MEI
        img[img == 128] = 255
        
        target = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)

        return img, target
