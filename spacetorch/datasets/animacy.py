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

# types
ComposedTransforms = Type[torchvision.transforms.transforms.Compose]

CATEGORIES: List[str] = [
    "animacy",
    "inanimacy",
]


@dataclass
class Contrast:
    name: str
    color: str
    on_categories: List[str]
    off_categories: Optional[List[str]] = None


DOMAIN_CONTRASTS: List[Contrast] = [
    Contrast(name="ANI", color="#33B0FF", on_categories=["animacy"]),
    Contrast(name="IANI", color="#20913E", on_categories=["inanimacy"]),

]



class Scaling(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = (image - (-2.117904))/(2.64 - (-2.117904))
        return {'image': image,
                'landmarks': landmarks}

    
IMAGENET_TRANSFORMS = torchvision.transforms.Compose(
    [
         torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
       
        #torchvision.transforms.Normalize(
            #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #),
    ]
)
    
    

class AnimacyData(Dataset):
    """VPNL fLoc Dataset"""

    def __init__(self, animacy_dir: str, inanimacy_dir: str, transforms: Optional[ComposedTransforms] = None):
        self.transforms = IMAGENET_TRANSFORMS
        
        #additional contrast
        #animacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Big"
        #inanimacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Small"
        print(1)
        #animacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Animate"
        #inanimacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Inanimate"
        
        
        #animacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/CC/corner"
        #inanimacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/CC/curve"
        
    
        #animacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/NML/five"
        #inanimacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/NML/else"
        
        animacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Big-Animate"
        inanimacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Small-Animate"
        
        #animacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Big-Inanimate"
        #inanimacy_dir = "/home/xinyuq/projects/def-bashivan/xinyuq/kernel_avpool/data/datasets/AnimacySize/Small-Inanimate"
        

        file_dir1 = Path(animacy_dir)
        self.file_list1 = sorted(file_dir1.glob("*.jpg"))# animacy is jpg fuck it is lot of trouble; corner is png
        raw_categories1 = [0 for f in self.file_list1]#[f.stem.split("-")[0] for f in self.file_list1]
        self.labels1 = [0 for category in raw_categories1]
        
        print(self.labels1)
        
        file_dir2 = Path(inanimacy_dir)
        self.file_list2 = sorted(file_dir2.glob("*.jpg"))
        raw_categories2 = [1 for f in self.file_list2]#[f.stem.split("-")[0] for f in self.file_list2]
        self.labels2 = [1 for category in raw_categories2]
        
        print(self.labels2)
        
        self.file_list = self.file_list1 + self.file_list2
        self.labels = self.labels1 + self.labels2

    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

       
        img = io.imread(self.file_list[idx])
        # expand third dim
        #img = np.repeat(img[:, :, np.newaxis], 3, axis=-1)  # animacy does not need this?? write if loop to add this line.  okay corner needs this

        
        
        target = self.labels[idx]
        if self.transforms:
            img = self.transforms(img)

        return img, target


class AnimacyResponses:
    def __init__(self, features: np.ndarray, labels: np.ndarray):

        self.domains = {
            "animacy": ["animacy"],
            "inanimacy": ["inanimacy"],
         
        }

        self.categories = [
            category for domain in self.domains.values() for category in domain
        ]

        self._data = xr.DataArray(
            data=flatten(features),
            coords={
                "categories": ("image_idx", [CATEGORIES[idx] for idx in labels]),
                "category_indices": ("image_idx", labels),
            },
            dims=["image_idx", "unit_idx"],
        )

        self._drop_scrambled()

    def _drop_scrambled(self):
        self._data = self._data.where(self._data.categories != "scrambled", drop=True)

    def selectivity(
        self,
        on_categories: List[str],
        off_categories: Optional[List[str]] = None,
        selectivity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = tstat,
    ) -> np.ndarray:
        """Computes the specified selectivity function for the given category
        Returns:
            Category-selectivity of each unit
        """

        # get indices of labels that match the "on" and "off" criteria
        on_indices = self._data.categories.isin(["animacy"])
        #print(self._data[on_indices])
        off_indices = self._data.categories.isin(["inanimacy"])
        #print(self._data[off_indices])
        return selectivity_fn(
            self._data[on_indices].values, self._data[off_indices].values
        )

    def selectivity_category_average(
        self,
        on_categories: List[str],
        off_categories: Optional[List[str]] = None,
        selectivity_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = tstat,
    ) -> np.ndarray:
        """Computes the specified selectivity function for the given category, but
        averages within each category first

        Returns:
            Category-selectivity of each unit
        """
        if off_categories is None:
            off_categories = list(set(self.categories) - set(on_categories))

        on_responses = []
        for cat in on_categories:
            responses = self._data[self._data.categories == cat].mean("image_idx")
            on_responses.append(responses.values)

        off_responses = []
        for cat in off_categories:
            responses = self._data[self._data.categories == cat].mean("image_idx")
            off_responses.append(responses.values)

        stacked_on_responses = np.stack(on_responses)
        stacked_off_responses = np.stack(off_responses)

        return selectivity_fn(stacked_on_responses, stacked_off_responses)

    def plot_rsm(
        self,
        ax,
        average_by_category: bool = False,
        cmap: str = "gist_heat",
        vmin: float = -0.25,
        vmax: float = 1.0,
        add_colorbar: bool = True,
        add_ticks: bool = True,
    ):
        if average_by_category:
            rsm = np.corrcoef(self.sorted_by_category.groupby("categories").mean())
        else:
            rsm = np.corrcoef(self.sorted_by_category)
        np.fill_diagonal(rsm, np.nan)
        img = ax.imshow(rsm, cmap=cmap, vmin=vmin, vmax=vmax)

        if add_colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ax.figure.colorbar(img, cax=cax, orientation="vertical")

        if add_ticks:
            category_labels = CATEGORIES[:-1]
            images_per_category = len(rsm) // len(category_labels)

            ticks = np.arange(
                images_per_category // 2,
                stop=len(rsm) + images_per_category // 2,
                step=images_per_category,
            )
            ax.set_xticks(ticks)
            ax.set_xticklabels(category_labels, rotation=30)
            ax.set_yticks(ticks)
            ax.set_yticklabels(category_labels)

    @property
    def sorted_by_category(self) -> xr.DataArray:
        return self._data.sortby("category_indices")

    def __len__(self) -> int:
        return self._data.sizes["unit_idx"]
