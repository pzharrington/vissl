# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Sequence

import numpy as np
import torchvision.transforms as pth_transforms
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from numpy import ndarray
import torch


@register_transform("ImgNpyToMultiCrop")
class ImgNpyToMultiCrop(ClassyTransform):
    """
    Convert a numpy array  to Multi-resolution Crops.
    The input is a numpy array and output is the list of image crops.
    The ndarrays are intermediately converted to torch.Tensor and back 
    since I am too lazy to rewrite RandomResizedCrop, so this could 
    probably be sped up slightly with a better implementation.

    This transform was proposed in SwAV - https://arxiv.org/abs/2006.09882
    """

    def __init__(
        self,
        total_num_crops: int,
        num_crops: Sequence[int],
        size_crops: Sequence[int],
        crop_scales: Sequence[Sequence[float]],
        flip_axes: bool
    ):
        """
        Returns total_num_crops square crops of an image. Each crop is a random crop
        extracted according to the parameters specified in size_crops and crop_scales.
        For ease of use, one can specify `num_crops` which removes the need to repeat
        parameters.

        Args:
            total_num_crops (int): Total number of crops to extract
            num_crops (List or Tuple of ints): Specifies the number of `type' of crops.
            size_crops (List or Tuple of ints): Specifies the height (height = width)
                                                of each patch
            crop_scales (List or Tuple containing [float, float]): Scale of the crop
            flip_axes (bool): whether or not to transpose axes before and after resizing

        Example usage:
        - (total_num_crops=2, num_crops=[1, 1],
           size_crops=[224, 96], crop_scales=[(0.14, 1.), (0.05, 0.14)])
           Extracts 2 crops total of size 224x224 and 96x96
        - (total_num_crops=2, num_crops=[1, 2],
           size_crops=[224, 96], crop_scales=[(0.14, 1.), (0.05, 0.14)])
           Extracts 3 crops total: 1 of size 224x224 and 2 of size 96x96
        """

        assert np.sum(num_crops) == total_num_crops
        assert len(size_crops) == len(num_crops)
        assert len(size_crops) == len(crop_scales)

        transforms = []
        for num, size, scale in zip(num_crops, size_crops, crop_scales):
            transforms.extend(
                [pth_transforms.RandomResizedCrop(size, scale=scale)] * num
            )
        self.transforms = transforms
        self.flipax = [0,2] if flip_axes else [1,1]

    def flip_fn(self, x):
        return np.swapaxes(x, self.flipax[0], self.flipax[1])

    def __call__(self, image: ndarray) -> List[ndarray]:
        return [self.flip_fn(transform(torch.from_numpy(self.flip_fn(image))).numpy()) for transform in self.transforms]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ImgNpyToMultiCrop":
        """
        Instantiates ImgNpyToMultiCrop from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ImgNpyToMultiCrop instance.
        """
        return cls(**config)
