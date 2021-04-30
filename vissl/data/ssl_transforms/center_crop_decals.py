import numpy as np
import logging
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Any, Dict

@register_transform("CenterCropDecals")
class CenterCropDecals(ClassyTransform):
    '''takes in image of size (npix, npix, nchannel), 
    jitters by uniformly drawn (-jitter_lim, jitter_lim),
    and returns (outdim, outdim, nchannel) central pixels'''

    def __init__(self, outdim, jitter_lim):
        self.outdim = outdim
        self.jitter_lim = jitter_lim

    def __call__(self, image):
        if self.jitter_lim:
            center_x = image.shape[0]//2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
            center_y = image.shape[0]//2 + int(np.random.randint(-self.jitter_lim, self.jitter_lim+1, 1))
        else:
            center_x = image.shape[0]//2
            center_y = image.shape[0]//2
        offset = self.outdim//2

        return image[(center_x-offset):(center_x+offset), (center_y-offset):(center_y+offset)].astype(np.float32)
        
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "CenterCropDecals":
        """
        Instantiates CenterCropDecals from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            CenterCropDecals instance.
        """
        outdim = config['outdim'] # default: 96
        jitter_lim = config['jitter_lim'] # default: 7
        return cls(outdim=outdim, jitter_lim=jitter_lim)
