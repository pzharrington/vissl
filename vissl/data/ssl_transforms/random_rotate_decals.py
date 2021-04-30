import skimage
import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Any, Dict

@register_transform("RandomRotateDecals")
class RandomRotateDecals(ClassyTransform):

    def __init__(self):
        # No init necessary
        return

    def __call__(self, image):
        """
        Takes in image of size (npix, npix, nchannel), flips l/r and or u/d, then rotates (0,360)
        """
        if np.random.randint(0, 2)==1:
            image = np.flip(image, axis=0)
        if np.random.randint(0, 2)==1:
            image = np.flip(image, axis=1)
        return skimage.transform.rotate(image, np.float32(360*np.random.rand(1))).astype(np.float32)
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "RandomRotateDecals":
        """
        Instantiates RandomRotateDecals from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            RandomRotateDecals instance.
        """
        return cls()
