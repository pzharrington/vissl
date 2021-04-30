import skimage
import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Any, Dict

@register_transform("SizeScaleDecals")
class SizeScaleDecals(ClassyTransform):
    '''takes in image of size (npix, npix, nchannel), and scales the size larger or smaller
    anti-aliasing should probably be enabled when down-sizing images to avoid aliasing artifacts
    
    This augmentation changes the number of pixels in an image. After sizescale, we still need enough
    pixels to allow for jitter crop to not run out of bounds. Therefore, 
    
    scale_min >= (outdim + 2*jitter_lim)/indim
    
    if outdim = 96, and indim=152 and jitter_lim = 7, then scale_min >= 0.73.
    
    When using sizescale, there is a small possibility that one corner of the image can be set to 0 in randomrotate,
    then the image can be scaled smaller, and if the image is jittered by near the maximum allowed value, that these
    0s will remain in a corner of the final image. Adding Gaussian noise after all the other augmentations should 
    remove any negative effects of this small 0 patch.
    '''

    def __init__(self, scale_min, scale_max):

        if scale_min < 0.73:
            print('scale_min reset to minimum value, given 152 pix image and 96 pix out')
            scale_min = 0.73
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __call__(self, image):

        scalei = np.random.uniform(self.scale_min, self.scale_max)
        return skimage.transform.rescale(image, scalei, anti_aliasing=False, multichannel=True).astype(np.float32)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "SizeScaleDecals":
        """
        Instantiates SizeScaleDecals from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            SizeScaleDecals instance.
        """
        scale_min = config['scale_min'] # default: 0.9
        scale_max = config['scale_max'] # default: 1.1
        return cls(scale_min=scale_min, scale_max=scale_max)
