import skimage
import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Any, Dict

@register_transform("GaussianBlurDecals")
class GaussianBlurDecals(ClassyTransform):
    '''adds Gaussian PSF blur consistent from distribution fit to decals psf_size
    from sweep catalogues as measured from 2e6 spectroscopic samples. 
    Images have already been filtered by PSF when observed on sky, so we do not want 
    to smooth images using a total smoothing, we only want to augment images by the 
    difference in smoothings between various objects in the survey.
    
    sigma = psf_size / pixelsize / 2.3548,
    where psf_size from the sweep catalogue is fwhm in arcsec, pixelsize=0.262 arcsec, 
    and 2.3548=2*sqrt(2ln(2)) converst from fwhm to Gaussian sigma
    
    PSF in each channel is uncorrelated, as images taken at different times/telescopes.

    A lognormal fit matches the measured PSF distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random 
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):
    
    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # g: (min, max)=(1.3233109, 5), Lognormal fit (shape, loc, scale)=(0.2109966, 1.0807153, 1.3153171)
    # r: (min, max)=(1.2667341, 4.5), Lognormal fit (shape, loc, scale)=(0.3008485, 1.2394326, 0.9164757)
    # z: (min, max)=(1.2126263, 4.25), Lognormal fit (shape, loc, scale)=(0.3471172, 1.1928363, 0.8233702)
    '''

    def __init__(self, im_ch, uniform):
        self.im_ch = im_ch
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2109966, 0.3008485, 0.3471172])
        self.loc_dist  = np.array([1.0807153, 1.2394326, 1.1928363])
        self.scale_dist  = np.array([1.3153171, 0.9164757, 0.8233702])

        self.sigma_dist  = np.log(self.scale_dist)

        self.psf_ch_min = np.array([1.3233109, 1.2667341, 1.2126263])
        self.psf_ch_max = np.array([5., 4.5, 4.25])

    def __call__(self, image):
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.psf_ch_min, self.psf_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                image[:,:,i] = skimage.filters.gaussian(image[:,:,i], sigma=self.sigma_augment[i], mode='reflect')

        return image.astype(np.float32)


    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GaussianBlurDecals":
        """
        Instantiates GaussianBlurDecals from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            GaussianBlurDecals instance.
        """
        im_ch = config['im_ch'] # default: 3
        uniform = config['uniform'] # default: False
        return cls(im_ch=im_ch, uniform=uniform)
