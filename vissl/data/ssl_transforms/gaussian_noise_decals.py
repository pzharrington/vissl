import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Any, Dict

@register_transform("GaussianNoiseDecals")
class GaussianNoiseDecals(ClassyTransform):
    '''adds Gaussian noise consistent from distribution fit to decals south \sigma_{pix,coadd}
    (see https://www.legacysurvey.org/dr9/nea/) as measured from 43e6 samples with zmag<20 .
    Images already have noise level when observed on sky, so we do not want 
    to add a total amount of noise, we only want to augment images by the 
    difference in noise levels between various objects in the survey.
    
    1/sigma_pix^2 = psf_depth * [4pi (psf_size/2.3548/pixelsize)^2],
    where psf_size from the sweep catalogue is fwhm in arcsec, pixelsize=0.262 arcsec, 
    and 2.3548=2*sqrt(2ln(2)) converst from fwhm to Gaussian sigma
    
    noise in each channel is uncorrelated, as images taken at different times/telescopes.

    A lognormal fit matches the measured noise distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random 
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):
    
    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # g: (min, max)=(0.001094, 0.013), Lognormal fit (shape, loc, scale)=(0.2264926, -0.0006735, 0.0037602)
    # r: (min, max)=(0.001094, 0.018), Lognormal fit (shape, loc, scale)=(0.2431146, -0.0023663, 0.0067417)
    # z: (min, max)=(0.001094, 0.061), Lognormal fit (shape, loc, scale)=(0.1334844, -0.0143416, 0.0260779)
    '''

    def __init__(self, mean, im_ch, uniform):
        self.mean = mean
        self.im_ch = im_ch
        self.uniform = uniform

        # Log normal fit paramaters
        self.shape_dist = np.array([0.2264926, 0.2431146, 0.1334844])
        self.loc_dist  = np.array([-0.0006735, -0.0023663, -0.0143416])
        self.scale_dist  = np.array([0.0037602, 0.0067417, 0.0260779])

        self.sigma_dist  = np.log(self.scale_dist)

        self.noise_ch_min = np.array([0.001094, 0.001094, 0.001094])
        self.noise_ch_max = np.array([0.013, 0.018, 0.061])


    def __call__(self, image):

        # Get image shape (assumes square img)
        im_dim = image.shape[1]
        # draw 'true' noise level of each channel from lognormal fits
        self.sigma_true = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        if self.uniform:
            # draw desired augmented noise level from uniform, to target tails more
            self.sigma_final = np.random.uniform(self.noise_ch_min, self.noise_ch_max)
        else:
            self.sigma_final = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        # Gaussian noise adds as c^2 = a^2 + b^2
        self.sigma_augment = self.sigma_final**2 - self.sigma_true**2
        self.sigma_augment[self.sigma_augment < 0.] = 0.
        self.sigma_augment = np.sqrt(self.sigma_augment)

        for i in range(self.im_ch):
            if self.sigma_augment[i] > 0.:
                image[:,:,i] += np.random.normal(self.mean, self.sigma_augment[i], size = (im_dim, im_dim))

        return image.astype(np.float32)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "GaussianNoiseDecals":
        """
        Instantiates GaussianNoiseDecals from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            GaussianNoiseDecals instance.
        """
        mean = config['mean'] # default: 0
        im_ch = config['im_ch'] # default: 3
        uniform = config['uniform'] # default: False
        return cls(mean=mean, im_ch=im_ch , uniform=uniform)
