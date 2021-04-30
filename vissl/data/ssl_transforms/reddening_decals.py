import numpy as np
from classy_vision.dataset.transforms import register_transform
from classy_vision.dataset.transforms.classy_transform import ClassyTransform
from typing import Any, Dict


@register_transform("ReddeningDecals")
class ReddeningDecals(ClassyTransform):
    '''Redden image with random sampling of ebv.
    
    A lognormal fit matches the measured E(B-V) distribution better than Gaussian. Fit with scipy,
    which has a different paramaterization of the log normal than numpy.random 
    (see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html):
    
    shape, loc, scale -> np.random.lognormal(np.log(scale), shape, size=1) + loc

    # (min, max)=(0.00, 1.0), Lognormal fit (shape, loc, scale)=(0.67306, 0.001146, 0.03338)
    '''

    def __init__(self, uniform, filters_use, ebv_max):
        self.uniform = uniform

        self.filters_use = filters_use

        # Log normal fit paramaters
        self.shape_dist = 0.67306
        self.loc_dist = 0.001146
        self.scale_dist = 0.03338

        self.sigma_dist  = np.log(self.scale_dist)

        self.ebv_min = 0.00
        self.ebv_max = ebv_max

    def ebv_to_transmission(self, ebv):
        # ebv to transmission is just a simple power law I fit for each band - works perfectly
        # (I couldnt figure it out from reference paper https://ui.adsabs.harvard.edu/abs/1998ApJ...500..525S/abstract)

        # ebv = Galactic extinction E(B-V) reddening from SFD98, used to compute MW_TRANSMISSION
        # transmission = Galactic transmission in g filter in linear units [0,1]
        filters   = ['g', 'r', 'z', 'W1', 'W2']
        exponents = np.array([-1.2856, -0.866, -0.4844, -0.0736, -0.04520])

        nfilters     = len(filters)
        nfilters_use = len(self.filters_use)

        exponents = {filters[i]: exponents[i] for i in range(nfilters)}
        exponents_use  = np.array([exponents[fi] for fi in self.filters_use])

        transmission_ebv = 10**(ebv*exponents_use)

        return transmission_ebv

    def __call__(self, image):
        # Random ebv sampled from lognormal
        if self.uniform:
            new_ebv = np.random.uniform(0, self.ebv_max)
        else:
            new_ebv = np.random.lognormal(self.sigma_dist, self.shape_dist) + self.loc_dist

        new_transmission = self.ebv_to_transmission(new_ebv)
        return np.float32(image * new_transmission)

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "ReddeningDecals":
        """
        Instantiates ReddeningDecals from configuration.

        Args:
            config (Dict): arguments for for the transform

        Returns:
            ReddeningDecals instance.
        """
        uniform=config['uniform'] # default: False
        filters_use=config['filters_use'] # default: ['g', 'r', 'z']
        ebv_max=config['ebv_max'] # default: 1.0
        return cls(uniform=uniform, filters_use=filters_use, ebv_max=ebv_max)
