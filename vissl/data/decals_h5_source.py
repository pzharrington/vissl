from torch.utils.data import Dataset 
import h5py
import numpy as np
import skimage.transform
import skimage.filters
import logging

def load_specz_labels(path, cfg):
    key = cfg["DATA"]["DECALS"]["H5KEY_LAB"]
    with h5py.File(path, 'r') as f:
        z = f[key][...].astype(np.float32)
    zlo, zhi = cfg["DATA"]["DECALS"]["SPECZ_RANGE"]
    Nbins = cfg["DATA"]["DECALS"]["SPECZ_NBINS"]
    zbins = np.linspace(zlo, zhi, num=Nbins+1, endpoint=True).astype(np.float32)
    labels = np.digitize(z, bins=zbins, right=True) - 1
    return labels

class DecalsHDF5Dataset(Dataset):
    def __init__(self, cfg, data_source, path, split, dataset_name):
        """
        Read from hdf5 file on disk, which holds images, targets, and auxiliary fields as 
        separate dataset keys under the root hdf5 group (first index for each indexes unique samples)
        """
        super(DecalsHDF5Dataset, self).__init__()
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "decals_hdf5"
        ], "data_source must be either disk_filelist or disk_folder or decals_hdf5"
        self.cfg = cfg
        self.split = split
        self.dataset_name = dataset_name
        self.data_source = data_source
        self._path = path


        self.imkey = cfg["DATA"]["DECALS"]["H5KEY_IMG"]
        self.labkey = cfg["DATA"]["DECALS"]["H5KEY_LAB"]
        self.dered = cfg["DATA"]["DECALS"]["DEREDDENED"]
        self.filters_use = cfg["DATA"]["DECALS"]["FILTERS_USE"]
        self._check_consistent_filters()

        with h5py.File(self._path, 'r') as hf:
            self._num_samples = hf[self.imkey].shape[0]

    def _open_file(self):
        self.hfile = h5py.File(self._path, 'r')

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):
        if not hasattr(self, 'hfile'):
            self._open_file()
        # SDSS data is in NCHW format, unlike HSC - which is in NHWC format.
        im = np.swapaxes(self.hfile[self.imkey][idx], 0, 2) # im is WHC
        if self.dered:
            ebv = self.hfile['ebv'][idx]
            im = self.deredden(im, ebv)
        return im.astype(np.float32), True

    def _check_consistent_filters(self):
        tforms = self.cfg["DATA"][self.split]['TRANSFORMS']
        for t in tforms:
           if 'filters_use' in t.keys():
               assert t['filters_use'] == self.filters_use, 'Filters used in transform %s must match DATA.FILTERS_USE'%t['name']

    def deredden(self, raw_image, ebv_i):
        # De-redden image given SFD98 ebv value
        transmission_i = self.ebv_to_transmission(ebv_i)
        return raw_image/transmission_i

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

