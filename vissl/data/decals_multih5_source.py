from torch.utils.data import Dataset 
import h5py
import numpy as np
import skimage.transform
import skimage.filters
import logging
import os

class DecalsMultiHDF5Dataset(Dataset):
    def __init__(self, cfg, data_source, path, split, dataset_name):
        """
        Read from hdf5 file on disk, which holds images, targets, and auxiliary fields as 
        separate dataset keys under the root hdf5 group (first index for each indexes unique samples)
        """
        super(DecalsMultiHDF5Dataset, self).__init__()
        assert data_source in [
            "disk_filelist",
            "disk_folder",
            "decals_hdf5",
            "decals_multihdf5"
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
        self.format = cfg["DATA"]["DECALS"]["FORMAT"]
        self._check_consistent_filters()

        self._num_samples = cfg["DATA"]["DECALS"]["MULTI"]["NTOT"]
        self.chunk = cfg["DATA"]["DECALS"]["MULTI"]["CHUNK"]
        self.file_list = []
        self.opened = []
        for n in sorted(os.listdir(cfg["DATA"]["DECALS"]["MULTI"]["DIR"])):
            if '.h5' in n or '.hdf5' in n: 
                self.file_list.append(os.path.join(cfg["DATA"]["DECALS"]["MULTI"]["DIR"], n))
                self.opened.append(False)
        logging.info('DecalsMultiHDF5Dataset using files from %s'%cfg["DATA"]["DECALS"]["MULTI"]["DIR"])

    def num_samples(self):
        """
        Size of the dataset
        """
        return self._num_samples

    def __len__(self):
        return self.num_samples()

    def __getitem__(self, idx):


        file_idx = idx//self.chunk
        off_idx = idx%self.chunk
        
        if not self.opened[file_idx]:
            self.opened[file_idx] = h5py.File(self.file_list[file_idx], 'r')
        hfile = self.opened[file_idx]
        lim = len(hfile[self.imkey])
        assert off_idx < lim, "offset %d for index %d is bigger than length %d of file %d: %s"%(off_idx, idx, lim, file_idx, self.file_list[file_idx])
        if self.format == 'HWC':
            im = np.swapaxes(hfile[self.imkey][off_idx], 0, 2)
        else:
            im = hfile[self.imkey][off_idx]
        if self.dered:
            ebv = hfile['ebv'][off_idx]
            im = self.deredden(im, ebv, self.format)
        return im.astype(np.float32), True

    def _check_consistent_filters(self):
        tforms = self.cfg["DATA"][self.split]['TRANSFORMS']
        for t in tforms:
           if 'filters_use' in t.keys():
               assert t['filters_use'] == self.filters_use, 'Filters used in transform %s must match DATA.FILTERS_USE'%t['name']

    def deredden(self, raw_image, ebv_i, fmt):
        # De-redden image given SFD98 ebv value
        transmission_i = self.ebv_to_transmission(ebv_i)
        if fmt == 'CHW':
            # So numpy broadcasting doesn't complain
            return raw_image/np.expand_dims(transmission_i, (1,2))
        else:
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

