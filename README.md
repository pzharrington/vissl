# VISSL + DeCaLS
### Self-supervised learning playground for sky surveys
---

This fork of [facebook research's vissl library](https://vissl.readthedocs.io/en/v0.1.5/) contains customizations to apply self-supervised learning computer vision techinques to sky survey data from the [Dark Energy Camera Legacy Survey](https://www.legacysurvey.org/decamls/), and evaluation code for downstream tasks like photo-z estimation.


### Installing+running at NERSC
To run in the pytorch ngc container using shifter, add the following flags to any `sbatch` scripts or `salloc` commands:
```
--image=nersc/pytorch:ngc-20.10-v0 --volume="/dev/infiniband:/sys/class/infiniband_verbs"
```

Install from source for now to incorporate more recent `vissl` changes (not sure how frequently the pypi package will be updated). To do this from the container with the appropriate path set for `$PYTHONUSERBASE` (to install base packages), `salloc` with the above flags and then do
```
srun --pty --nodes=1 --ntasks=1 shifter --env HDF5_USE_FILE_LOCKING=FALSE --env PYTHONUSERBASE=[your home directory here]/.local/cori/pytorch_ngc_20.10-v0 bash
```
to setup an interactive session for the install. Then,
```
cd $HOME && git clone --recursive https://github.com/pzharrington/vissl.git && cd $HOME/vissl/
pip install --user --progress-bar off -r requirements.txt
pip install --user opencv-python
pip uninstall -y classy_vision
pip install --user classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/master
pip install --user -e .[dev]
python -c 'import vissl, apex, cv2'
```
Also install decals-specfic modules (`astropy`, `skimage`, etc, ... I forget if there's anything else), then should be good to go! 

The `vissl/tools/run_distributed_engines.py` file will be your launcher for trainings/evals/etc. Call this from your `sbatch` or launch script, along with some config, to run a training or eval, e.g.
```
python3 run_distributed_engines.py config=quick_1gpu_resnet50_simclr
```

Sanity edit: locate your local installation of `fvcore` at `$HOME/.local/cori/pytorch_ngc_20.10-v0`, find the file `common/file_io.py`, and comment out the following lines: 
```
logger.warning(
         "** fvcore version of PathManager will be deprecated soon. **\n"
         "** Please migrate to the version in iopath repo. **\n"
         "https://github.com/facebookresearch/iopath \n"

)
```
This will save you tons of chatter until upstream `vissl` fixes this.

### Current issues, gotchas, etc
Threading issue I think from multithreaded data loading w/ numpy arrays (decals data augmentations) -- if you get `OpenBLAS` thread errors, set env var `--env OPENBLAS_NUM_THREADS=1` in your shifter command.

