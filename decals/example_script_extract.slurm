#!/bin/bash -l
#SBATCH --nodes=4
#SBATCH --time=04:00:00  
#SBATCH -C gpu
#SBATCH -J vissl
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=128

#SBATCH -o /global/homes/p/pharring/vissl/decals/sout/dino.out

date
#echo "Extract ImageNet to /tmp time:"
#time ./ImageNetcpy.sh

module load pytorch/1.8.0
export HDF5_USE_FILE_LOCKING=FALSE

srun -u bash -c '

python3 /global/homes/p/pharring/vissl/tools/run_distributed_engines.py \
     config=feature_extraction/extract_ViTS16_decals_8gpu.yaml \
     config=feature_extraction/trunk_only/=ViT-S16.yaml \
     config.CHECKPOINT.DIR="/grand/skysurvey_adsp/vissl/dino/feats/patch8_ep0" \
     config.WEIGHTS_INIT.PARAMS_FILE="/grand/skysurvey_adsp/vissl/dino/ckpts/patch8_ep0.torch
'

date

