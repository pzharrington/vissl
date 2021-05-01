#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=00:30:00  
#SBATCH -C gpu
#SBATCH --account=m1759

#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=80

#SBATCH --image=nersc/pytorch:ngc-20.10-v0
#SBATCH --volume="/dev/infiniband:/sys/class/infiniband_verbs"
#SBATCH -o /path/to/slumr/output/dir/train.out

date
srun --nodes=1 --ntasks=1 shifter --env HDF5_USE_FILE_LOCKING=FALSE --env PYTHONUSERBASE=$HOME/.local/cori/pytorch_ngc_20.10-v0 --env OPENBLAS_NUM_THREADS=1 <<EOF

python3 /global/homes/p/pharring/vissl/tools/run_distributed_engines.py \
     config=benchmark/photoz/fullsup_resnet_specz \
     config.CHECKPOINT.DIR="/path/to/checkptdir/" \
     config.CHECKPOINT.AUTO_RESUME=true \
     config.DISTRIBUTED.NUM_PROC_PER_NODE=8 \
     config.SEED_VALUE=$RANDOM \
     config.DATA.TRAIN.DATA_LIMIT=-1 \
     config.DATA.TEST.DATA_LIMIT=-1

EOF
date

