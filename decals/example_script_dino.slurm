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
     config=pretrain/dino/decals_16gpu_deits8.yaml \
     config.CHECKPOINT.DIR="/pscratch/sd/p/pharring/vissl/dino/base" \
     config.CHECKPOINT.AUTO_RESUME=true \
     config.DISTRIBUTED.RUN_ID="localhost:40050" \
     config.SEED_VALUE=$RANDOM \
     config.SLURM.USE_SLURM=true \
     config.SLURM.NAME="${SLURM_JOB_NAME}" \ 
     config.SLURM.PARTITION="gpu" \
     config.SLURM.LOG_FOLDER=/global/homes/p/pharring/vissl/decals/sout/ \
     config.SLURM.TIME_HOURS=4 \
     config.SLURM.CONSTRAINT="gpu" \
     config.SLURM.MEM_GB=250 \
     config.SLURM.PORT_ID=40050 \
     config.SLURM.NUM_CPU_PER_PROC=${SLURM_CPUS_PER_TASK}
'

date

