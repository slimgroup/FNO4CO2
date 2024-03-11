#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --job-name Parametric_w_fourier_2d_vc
#SBATCH --output=parametric_vc
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=57G
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1

module load Julia/1.8/5 cudnn-11 nvhpc-mpi Miniconda/3

export JULIA_NUM_THREADS=$SLURM_CPUS_ON_NODE

export DEVITO_LANGUAGE=openacc
export DEVITO_ARCH=nvc
export DEVITO_PLATFORM=nvidiaX

export DFNO_3D_GPU=1
export FNO4CO2GPU=1

srun julia scripts/fourier_cig.jl

exit 0
EOT
