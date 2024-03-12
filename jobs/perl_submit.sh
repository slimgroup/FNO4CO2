#!/bin/bash
sbatch <<EOT
#!/bin/bash

#SBATCH --nodes=1
#SBATCH --constraint=gpu
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --gpu-bind=none
#SBATCH --qos=regular
#SBATCH --job-name Parametric_w_fourier_2d_vc
#SBATCH --mail-user=richardr2926@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --time=15:00:00
#SBATCH --account=m3863_g

nvidia-smi
export SLURM_CPU_BIND="cores"
export PATH=$PATH:$HOME/.julia/bin
export DFNO_3D_GPU=1
export FNO4CO2GPU=1
# export LD_LIBRARY_PATH=
export LD_PRELOAD=/opt/cray/pe/lib64/libmpi_gtl_cuda.so.0
module load cudnn/8.9.3_cuda12 julia/1.8

srun julia scripts/fourier_cig.jl

exit 0
EOT
