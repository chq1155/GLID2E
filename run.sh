#!/bin/bash

#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH -a 0-6
#SBATCH --output=myjob.%A_%a.out
#SBATCH --error=myjob.%A_%a.err

cd /research/d1/gds/hsshi23/project/protein_try/glid2e_protein/fmif

conda activate multiflow

bash run$SLURM_ARRAY_TASK_ID.sh
