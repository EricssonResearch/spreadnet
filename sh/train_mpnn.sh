#! /bin/bash -l
#
#SBATCH -p core -n 2 -G 1 -t 2-00:00:00
#SBATCH -A uppmax2022-2-23 -M snowy

# To use WANDB, uncomment the next line and add your key
# export WANDB_API_KEY=

cd ../experiments
singularity exec --nv /proj/uppmax2022-2-23/spreadnet-runner-py3-pyg-cu116.sif python train.py --model="MPNN"
# singularity exec --nv /proj/uppmax2022-2-23/spreadnet-runner-py3-pyg-cu116.sif python train.py --model="MPNN" --wandb
