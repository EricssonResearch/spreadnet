#! /bin/bash -l
#
#SBATCH -p core -n 1 -t 1-00:00:00
#SBATCH -A uppmax2022-2-23 -M snowy

cd ../
singularity exec --nv /proj/uppmax2022-2-23/spreadnet-runner-py3-pyg-cu116.sif python generate_dataset.py
