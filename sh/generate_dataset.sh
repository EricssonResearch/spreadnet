#! /bin/bash -l
#
#SBATCH -p core -n 12 -t 1-00:00:00
#SBATCH -A uppmax2022-2-23 -M snowy

cd ../experiments
singularity exec --nv /proj/uppmax2022-2-23/spreadnet-runner-py3-pyg-cu116.sif python generate_dataset.py
