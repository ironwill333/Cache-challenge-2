#!/bin/bash
#SBATCH -J train_molecules
#SBATCH -c 1
#SBATCH --partition=dept_cpu
#SBATCH -o train-%A_%a.out
#SBATCH -e train-%A_%a.err
#SBATCH --mail-type=fail
#SBATCH --mail-user=aguruswamy@vassar.edu
#SBATCH --array=1-5


module load anaconda
source activate sklearn-env
cd $SLURM_SUBMIT_DIR

cmd=`sed -n "${SLURM_ARRAY_TASK_ID}p" model_cmds.txt`
eval $cmd

exit