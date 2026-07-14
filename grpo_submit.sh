#!/bin/bash
# 
# CompecTA (c) 2018
# 
# You should only work under the /scratch/users/<username> directory.
#
# Jupyter job submission script
#
# TODO:
#   - Set name of the job below changing "JupiterNotebook" value.
#   - Set the up_example.shequested number of nodes (servers) with --nodes parameter.
#   - Set the requested number of tasks (cpu cores) with --ntasks parameter. (Total accross all nodes)
#   - Select the partition (queue) you want to run the job in:
#     - short : For jobs that have pessimismimum run time of 120 mins. Has higher priority.
#     - mid   : For jobs that have pessimismimum run time of 1 day..
#     - long  : For jobs that have pessimismimum run time of 7 days. Lower priority than short.
#     - longer: For testing purposes, queue has 31 days limit but only 3 nodes.
#   - Set the required time limit for the job with --time parameter.
#     - Acceptable time formats include "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", "days-hours:minutes" and "days-hours:minutes:seconds"
#   - Put this script and all the input file under the same directory.
#   - Set the required parameters, input/output file names below.
#   - If you do not want mail please remove the line that has --mail-type and --mail-user. If you do want to get notification emails, set your email address.
#   - Put this script and all the input file under the same directory.
#   - Submit this file using:
#      sbatch jupyter_submit.sh
#
# -= Resources =-
#

#SBATCH --job-name=test
#SBATCH --cpus-per-task=16
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --mem=150g
#SBATCH --gres=gpu:ampere_a40:4
#SBATCH --time=14-00:00:00
#SBATCH --output=ais-reproductions/%J.log
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ocagatan19@ku.edu.tr

# Please read before you run: http://login.kuacc.ku.edu.tr/#h.3qapvarv2g49

################################################################################
##################### !!! DO NOT EDIT BELOW THIS LINE !!! ######################
################################################################################
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
    echo "GPUs occupied, requeuing..."
    scontrol requeue $SLURM_JOB_ID
    exit 0
fi

source ~/.bashrc 
module load gnu14
module load cuda/12.9.1
conda activate pipeline-rl

python -m pipelinerl.launch --config-name=reward_hacking output_dir="results/reward_hacking-${SLURM_JOB_ID}"
