#!/bin/bash -e

#SBATCH --job-name=cifarvpt # create a short name for your job
#SBATCH --output=/lustre/scratch/client/vinai/users/quanpn2/Bayesian_finetuning/sbatch_results/mbpp%A.out # create a output file
#SBATCH --error=/lustre/scratch/client/vinai/users/quanpn2/Bayesian_finetuning/sbatch_results/mbpp%A.err # create a error file
#SBATCH --partition=research # choose partition
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-gpu=40GB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=10-00:00          # total run time limit (DD-HH:MM)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail          # send email when job fails
#SBATCH --mail-user=v.quanpn2@vinai.io



module purge
module load python/miniconda3/miniconda3
eval "$(conda shell.bash hook)"



conda activate very_angry









model_root=../released_models
data_path=/lustre/scratch/client/vinai/users/quanpn2/vtab-1k/cifar100/
output_dir=../cifar100
        
export SLURMD_NODENAME=localhost
export CUDA_VISIBLE_DEVICES=0

CUDA_VISIBLE_DEVICES=0 python train.py \
    --config-file configs/prompt/cub.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.PROMPT.NUM_TOKENS "10" \
    MODEL.PROMPT.DEEP "True" \
    MODEL.PROMPT.DROPOUT "0.1" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    DATA.NAME "vtab-cifar100" \
    DATA.NUMBER_CLASSES "100" \
    SOLVER.BASE_LR "2.0" \
    SOLVER.WEIGHT_DECAY "0.001" \
    SEED 42 \
    MODEL.MODEL_ROOT "${model_root}" \
    DATA.DATAPATH "${data_path}" \
    OUTPUT_DIR "${output_dir}" 
