


model_root=../released_models
data_path=/lustre/scratch/client/vinai/users/quanpn2/vtab-1k/cifar100/
output_dir=../cifar-tuning-10tokens
        
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
    SOLVER.BASE_LR "3.0" \
    SOLVER.WEIGHT_DECAY "0.001" \
    SEED 42 \
    MODEL.MODEL_ROOT "${model_root}" \
    DATA.DATAPATH "${data_path}" \
    OUTPUT_DIR "${output_dir}" 
