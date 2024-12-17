
model_root=../released_models
data_path=/lustre/scratch/client/vinai/users/quanpn2/vtab-1k/dmlab/
output_dir=../outputs
        
export SLURMD_NODENAME=localhost
export CUDA_VISIBLE_DEVICES=0

python train.py \
    --config-file configs/prompt/cub.yaml \
    MODEL.TYPE "vit" \
    DATA.BATCH_SIZE "128" \
    MODEL.PROMPT.NUM_TOKENS "100" \
    MODEL.PROMPT.DEEP "True" \
    MODEL.PROMPT.DROPOUT "0.1" \
    DATA.FEATURE "sup_vitb16_imagenet21k" \
    DATA.NAME "vtab-dmlab" \
    DATA.NUMBER_CLASSES "6" \
    SOLVER.BASE_LR "0.5" \
    SOLVER.WEIGHT_DECAY "0.001" \
    SEED 42 \
    MODEL.MODEL_ROOT "${model_root}" \
    DATA.DATAPATH "${data_path}" \
    OUTPUT_DIR "${output_dir}" 
