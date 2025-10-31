#!/bin/bash

PROJECT_ROOT=$(python -c "from utils.path_utils import get_project_root; print(get_project_root())")

NPROC_PER_NODE=4 \
CUDA_VISIBLE_DEVICES=1,2,3,4 \
swift sft \
    --model Qwen/Qwen3-32B \
    --train_type lora \
    --dataset "$PROJECT_ROOT/data/train_polarity_ratio121.json" \ 
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --lora_rank 8 \
    --lora_alpha 16 \
    --target_modules all-linear \
    --gradient_accumulation_steps 2 \
    --eval_steps 100 \
    --save_steps 200 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir "$PROJECT_ROOT/classifiers/output/pol_ratio121/" \  # Replace with path to save trained model
    --warmup_ratio 0.05 \
    --lora_dropout 0.05 \
    --deepspeed zero3 \
    --report_to wandb \
    --dataloader_num_workers 4 $@ 2>&1 | tee "$PROJECT_ROOT/classifiers/output/pol_ratio121/pol_ratio121.log"
