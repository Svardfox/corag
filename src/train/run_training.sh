#!/bin/bash
set -e
set -x

# 1. Prepare Data
# Note: Ensure the Graph API server is running before executing this!
echo "Running data preparation..."
# python3 src/train/prepare_training_data.py --output_file data/train_with_graph_retrieval.jsonl --retrieve --top_k 3

# 2. Run Training
# Use dry_run first
echo "Running training..."
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --nproc_per_node 8 src/train/train.py \
    --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
    --train_file data/train_with_graph_retrieval.jsonl \
    --output_dir output/corag_qwen_finetuned \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --deepspeed src/train/ds_config_zero3.json \
    --learning_rate 2e-5 \
    --max_len 4096 \
    --do_train \
    --save_steps 500 \
    --logging_steps 10 \
    --report_to none \
    --bf16 True \
    --overwrite_output_dir
