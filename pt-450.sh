#! /usr/bin/env bash

set -ex

PRE_SEQ_LEN=128
LR=5e-3
NUM_GPUS=1
MAX_SOURCE_LEN=128
MAX_TARGET_LEN=512
DEV_BATCH_SIZE=16
GRAD_ACCUMULARION_STEPS=1
MAX_STEP=500
SAVE_INTERVAL=500

DATESTR=`date +%Y%m%d-%H%M%S`
RUN_NAME=EXCEL_pt

BASE_MODEL_PATH=/home/featurize/work/yzh/ChatGLM3/chatglm3-6b
DATASET_PATH=/home/featurize/work/yzh/ChatGLM3/pt_train/formatted_data/450_data.jsonl
OUTPUT_DIR=output/${RUN_NAME}-${MAX_STEP}-${LR}-${PRE_SEQ_LEN}

mkdir -p $OUTPUT_DIR

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS finetune_chatmodel_demo/finetune.py \
    --train_format input-output \
    --train_file $DATASET_PATH \
    --preprocessing_num_workers 1 \
    --model_name_or_path $BASE_MODEL_PATH \
    --output_dir $OUTPUT_DIR \
    --max_source_length $MAX_SOURCE_LEN \
    --max_target_length $MAX_TARGET_LEN \
    --per_device_train_batch_size $DEV_BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUMULARION_STEPS \
    --max_steps $MAX_STEP \
    --logging_steps 10 \
    --save_steps $SAVE_INTERVAL \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN 2>&1 | tee ${OUTPUT_DIR}/train.log
