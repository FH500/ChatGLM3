DEV_PATH=excel_data/dev_60.json
CHECKPOINTPATH=output/EXCEL_pt-2000-5e-3-128

python finetune_chatmodel_demo/evaluate.py \
    --pt-checkpoint $CHECKPOINTPATH \
    --model chatglm3-6b \
    --dev-path $DEV_PATH \
    --pt-pre-seq-len 128 \
    --max-new-tokens 512