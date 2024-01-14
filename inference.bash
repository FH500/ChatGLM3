CHECKPOINTPATH=output/EXCEL_pt-500-5e-3-128

python finetune_chatmodel_demo/inference.py \
    --pt-checkpoint $CHECKPOINTPATH \
    --model chatglm3-6b \
    --pt-pre-seq-len 128 \
    --max-new-tokens 512