#!/bin/bash
#
# ljy@20250707
#
python clever.py \
    --output_dir=saved_models \
    --pretrain_text_model_name=pretrain_text_model \
    --pretrain_code_model_name=pretrain_code_model \
    --do_train \
    --do_test \
    --dataset=vcldata \
    --from_checkpoint=pretrain_vul_model \
    --to_checkpoint=pretrain_vul_model \
    --epoch 30 \
    --code_length 512 \
    --hidden_size 768 \
    --train_batch_size 32 \
    --eval_batch_size 64 \
    --learning_rate 2e-5 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log
