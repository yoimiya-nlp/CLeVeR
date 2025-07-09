#!/bin/bash
#
# ljy@20250707
#
python linear_probe.py \
    --output_dir=saved_models \
    --pretrain_text_model_name=pretrain_text_model \
    --pretrain_code_model_name=pretrain_code_model \
    --do_test \
    --dataset=vcltestdata \
    --pretrain_checkpoint=pretrain_vul_model \
    --from_checkpoint=probe_vcltestdata_cls \
    --to_checkpoint=probe_vcltestdata_cls \
    --epoch 30 \
    --code_length 512 \
    --hidden_size 768 \
    --train_batch_size 64 \
    --eval_batch_size 64 \
    --learning_rate 2e-4 \
    --max_grad_norm 1.0 \
    --evaluate_during_training \
    --seed 123456 2>&1| tee saved_models/train.log