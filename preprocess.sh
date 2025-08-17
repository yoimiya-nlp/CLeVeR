#!/bin/bash
#
# ljy@20250707
#
python data_preprocess.py \
    --pretrain_text_model_name=pretrain_text_model \
    --pretrain_code_model_name=pretrain_code_model \
    --program_language=c \
    --dataset=dataset/dataset.jsonl \
    --dataset_name=dataset \
    --seed 123456
