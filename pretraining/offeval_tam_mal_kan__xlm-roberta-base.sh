#!/bin/bash

mkdir offeval
echo "----------"
echo "CAUTION: check output dir"
echo "----------"
echo " "

mkdir offeval/tamil
CUDA_VISIBLE_DEVICES=0 python run_bert_mlm.py \
    --overwrite_output_dir --logging_steps 500 --save_steps 5000 --save_total_limit 1 --num_train_epochs 3.0 \
    --output_dir="./offeval/tamil/xlm-roberta-base" \
    --model_name_or_path=xlm-roberta-base \
    --do_train --train_data_file="../offeval/tamil/pretraining_data/train.txt" \
    --line_by_line --mlm \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --do_eval --eval_data_file="../offeval/tamil/pretraining_data/test.txt"

mkdir offeval/malayalam
CUDA_VISIBLE_DEVICES=0 python run_bert_mlm.py \
    --overwrite_output_dir --logging_steps 500 --save_steps 5000 --save_total_limit 1 --num_train_epochs 3.0 \
    --output_dir="./offeval/malayalam/xlm-roberta-base" \
    --model_name_or_path=xlm-roberta-base \
    --do_train --train_data_file="../offeval/malayalam/pretraining_data/train.txt" \
    --line_by_line --mlm \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --do_eval --eval_data_file="../offeval/malayalam/pretraining_data/test.txt"

mkdir offeval/kannada
CUDA_VISIBLE_DEVICES=0 python run_bert_mlm.py \
    --overwrite_output_dir --logging_steps 500 --save_steps 5000 --save_total_limit 1 --num_train_epochs 3.0 \
    --output_dir="./offeval/kannada/xlm-roberta-base" \
    --model_name_or_path=xlm-roberta-base \
    --do_train --train_data_file="../offeval/kannada/pretraining_data/train.txt" \
    --line_by_line --mlm \
    --per_device_train_batch_size=8 --per_device_eval_batch_size=8 \
    --do_eval --eval_data_file="../offeval/kannada/pretraining_data/test.txt"