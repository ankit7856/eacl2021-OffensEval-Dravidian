#!/bin/bash

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --max-epochs 10 --mode train_dev_test --model-name bert-charlstm-lstm --text-type "trt" --dataset-name eacl2021/offeval/kannada
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --max-epochs 10 --mode train_dev_test --model-name bert-charlstm-lstm --text-type "trt" --dataset-name eacl2021/offeval/tamil
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --max-epochs 10 --mode train_dev_test --model-name bert-charlstm-lstm --text-type "trt" --dataset-name eacl2021/offeval/malayalam

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --max-epochs 10 --mode train_dev_test --model-name bert-charlstm-lstm --text-type "" --dataset-name eacl2021/offeval/kannada
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --max-epochs 10 --mode train_dev_test --model-name bert-charlstm-lstm --text-type "" --dataset-name eacl2021/offeval/tamil
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --max-epochs 10 --mode train_dev_test --model-name bert-charlstm-lstm --text-type "" --dataset-name eacl2021/offeval/malayalam

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/combined
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode dev_kannada --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/combined --eval-ckpt-path ../checkpoints/eacl2021/offeval/combined/xlm-roberta-base/text_raw/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode dev_tamil --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/combined --eval-ckpt-path ../checkpoints/eacl2021/offeval/combined/xlm-roberta-base/text_raw/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode dev_malayalam --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/combined --eval-ckpt-path ../checkpoints/eacl2021/offeval/combined/xlm-roberta-base/text_raw/

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
##CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-charlstm-lstm --text-type "" --dataset-name eacl2021/offeval/kannada --max-epochs 10 --custom-pretrained-path ../../pretraining/offeval/kannada/xlm-roberta-base/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-fasttext-lstm --text-type "trt" --dataset-name eacl2021/offeval/kannada --max-epochs 10 --custom-pretrained-path ../../pretraining/offeval/kannada/xlm-roberta-base/
#rm -r ../checkpoints/eacl2021
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-charlstm-lstm --text-type "" --dataset-name eacl2021/offeval/tamil --max-epochs 10 --custom-pretrained-path ../../pretraining/offeval/tamil/xlm-roberta-base/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-fasttext-lstm --text-type "trt" --dataset-name eacl2021/offeval/tamil --max-epochs 10 --custom-pretrained-path ../../pretraining/offeval/tamil/xlm-roberta-base/
#rm -r ../checkpoints/eacl2021
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-charlstm-lstm --text-type "" --dataset-name eacl2021/offeval/malayalam --max-epochs 10 --custom-pretrained-path ../../pretraining/offeval/malayalam/xlm-roberta-base/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-fasttext-lstm --text-type "trt" --dataset-name eacl2021/offeval/malayalam --max-epochs 10 --custom-pretrained-path ../../pretraining/offeval/malayalam/xlm-roberta-base/

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/kannada --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/tamil  --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/malayalam  --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-base-cased --text-type "trt" --dataset-name eacl2021/offeval/kannada
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-base-cased --text-type "trt" --dataset-name eacl2021/offeval/tamil
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-base-cased --text-type "trt" --dataset-name eacl2021/offeval/malayalam

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "trt" --dataset-name eacl2021/offeval/kannada
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "trt" --dataset-name eacl2021/offeval/tamil
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "trt" --dataset-name eacl2021/offeval/malayalam

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/kannada --custom-pretrained-path ../../pretraining/offeval/kannada/xlm-roberta-base/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/tamil  --custom-pretrained-path ../../pretraining/offeval/tamil/xlm-roberta-base/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/malayalam  --custom-pretrained-path ../../pretraining/offeval/malayalam/xlm-roberta-base/

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-base-multilingual-cased --text-type "" --dataset-name eacl2021/offeval/kannada --custom-pretrained-path ../../pretraining/offeval/kannada/bert-base-multilingual-cased/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-base-multilingual-cased --text-type "" --dataset-name eacl2021/offeval/tamil  --custom-pretrained-path ../../pretraining/offeval/tamil/bert-base-multilingual-cased/
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-base-multilingual-cased --text-type "" --dataset-name eacl2021/offeval/malayalam  --custom-pretrained-path ../../pretraining/offeval/malayalam/bert-base-multilingual-cased/

#rm -r ../../checkpoints/eacl2021
#cd pretraining
#bash offeval_tam_mal_kan__bert-base-multilingual-cased.sh

#rm -r ../../checkpoints/eacl2021
#cd ../../scripts
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/kannada
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/tamil
#CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/malayalam
