This folder contains scripts to create datasets, process them and to run code-mixed experiments. Please move to ```./scripts``` folder by doing ```cd ./scripts``` before running following commands

# classification models

#### `interactive`
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --eval-ckpt-path "../checkpoints/arxiv-sail2017/Hinglish/baseline/xlm-roberta-base/text_raw"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --eval-ckpt-path "../checkpoints/arxiv-sail2017/Hinglish/data_aug with MLM pretraining/xlm-roberta-base/text_raw"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode interactive --model-name xlm-roberta-base --text-type "" --eval-ckpt-path "../checkpoints/arxiv-sail2017/Hinglish/data_aug with MLM pretraining/xlm-roberta-base/text_raw"
```

#### baselines
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name sentimix2020/Hinglish
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name sail2017/Hinglish
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name kumaretal_2019_agg/Hinglish-R
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name kumaretal_2020_agg/Hinglish-R  --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name vijayetal_2018_hatespeech/Hinglish  --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name kauretal_2019_reviews/Hinglish
```
```
# follow ../checkpoints/pretrained/download_huggingface_models.py script to download necessary model(s)
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name bert-base-cased --text-type "" --custom-pretrained-path ../checkpoints/pretrained/murali1996/bert-base-cased-spell-correction
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name bert-base-multilingual-cased --text-type "" --custom-pretrained-path ../checkpoints/pretrained/sagorsarker/codeswitch-hineng-pos-lince
```

#### baselines (extended)
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "D"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "non_english"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "non_hindi"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "non_english_D"
```

#### sentence-bert models
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --sentence-bert --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name sentimix2020/Hinglish
CUDA_VISIBLE_DEVICES=0 python run_classification.py --sentence-bert --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name sail2017/Hinglish
CUDA_VISIBLE_DEVICES=0 python run_classification.py --sentence-bert --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name kumaretal_2019_agg/Hinglish-R
CUDA_VISIBLE_DEVICES=0 python run_classification.py --sentence-bert --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name kumaretal_2020_agg/Hinglish-R  --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --sentence-bert --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name vijayetal_2018_hatespeech/Hinglish  --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --sentence-bert --mode train_test --model-name xlm-roberta-base --text-type "" --dataset-name kauretal_2019_reviews/Hinglish
```

#### MLM pretrained bert models
- see ```./run_bert_mlm/download_checkpoints.py``` for pretrained models
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --custom-pretrained-path ./run_bert_mlm/on_raw__pretrained_cs_mlm_models/xlm-roberta-base
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --custom-pretrained-path ./run_bert_mlm/on_raw_new_noisy__pretrained_cs_mlm_models/xlm-roberta-base
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --custom-pretrained-path ./run_bert_mlm/on_raw_new_hintrtaug__pretrained_cs_mlm_models/xlm-roberta-base
```
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name xlm-roberta-base --text-type "D" --custom-pretrained-path ./run_bert_mlm/on_raw_new_D__pretrained_cs_mlm_models/xlm-roberta-base
```

### MLM pretrained and Semantic Similarity Supervised Task Training
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name custom/semantic_similarity --model-name bert-semantic-similarity --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base --max-epochs 3
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode test --dataset-name custom/semantic_similarity --model-name bert-semantic-similarity --eval-ckpt-path ../checkpoints/arxiv-custom/semantic_similarity/bert-semantic-similarity/text_None/epochs_20
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode test.paws --dataset-name custom/semantic_similarity --model-name bert-semantic-similarity --eval-ckpt-path ../checkpoints/arxiv-custom/semantic_similarity/bert-semantic-similarity/text_None/epochs_20
```
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --custom-pretrained-path ../checkpoints/arxiv-custom/semantic_similarity/bert-semantic-similarity/text_None
```

#### multi-task bert models
```
```

#### data augmentations/variations
transliterated, fully-hindi, fully-english inputs
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --fusion-text-types text text_en text_hi text_trt --fusion-strategy concat
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --fusion-text-types text text_en text_hi text_trt --fusion-strategy max_pool
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --fusion-text-types text text_en text_hi text_trt --fusion-strategy mean_pool
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name xlm-roberta-base --text-type "" --fusion-text-types text text_en text_hi text_trt --fusion-strategy max_pool --custom-pretrained-path ./run_bert_mlm/on_raw__pretrained_cs_mlm_models/xlm-roberta-base
```
other related Hinglish as well as English/Hindi datasets
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --augment-train-datasets "sentimix2020/Hinglish,subwordlstm2016/Hinglish,semeval2017_en_sa/English,iitp_product_reviews_hi_sa/Hinglish" --model-name xlm-roberta-base --text-type ""
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --augment-train-datasets "sail2017/Hinglish,subwordlstm2016/Hinglish,semeval2017_en_sa/English,iitp_product_reviews_hi_sa/Hinglish" --model-name xlm-roberta-base --text-type ""
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name kumaretal_2019_agg/Hinglish-R --augment-train-datasets "kumaretal_2019_agg/English-R" --model-name xlm-roberta-base --text-type ""
```
- plus pretrained models
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --augment-train-datasets "sentimix2020/Hinglish,subwordlstm2016/Hinglish,semeval2017_en_sa/English,iitp_product_reviews_hi_sa/Hinglish" --model-name xlm-roberta-base --text-type "" --custom-pretrained-path ./run_bert_mlm/on_raw__pretrained_cs_mlm_models/xlm-roberta-base
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --augment-train-datasets "sail2017/Hinglish,subwordlstm2016/Hinglish,semeval2017_en_sa/English,iitp_product_reviews_hi_sa/Hinglish" --model-name xlm-roberta-base --text-type "" --custom-pretrained-path ./run_bert_mlm/on_raw__pretrained_cs_mlm_models/xlm-roberta-base
```
- plus pretrained models, plus synthetically labelled data
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --augment-train-datasets "synthetic_mt_all/Hinglish,sail2017/Hinglish,subwordlstm2016/Hinglish,semeval2017_en_sa/English,iitp_product_reviews_hi_sa/Hinglish" --model-name xlm-roberta-base --custom-pretrained-path ./run_bert_mlm/on_raw__pretrained_cs_mlm_models/xlm-roberta-base --text-type ""
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --augment-train-datasets "synthetic_mt_all/Hinglish" --model-name xlm-roberta-base --custom-pretrained-path ./run_bert_mlm/on_raw__pretrained_cs_mlm_models/xlm-roberta-base --text-type ""
```
synthetic data due to back-labeling
```
# go to ../datasets/synthetic/ and run `synthetic_datasets.py`
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --augment-train-datasets "synthetic/sail2017_baseline_mt_all_0.25/Hinglish" --model-name xlm-roberta-base --text-type ""
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --augment-train-datasets "synthetic/sail2017_others_mt_all_0.40/Hinglish" --model-name xlm-roberta-base --text-type ""
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --augment-train-datasets "synthetic/sentimix2020_baseline_mt_all_0.25/Hinglish" --model-name xlm-roberta-base --text-type ""
```
synthetic data due to noisy text creation
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name custom/noisy/sail2017/Hinglish --model-name xlm-roberta-base --text-type "+noisy_11927+noisy_2020"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name custom/noisy/sentimix2020/Hinglish --model-name xlm-roberta-base --text-type "+noisy_11927+noisy_2020"
```

#### towards char-invariance
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name bert-lstm --text-type "" --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name fasttext-lstm --text-type "" --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name charlstmlstm --text-type "" --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name sclstm --text-type "" --max-epochs 10
```
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name bert-fasttext-lstm --text-type "" --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name bert-charlstm-lstm --text-type "" --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name bert-sc-lstm --text-type "" --max-epochs 10
```
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name bert-charlstm-lstm-v2 --text-type "" --max-epochs 10 --custom-pretrained-path "../checkpoints/arxiv-sail2017/Hinglish/baseline/xlm-roberta-base/text_raw"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name bert-charlstm-lstm-v2 --text-type "" --max-epochs 10 --custom-pretrained-path "../checkpoints/arxiv-sail2017/Hinglish/data_aug with MLM pretraining/xlm-roberta-base/text_raw" --augment-train-datasets "sentimix2020/Hinglish,subwordlstm2016/Hinglish,semeval2017_en_sa/English,iitp_product_reviews_hi_sa/Hinglish"
```
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sail2017/Hinglish --model-name bert-sc-lstm --text-type "" --max-epochs 10 --augment-train-datasets "sentimix2020/Hinglish,subwordlstm2016/Hinglish,semeval2017_en_sa/English,iitp_product_reviews_hi_sa/Hinglish"
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name custom/noisy/sentimix2020/Hinglish --model-name bert-charlstm-lstm --text-type "+noisy_11927+noisy_2020" --max-epochs 10
```

#### Language-informed modeling
```
cd checkpoints/pretrained
python download_huggingface_models.py
```
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name li-xlm-roberta-base --text-type "" --custom-pretrained-path ../checkpoints/pretrained/xlm-roberta-base --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name li-xlm-roberta-base --text-type "" --custom-pretrained-path ./run_bert_mlm/on_raw__pretrained_cs_mlm_models/xlm-roberta-base --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name sentimix2020/Hinglish --model-name li-xlm-roberta-base --text-type "" --custom-pretrained-path ./run_li-bert_mlm/on_raw__mlmpretrained_pretrained_cs_li-mlm_models/xlm-roberta-base --max-epochs 10
```

#### POS-informed modeling
```
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name custom/parul/sail2017/Hinglish --model-name posi-xlm-roberta-base --text-type parul --custom-pretrained-path ../checkpoints/pretrained/xlm-roberta-base --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name custom/parul/sentimix2020/Hinglish --model-name posi-xlm-roberta-base --text-type parul --custom-pretrained-path ../checkpoints/pretrained/xlm-roberta-base --max-epochs 10
CUDA_VISIBLE_DEVICES=0 python run_classification.py --mode train_test --dataset-name custom/parul/sentimix2020/Hinglish --model-name posi-bert-base-multilingual-cased --text-type parul --custom-pretrained-path ../checkpoints/pretrained/bert-base-multilingual-cased --max-epochs 10
```

# tagging models
Usage with cross validation
```
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type postags --mode train --dataset-name gluecos_pos_ud/Hinglish/ --model-name xlm-roberta-base --cross-validate True
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type postags --mode dev --dataset-name gluecos_pos_ud/Hinglish/ --model-name xlm-roberta-base --cross-validate True --eval-ckpt-path ../checkpoints/gluecos_pos_ud/Hinglish/xlm-roberta-base/
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type postags --mode train --dataset-name gluecos_pos_fg/Hinglish/ --model-name xlm-roberta-base --cross-validate True --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base
```
```
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type nertags --mode train --dataset-name gluecos_ner/Hinglish/ --model-name xlm-roberta-base --cross-validate True
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type nertags --mode train --dataset-name gluecos_ner/Hinglish/ --model-name xlm-roberta-base --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base
```
```
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type langids --mode train --dataset-name lince_lid/Hinglish/ --model-name xlm-roberta-base --cross-validate True
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type langids --mode train --dataset-name lid_all/ --model-name xlm-roberta-base --custom-pretrained-path ./run_bert_mlm/on_raw_new__pretrained_cs_mlm_models/xlm-roberta-base
```
Usage without cross-validation usage:
```
CUDA_VISIBLE_DEVICES=0 python run_lid.py --text-type text --tag-type postags --mode train_test --dataset-name gluecos_pos_ud/Hinglish/ --model-name xlm-roberta-base
```

# transliterate (transliterate_simple.py)
- takes 5 minutes for sentimix (20k)
```
python transliterate_simple.py --base-path ../datasets/sentimix2020/Hinglish/ --files train.jsonl dev.jsonl test.jsonl
python transliterate_simple.py --base-path ../datasets/sail2017/Hinglish/ --files train.jsonl dev.jsonl test.jsonl
python transliterate_simple.py --base-path ../datasets/kumaretal_2019_agg/Hinglish-R/ --files train.jsonl dev.jsonl test.jsonl
```
```
python transliterate_simple.py --base-path ../datasets/kumaretal_2019_agg/Hinglish-R/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
python transliterate_simple.py --base-path ../datasets/kumaretal_2020_agg/Hinglish-R/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
python transliterate_simple.py --base-path ../datasets/vijayetal_2018_hatespeech/Hinglish/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
python transliterate_simple.py --base-path ../datasets/kauretal_2019_reviews/Hinglish/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
```

# transliterate (transliterate_with_langids.py) to obtain eng-devanagari mixed scripts in each sentence
- takes 5 minutes for sentimix (20k)
```
python transliterate_with_langids.py --base-path ../datasets/sentimix2020/Hinglish/ --files train.jsonl dev.jsonl test.jsonl
python transliterate_with_langids.py --base-path ../datasets/hinglishpedia/Hinglish/ --files train.jsonl dev.jsonl test.jsonl
```
```
python transliterate_with_langids.py --base-path ../datasets/kumaretal_2019_agg/Hinglish-R/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
python transliterate_with_langids.py --base-path ../datasets/kumaretal_2020_agg/Hinglish-R/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
python transliterate_with_langids.py --base-path ../datasets/vijayetal_2018_hatespeech/Hinglish/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
python transliterate_with_langids.py --base-path ../datasets/kauretal_2019_reviews/Hinglish/ --files train.jsonl dev.jsonl test.jsonl --text-type text_msftlid --langids-type langids_msftlid
```
```
python transliterate_with_langids.py --base-path ../datasets/lince_lid/Hinglish/ --files train.jsonl dev.jsonl
python transliterate_with_langids.py --base-path ../datasets/gluecos_ner/Hinglish/ --files train.jsonl dev.jsonl --text-type text_msftlid --langids-type langids_msftlid
```

# translate
- takes 3 minutes for sentimix (20k)
```
python translate.py --base-path ../datasets/sentimix2020/Hinglish/ --files train.jsonl dev.jsonl test.jsonl
```

# run masked language modeling
```
see ./run_bert_mlm 
and ./run_li-bert_mlm 
```