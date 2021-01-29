## Datasets
- Download official data, convert to jsonl format (because that's the format the code base expects!)
```
bash make_data.sh
```
- Transliterate to romanized (!! alo included in make_data.sh above !!)
```
python transliterate.py --base-path ./offeval/kannada/ --src-lang kan --tgt-lang eng --files train.jsonl dev.jsonl test.jsonl
python transliterate.py --base-path ./offeval/tamil/ --src-lang kan --tgt-lang eng --files train.jsonl dev.jsonl test.jsonl
python transliterate.py --base-path ./offeval/malayalam/ --src-lang kan --tgt-lang eng --files train.jsonl dev.jsonl test.jsonl
```

## To pretrain using task specific datasets, do the following
```
cd ../../../pretraining
bash offeval_tam_mal_kan__bert-base-multilingual-cased.sh
bash offeval_tam_mal_kan__xlm-roberta-base.sh
```

## To run classification models (careful about the paths you specify in various arguments)
move to [scripts](../../scripts) and run ```run_eacl2021.py``` by using commands below:
- simple baselines
```
CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name bert-base-multilingual-cased --text-type "" --dataset-name eacl2021/offeval/kannada
CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/kannada
```
- w/ pretrained models in huggingface
  (requires downloading of respective models using [download_huggingface_models.py](../../../pretraining/huggingface/download_huggingface_models.py))
```
CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/kannada --custom-pretrained-path ../../pretraining/huggingface/ai4bharat/indic-bert
```
- w/ custom pretrained models
```
CUDA_VISIBLE_DEVICES=0 python run_eacl2021.py --mode train_dev --model-name xlm-roberta-base --text-type "" --dataset-name eacl2021/offeval/kannada --custom-pretrained-path ../../pretraining/eacl2021/pretraining/offeval/kannada/xlm-roberta-base/
```

## Resources
- To download ```indic-trans```, follow [README.md](../../scripts/indictrans/README.md) to install transliterator
