#!/bin/bash

python gdrive_downloads.py
python make_data.py

#python transliterate.py --base-path ./offeval/kannada/ --src-lang kan --tgt-lang eng --files train.jsonl dev.jsonl test.jsonl
#python transliterate.py --base-path ./offeval/tamil/ --src-lang kan --tgt-lang eng --files train.jsonl dev.jsonl test.jsonl
#python transliterate.py --base-path ./offeval/malayalam/ --src-lang kan --tgt-lang eng --files train.jsonl dev.jsonl test.jsonl

#python combine_datasets.py
#python hier_datasets.py
