### huggingface models

This directory contains pretrained models from huggingface. Due to some logic in the code, it is not posiible to directly use any given huggingface model although it's architecture is same as BERT or XLM. 

Instead, you first need to download the huggingface pretrained model's weights by running the script in this folder [download_huggingface_models.py](./download_huggingface_models.py), and then while using code from scripts folder, use the argument _--custom-pretrained-path_ by specifying the correct model architecture with the argument _--model-name=bert-base-cased_ or _--model-name=xlm-roberta-base_.

See README in the ```./scripts``` folder for example usage.

