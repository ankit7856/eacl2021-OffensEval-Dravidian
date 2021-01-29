import os
from transformers import AutoTokenizer, AutoModel

downoad_models = {
    # "bert-base-cased": "./bert-base-cased",
    # "bert-base-multilingual-cased": "./bert-base-multilingual-cased",
    # "xlm-roberta-base": "./xlm-roberta-base",
    # "sagorsarker/codeswitch-hineng-pos-lince": "./sagorsarker/bert-codeswitch-hineng-pos-lince",
    # "sagorsarker/codeswitch-hineng-ner-lince": "sagorsarker/codeswitch-hineng-ner-lince",
    # "murali1996/bert-base-cased-spell-correction": "murali1996/bert-base-cased-spell-correction",
    # "vinai/bertweet-base": "vinai/bertweet-base",
    "ai4bharat/indic-bert": "ai4bharat/indic-bert"
}

for model_name, dest_path in downoad_models.items():
    if os.path.exists(dest_path):
        print(f"{model_name} already exists at {dest_path}, ignoring download for this model")
        continue
    else:
        try:
            os.rmdir(model_name)
        except Exception as e:
            pass
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(pretrained_model_name_or_path=model_name)
        os.makedirs(dest_path)
        print(f"created dir {dest_path} for downloading {model_name}")
        # saves config.json and pytorch_model.bin
        model.save_pretrained(model_name)
        # saves vocab.txt
        tokenizer.save_vocabulary(model_name)
        del model, tokenizer

