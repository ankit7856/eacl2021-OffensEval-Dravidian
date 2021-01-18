import os
import json
import jsonlines

# src_folder_path = "../../checkpoints/arxiv-sentimix2020/Hinglish/data_aug with MLM pretraining/xlm-roberta-base/text_raw/2020-11-25_18:19:29.976985"
src_folder_path = "../../checkpoints/arxiv-sail2017/Hinglish/data_aug with MLM pretraining/xlm-roberta-base/text_raw/2020-11-25_16:27:20.929826"
dest_folder_path = os.path.join(src_folder_path, "jsonl2json")
if not os.path.exists(dest_folder_path):
    os.makedirs(dest_folder_path)
filenames = os.listdir(src_folder_path)

for filename in filenames:
    if filename.startswith("errors") and filename.endswith(".jsonl"):
        print(filename)
        lines = [line for line in jsonlines.open(os.path.join(src_folder_path, filename))]
        with open(os.path.join(dest_folder_path, filename[:-1]), "w") as opfile:
            json.dump(lines, opfile, sort_keys=True, indent=4, ensure_ascii=False)
            opfile.close()


print("complete")
