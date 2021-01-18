import sys
sys.path.append("..")

import os
from tqdm import tqdm
from datasets import read_datasets_jsonl

DATSETS_PATH = "../../datasets"

""" presenting original LIDs and msft LIDs side by side for processed text in sentimix2020 """
dataset_name = "sentimix2020/Hinglish"
files = ["dev", "test", "train"]
dest_folder = os.path.join(DATSETS_PATH, dataset_name, "presentations")
if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)
for file_name in files:
    oblivious = 0
    file_path = os.path.join(DATSETS_PATH, dataset_name, f"{file_name}.jsonl")
    if f"{file_name}.jsonl" not in os.listdir(os.path.join(DATSETS_PATH, dataset_name)):
        print(f"file_name {file_name} not found in {os.path.join(DATSETS_PATH, dataset_name)}. moving ahead...")
        print(os.listdir(os.path.join(DATSETS_PATH, dataset_name)))
        continue
    examples = read_datasets_jsonl(file_path, mode=file_name)
    opfile = open(os.path.join(dest_folder, f"{file_name}_lid_comparison.csv"), "w")
    for ex in tqdm(examples):
        text_pp, langids_pp, langids_msftlid = ex.text_pp.split(), ex.langids_pp.split(), ex.langids_msftlid.split()
        if not (len(text_pp) == len(langids_pp) == len(langids_msftlid)):
            # print(ex.uid, ex.text_pp)
            # apparently, msft_lid process has some issues when the sentence has the token "/"
            # also, due to some different tokenization strategy followed by msft_lid processor,
            #   some resullts might not match
            oblivious += 1
            continue
        for a, b, c in zip(text_pp, langids_pp, langids_msftlid):
            c = "Hin" if c == "HI" else "Eng" if c == "EN" else "O"
            opfile.write(f"{a},{b},{c}\n")
        opfile.write("\n")
    print(f"missed comparison for {oblivious} number of lines")
    opfile.close()


