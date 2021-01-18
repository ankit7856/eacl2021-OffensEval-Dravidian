"""
stats
-----
offeval/tamil/train.tsv 35139
offeval/tamil/dev.tsv 4388
offeval/tamil/test.tsv 4392
offeval/malayalam/train.tsv 16010
offeval/malayalam/dev.tsv 1999
offeval/malayalam/test.tsv 2001
offeval/kannada/train.tsv 6217
offeval/kannada/dev.tsv 777
offeval/kannada/test.tsv 778
------
complete
"""

import os
import jsonlines

MAX_CHAR_LEN = 300

for lang in ["tamil", "malayalam", "kannada"]:

    pretraining_train, pretraining_test = [], []
    pretraining_data_save_path = os.path.join("offeval", lang, "pretraining_data")
    if not os.path.exists(pretraining_data_save_path):
        os.makedirs(pretraining_data_save_path)
    n_trimmed = 0

    for filename in ["train", "dev", "test"]:

        lines = [line.strip() for line in open(os.path.join("offeval", lang, filename+".tsv"))]
        print(os.path.join("offeval", lang, filename + ".tsv"), len(lines))
        
        if filename == "test":
            texts = lines
            labels = [None for _ in texts]
        else:
            splits = [line.split("\t") for line in lines]
            texts, labels = list(zip(*splits))

        if filename == "test":
            pretraining_test.extend(texts)
        else:
            pretraining_train.extend(texts)

        new_texts = []
        for txt in texts:
            if len(txt) > MAX_CHAR_LEN:
                n_trimmed += 1
                newtokens, currsum = [], 0
                for tkn in txt.split():  # count 1 char length for space
                    if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                        newtokens.append(tkn)
                        currsum += len(tkn) + 1
                    else:
                        break
                txt = " ".join(newtokens)
            new_texts.append(txt)
        texts = new_texts
        opfile = jsonlines.open(os.path.join("offeval", lang, filename+".jsonl"), "w")
        for i, (text, label) in enumerate(zip(texts, labels)):
            dct = {"uid": i, "text": text, "label": label}
            opfile.write(dct)
        opfile.close()
        print(f"n_trimmed = {n_trimmed} to max len {MAX_CHAR_LEN}")

    print(f"total train lines obtained: {len(pretraining_train)} with {n_trimmed} trimmed to max len {MAX_CHAR_LEN}")
    opfile = open(os.path.join(pretraining_data_save_path, "train.txt"), "w")
    c, cc = 0, 0
    for line in pretraining_train:
        line = line.strip()
        if not line:
            continue
        if len(line) < MAX_CHAR_LEN:
            opfile.write(f"{line}\n")
            c += 1
        else:
            curr_len, curr_tokens = 0, []
            tokens = [tkn + " " for tkn in line.split()]
            tokens[-1] = tokens[-1][:-1]
            for token in tokens:
                if curr_len + len(token) > MAX_CHAR_LEN:
                    sub_line = "".join(curr_tokens)
                    assert len(sub_line) <= MAX_CHAR_LEN, print(len(sub_line), sub_line)
                    opfile.write(f"{sub_line}\n")
                    cc += 1
                    if len(token) > MAX_CHAR_LEN:
                        break
                    curr_len, curr_tokens = 0, []
                curr_tokens.append(token)
                curr_len += len(token)
    opfile.close()
    print(f"total train lines written: {c}+{cc}={c + cc}")

    print(f"total test lines obtained: {len(pretraining_test)}")
    opfile = open(os.path.join(pretraining_data_save_path, "test.txt"), "w")
    c, cc = 0, 0
    for line in pretraining_test:
        line = line.strip()
        if not line:
            continue
        if len(line) < MAX_CHAR_LEN:
            opfile.write(f"{line}\n")
            c += 1
        else:
            curr_len, curr_tokens = 0, []
            tokens = [tkn + " " for tkn in line.split()]
            tokens[-1] = tokens[-1][:-1]
            for token in tokens:
                if curr_len + len(token) > MAX_CHAR_LEN:
                    sub_line = "".join(curr_tokens)
                    assert len(sub_line) <= MAX_CHAR_LEN, print(len(sub_line), sub_line)
                    opfile.write(f"{sub_line}\n")
                    cc += 1
                    if len(token) > MAX_CHAR_LEN:
                        break
                    curr_len, curr_tokens = 0, []
                curr_tokens.append(token)
                curr_len += len(token)
    opfile.close()
    print(f"total test lines written: {c}+{cc}={c + cc}")
