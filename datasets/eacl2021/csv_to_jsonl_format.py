import os
import jsonlines

for lang in ["tamil", "malayalam", "kannada"]:

    for filename in ["train", "dev", "test"]:

        lines = [line.strip() for line in open(os.path.join("offeval", lang, filename+".tsv"))]
        print(os.path.join("offeval", lang, filename + ".tsv"), len(lines))

        if filename == "test":

            texts = lines
            opfile = jsonlines.open(os.path.join("offeval", lang, filename+".jsonl"), "w")
            for i, text in enumerate(texts):
                dct = {"uid": i, "text": text.strip(), "label": None}
                opfile.write(dct)
            opfile.close()

        else:

            splits = [line.split("\t") for line in lines]
            texts, labels = list(zip(*splits))
            opfile = jsonlines.open(os.path.join("offeval", lang, filename+".jsonl"), "w")
            for i, (text, label) in enumerate(zip(texts, labels)):
                dct = {"uid": i, "text": text.strip(), "label": label.strip()}
                opfile.write(dct)
            opfile.close()

print("complete")

"""
offeval/tamil/train.tsv 35139
offeval/tamil/dev.tsv 4388
offeval/tamil/test.tsv 4392
offeval/malayalam/train.tsv 16010
offeval/malayalam/dev.tsv 1999
offeval/malayalam/test.tsv 2001
offeval/kannada/train.tsv 6217
offeval/kannada/dev.tsv 777
offeval/kannada/test.tsv 778
complete
"""
