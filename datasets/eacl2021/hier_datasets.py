import os
import jsonlines

"""
create four data folders: 
    lang_classy, off_not_classy, off_classy, non_lang_classy

+-----------------+--------------------------+----------+
| lang_classy     | lang                     | not_lang |
+-----------------+--------------------------+----------+
| off_not_classy  | off                      | not      |
+-----------------+--------------------------+----------+
| off_classy      | fine-grained OFF classes |          |
+-----------------+--------------------------+----------+
| non_lang_classy | fine-grained OFF classes | not      |
+-----------------+--------------------------+----------+
"""


def create_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


for lang in ["kannada", "tamil", "malayalam"]:

    main_folder = f"./offeval/{lang}/hier"
    create_dirs(main_folder)

    for file in ["train.jsonl", "dev.jsonl", "test.jsonl"]:
        print(file)
        #
        # lang_classy
        lines = [line for line in jsonlines.open(os.path.join(f"./offeval/{lang}", file))]
        if file != "test.jsonl":
            for i, line in enumerate(lines):
                line["label"] = "not_lang" if line["label"].lower() == f"not-{lang}" else "lang"
                lines[i] = line
        this_folder = os.path.join(main_folder, "lang_classy")
        create_dirs(this_folder)
        opfile = jsonlines.open(os.path.join(this_folder, file), "w")
        print(f"# lines: {len(lines)}")
        for line in lines:
            opfile.write(line)
        opfile.close()
        #
        # off_not_classy
        lines = [line for line in jsonlines.open(os.path.join(f"./offeval/{lang}", file))]
        if file != "test.jsonl":
            new_lines = []
            for i, line in enumerate(lines):
                label = line["label"]
                if "offensive" not in label.lower():
                    continue
                line["label"] = "Offensive" if label != "Not_offensive" else label
                new_lines.append(line)
            lines = [line for line in new_lines]
        this_folder = os.path.join(main_folder, "off_not_classy")
        create_dirs(this_folder)
        opfile = jsonlines.open(os.path.join(this_folder, file), "w")
        print(f"# lines: {len(lines)}")
        for line in lines:
            opfile.write(line)
        opfile.close()
        #
        # off_classy
        lines = [line for line in jsonlines.open(os.path.join(f"./offeval/{lang}", file))]
        if file != "test.jsonl":
            new_lines = []
            for i, line in enumerate(lines):
                label = line["label"]
                if ("offensive" not in label.lower()) or (label == "Not_offensive"):
                    continue
                new_lines.append(line)
            lines = [line for line in new_lines]
        this_folder = os.path.join(main_folder, "off_classy")
        create_dirs(this_folder)
        opfile = jsonlines.open(os.path.join(this_folder, file), "w")
        print(f"# lines: {len(lines)}")
        for line in lines:
            opfile.write(line)
        opfile.close()
        #
        # non_lang_classy
        lines = [line for line in jsonlines.open(os.path.join(f"./offeval/{lang}", file))]
        if file != "test.jsonl":
            new_lines = []
            for i, line in enumerate(lines):
                label = line["label"]
                if "offensive" not in label.lower():
                    continue
                new_lines.append(line)
            lines = [line for line in new_lines]
        this_folder = os.path.join(main_folder, "non_lang_classy")
        create_dirs(this_folder)
        opfile = jsonlines.open(os.path.join(this_folder, file), "w")
        print(f"# lines: {len(lines)}")
        for line in lines:
            opfile.write(line)
        opfile.close()

"""
output
-----
muralidhar@Sais-MacBook-Pro eacl2021 % python hier_datasets.py
train.jsonl
# lines: 6217
# lines: 4695
# lines: 1151
# lines: 4695
dev.jsonl
# lines: 777
# lines: 586
# lines: 160
# lines: 586
test.jsonl
# lines: 778
# lines: 778
# lines: 778
# lines: 778
train.jsonl
# lines: 35139
# lines: 33685
# lines: 8260
# lines: 33685
dev.jsonl
# lines: 4388
# lines: 4216
# lines: 1023
# lines: 4216
test.jsonl
# lines: 4392
# lines: 4392
# lines: 4392
# lines: 4392
train.jsonl
# lines: 16010
# lines: 14723
# lines: 570
# lines: 14723
dev.jsonl
# lines: 1999
# lines: 1836
# lines: 57
# lines: 1836
test.jsonl
# lines: 2001
# lines: 2001
# lines: 2001
# lines: 2001
complete
"""

print("complete")

# """
# save individual classes (mainly for data analysis)
# """
# for lang in ["tamil"]:
#     save_path = f"./offeval/{lang}/individual_class_data"
#     create_dirs(save_path)
#     train_lines = [line for line in jsonlines.open(os.path.join("./offeval", lang, "train.jsonl"))]
#     class_wise = {}
#     for line in train_lines:
#         label, text = line["label"], line["text"]
