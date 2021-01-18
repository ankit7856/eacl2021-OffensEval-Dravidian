import os
import jsonlines

os.makedirs(os.path.join("./offeval", "combined"))

for file in ["train.jsonl", "test.jsonl", "dev.jsonl"]:

    combined_lines = []

    for lang in ["kannada", "malayalam", "tamil"]:

        lines = jsonlines.open(os.path.join("./offeval", lang, file))
        combined_lines.extend(lines)

    opfile = jsonlines.open(os.path.join("./offeval", "combined", file), "w")
    for line in combined_lines:
        opfile.write(line)
    opfile.close()

os.system("cp ./offeval/kannada/dev.jsonl ./offeval/combined/dev_kannada.jsonl")
os.system("cp ./offeval/tamil/dev.jsonl ./offeval/combined/dev_tamil.jsonl")
os.system("cp ./offeval/malayalam/dev.jsonl ./offeval/combined/dev_malayalam.jsonl")
