import os
import copy
import argparse
import jsonlines
from tqdm import tqdm

import sys
sys.path.append("../../scripts")
from indictrans import Transliterator


def main():
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument(
        "--files",
        nargs='+',
        help="List all jsonl files to which transliterations have to be added",
    )
    parser.add_argument(
        "--base-path",
        type=str,
    )
    parser.add_argument(
        "--text-type",
        type=str,
        default='text',
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default='eng',
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default='hin',
    )
    args = parser.parse_args()

    src2tgt = Transliterator(source=args.src_lang, target=args.tgt_lang)

    for f in args.files:
        new_samples = []
        with jsonlines.open(os.path.join(args.base_path, f), 'r') as reader:
            for sample in tqdm(reader):
                new_sample = copy.deepcopy(sample)
                tokens = sample[args.text_type].split(" ")
                new_tokens = [src2tgt.transform(token) for token in tokens]
                new_sample[args.text_type + "_trt"] = " ".join(new_tokens)
                new_samples.append(new_sample)
        with jsonlines.open(os.path.join(args.base_path, f), 'w') as writer:
            for new_sample in new_samples:
                writer.write(new_sample)


if __name__ == "__main__":
    main()
