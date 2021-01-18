import os
import copy
import argparse
import jsonlines
from tqdm import tqdm
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
        "--text-type-target-name",
        type=str,
        default="",
    )
    parser.add_argument(
        "--langids-type",
        type=str,
        default='langids',
    )
    parser.add_argument(
        "--langids-type-target-name",
        type=str,
        default="",
    )
    args = parser.parse_args()
    eng2hin = Transliterator(source='eng', target='hin')

    # eng = ['Thats a sentence!! """" .... Aapke shubh chintakk lalit jaiswal ke taraf se aapko aapki jeet ki '
    #        'hardik subhkamnaye']
    # hin = [eng2hin.transform(e) for e in eng]
    # print(hin)

    text_type_target_name, langids_type_target_name = args.text_type_target_name, args.langids_type_target_name
    if not text_type_target_name.strip():
        text_type_target_name = args.text_type + "_trt"
    if not langids_type_target_name.strip():
        langids_type_target_name = args.langids_type + "_trt"

    for f in args.files:
        new_samples = []
        with jsonlines.open(os.path.join(args.base_path, f), 'r') as reader:
            for sample in tqdm(reader):
                new_sample = copy.deepcopy(sample)
                tokens = sample[args.text_type].split(" ")
                langids = sample[args.langids_type].split(" ")
                new_tokens, new_langids = zip(*[(eng2hin.transform(token), langid) if langid.lower() == "hi"
                                                else (token, langid) for (token, langid) in zip(tokens, langids)
                                                if token != ""])
                assert len(new_tokens) == len(new_langids)
                new_sample[text_type_target_name] = " ".join(new_tokens)
                new_sample[langids_type_target_name] = " ".join(new_langids)
                new_samples.append(new_sample)
        with jsonlines.open(os.path.join(args.base_path, f), 'w') as writer:
            for new_sample in new_samples:
                writer.write(new_sample)


if __name__ == "__main__":
    main()
