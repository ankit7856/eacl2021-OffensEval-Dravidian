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
        "--langids-type",
        type=str,
        default='langids',
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

    eng2hin = Transliterator(source=args.src_lang, target=args.tgt_lang)

    # eng = ['Thats a sentence!! """" .... Aapke shubh chintakk lalit jaiswal ke taraf se aapko aapki '
    #        'jeet ki hardik subhkamnaye']
    # hin = [eng2hin.transform(e) for e in eng]
    # print(hin)

    for f in args.files:
        new_samples = []
        with jsonlines.open(os.path.join(args.base_path, f), 'r') as reader:
            for sample in tqdm(reader):
                new_sample = copy.deepcopy(sample)
                tokens = sample[args.text_type].split(" ")
                new_tokens = [eng2hin.transform(token) for token in tokens]
                new_sample[args.text_type + "_D"] = " ".join(new_tokens)
                if args.langids_type in sample and sample[args.langids_type]:
                    langids = sample[args.langids_type].split(" ")
                    assert len(langids) == len(tokens) == len(new_tokens)
                    non_english = [token for token, langid in zip(tokens, langids) if langid != "en"]
                    non_hindi = [token for token, langid in zip(tokens, langids) if langid != "hi"]
                    non_english_devanagari = [token for token, langid in zip(new_tokens, langids) if langid != "en"]
                    new_sample[args.text_type + "_non_english"] = " ".join(non_english)
                    new_sample[args.text_type + "_non_hindi"] = " ".join(non_hindi)
                    new_sample[args.text_type + "_non_english_D"] = " ".join(non_english_devanagari)
                new_samples.append(new_sample)
        with jsonlines.open(os.path.join(args.base_path, f), 'w') as writer:
            for new_sample in new_samples:
                writer.write(new_sample)


if __name__ == "__main__":
    main()
