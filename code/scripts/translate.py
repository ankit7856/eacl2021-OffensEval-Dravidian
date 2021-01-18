from google.cloud import translate
import argparse
import jsonlines
from tqdm import tqdm
import copy
import os
import re


# Make sure export GOOGLE_APPLICATION_CREDENTIALS=My\ First\ Project-62b86dcd7bce.json
def google_translate(text_list, target_language):
    # print(text)
    client = translate.TranslationServiceClient()
    parent = "projects/hybrid-formula-290820/locations/global"
    response_list = []

    response = client.translate_text(text_list, target_language, parent)
    for translation in response.translations:
        response_list.append(translation.translated_text)
    # print(response.translations[0].translated_text)
    return response_list


# The variables in the following function were named in the context of translating the entire sentence to Devanagari
def translate_file(file_path, target_language, args):
    to_translate = []
    new_samples = []
    with jsonlines.open(file_path, 'r') as reader:
        for sample in tqdm(reader):
            new_sample = copy.deepcopy(sample)
            tokens = sample[args.text_type].split(" ")
            langids = sample[args.langids_type].split(" ")
            text_fd = []
            eng_phrase = []
            for (token, lang_id) in zip(tokens, langids):
                if lang_id.lower() == target_language:
                    if len(eng_phrase) != 0:
                        to_translate.append(" ".join(eng_phrase))
                        text_fd.append("UNIQUE_TRANSLATION_ID_" + str(len(to_translate)))
                        eng_phrase.clear()
                    text_fd.append(token)
                else:
                    eng_phrase.append(token)
            if len(eng_phrase) != 0:
                to_translate.append(" ".join(eng_phrase))
                text_fd.append("UNIQUE_TRANSLATION_ID_" + str(len(to_translate)))
            new_sample["text_" + target_language] = " ".join(text_fd)
            new_samples.append(new_sample)

    def my_repl(matchobj):
        ans = response_translations[int(matchobj.group(0).split("_")[-1])-1]
        return ans

    response_translations = []
    for i in range(int(len(to_translate) / 1024) + 1):
        response_translations.extend(google_translate(to_translate[i * 1024: (i + 1) * 1024], target_language))
    assert (len(to_translate) == len(response_translations))
    for sample in new_samples:
        new_text_fd = re.sub("UNIQUE_TRANSLATION_ID_\d{1,}", my_repl, sample["text_" + target_language])
        sample["text_" + target_language] = new_text_fd
    with jsonlines.open(file_path, 'w') as writer:
        for new_sample in new_samples:
            writer.write(new_sample)


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
        default='text_trt',
    )
    parser.add_argument(
        "--langids-type",
        type=str,
        default='langids_trt',
    )
    args = parser.parse_args()

    for f in args.files:
        translate_file(os.path.join(args.base_path, f), "hi", args)
        translate_file(os.path.join(args.base_path, f), "en", args)


if __name__ == "__main__":
    main()