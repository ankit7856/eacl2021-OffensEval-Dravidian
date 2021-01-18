import os
import re
import csv
import json
import random
import jsonlines
from time import time
from tqdm import tqdm
from collections import namedtuple

from helpers import progress_bar
from preprocess import clean_generic, clean_sail2017_lines, clean_sentimix2020_lines

# from indictrans import Transliterator
# TRANSLITERATOR = Transliterator(source='eng', target='hin')

# Make sure export GOOGLE_APPLICATION_CREDENTIALS=My\ First\ Project-62b86dcd7bce.json
# from google.cloud import translate

SEED = 11927
FIELDS = ["dataset", "task", "split_type", "uid",
          "text", "langids", "label", "seq_labels",
          "text_pp", "langids_pp",
          "meta_data"]
# EXAMPLE = namedtuple(f"example", FIELDS, defaults=(None,) * len(FIELDS))
EXAMPLE = namedtuple(f"example", FIELDS)
MAX_CHAR_LEN = None


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


def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return


def read_csv_file(path, has_header=True, delimiter=","):
    with open(path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=delimiter)
        if has_header:
            lines = [row for row in csv_reader][1:]
        else:
            lines = [row for row in csv_reader]
    return lines


def read_datasets_jsonl(path, mode=""):
    examples = []
    for i, line in enumerate(jsonlines.open(path)):
        if i == 0:
            fields_ = line.keys()
            # Example = namedtuple(f"{mode}_example", fields_, defaults=(None,) * len(fields_))
            Example = namedtuple(f"{mode}_example", fields_)
        examples.append(Example(**line))
    print(f"in read_datasets_jsonl(): path:{path}, mode:{mode}, #examples:{len(examples)}")
    return examples


def read_lince_downloads(path, mode, dataset_name, task_name, is_pos=False, is_ner=False, standardizing_tags={}):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN
    tokens, langids, postags, nertags, label, uid = [], [], [], [], None, None
    FIELDS += [fieldname for fieldname in ("nertags", "postags") if fieldname not in FIELDS]

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for line_num, line in enumerate(all_lines):
        if line.strip() == "":
            if mode == "test":
                uid = len(examples)
            txt = " ".join(tokens)
            new_txt = clean_generic(txt)
            if new_txt.strip() == "":
                new_txt = txt
            if len(new_txt) > MAX_CHAR_LEN:
                n_trimmed += 1
                newtokens, currsum = [], 0
                for tkn in new_txt.split():  # 1 for space
                    if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                        newtokens.append(tkn)
                        currsum += len(tkn) + 1
                    else:
                        break
                new_txt = " ".join(newtokens)
            example = Example(dataset=dataset_name,
                              task=task_name,
                              split_type=mode,
                              uid=uid,
                              label=label,
                              text=txt,
                              text_pp=new_txt,
                              langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                                for lid in langids]) if langids else None,
                              postags=" ".join(postags) if postags else None,
                              nertags=" ".join(nertags) if nertags else None)
            examples.append(example)

            # because test does not have a next line as `# sent_enum = xx`
            if mode == "test":
                label = None
                uid = None
                tokens, langids, postags, nertags = [], [], [], []
        elif "# sent_enum =" in line:
            # start a new line and reset field values
            vals = line.strip().split("\t")
            label = vals[-1] if len(vals) > 1 else None
            uid = vals[0].split("=")[-1].strip()
            tokens, langids, postags, nertags = [], [], [], []
        else:
            vals = line.strip().split("\t")
            if not mode == "test":
                tokens.append(vals[0])
                langids.append(vals[1])
                if is_pos:
                    postags.append(vals[2])
                elif is_ner:
                    nertags.append(vals[2])
            else:
                tokens.append(vals[0])
                if is_pos or is_ner:
                    langids.append(vals[1])
        progress_bar(line_num, len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_gluecos_downloads(path, mode, dataset_name, task_name, is_pos=False, is_ner=False,
                           standardizing_tags={}):
    st_time = time()

    global FIELDS
    tokens, langids, postags, nertags, label, uid = [], [], [], [], None, None
    FIELDS += [fieldname for fieldname in ("nertags", "postags") if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for line_num, line in enumerate(all_lines):
        if line.strip() == "":
            if not tokens:
                continue
            uid = len(examples)
            tokens = ["".join(tkn.split()) for tkn in tokens]
            txt = " ".join(tokens)
            new_txt = clean_generic(txt)
            if new_txt.strip() == "":
                new_txt = txt
            example = Example(dataset=dataset_name,
                              task=task_name,
                              split_type=mode,
                              uid=uid,
                              label=label,
                              text=txt,
                              text_pp=new_txt,
                              langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                                for lid in langids]) if langids else None,
                              postags=" ".join(postags) if postags else None,
                              nertags=" ".join(nertags) if nertags else None)
            examples.append(example)
            tokens, langids, postags, nertags, label, uid = [], [], [], [], None, None
        else:
            vals = line.strip().split("\t")
            if not mode == "test":
                if is_pos:
                    if len(vals) < 3:
                        continue
                    tokens.append(vals[0])
                    langids.append(vals[1])
                    postags.append(vals[2])
                elif is_ner:
                    if len(vals) < 2:
                        continue
                    tokens.append(vals[0])
                    nertags.append(vals[1])
            else:
                tokens.append(vals[0])
                if is_pos:
                    langids.append(vals[1])
        progress_bar(line_num, len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)}")
    return examples


def read_vsingh_downloads(path, mode="train"):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    tokens_dict, tags_dict = {}, {}
    for line_num, line in enumerate(all_lines):
        line = line.strip()
        if line_num == 0 or "sent" not in line:
            continue
        line_tokens = line.split(",")
        if '","' in line:
            _id_info, word, tag = line_tokens[0], ",", line_tokens[-1]
        else:
            _id_info, word, tag = line_tokens[0], line_tokens[1], line_tokens[-1]
        _id = _id_info[6:]
        if not _id in tokens_dict:
            tokens_dict[_id], tags_dict[_id] = [], []
        tokens_dict[_id].append(word)
        tags_dict[_id].append(tag)

    for _id in tokens_dict:
        txt = " ".join(tokens_dict[_id])
        tags = " ".join(tags_dict[_id])
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            # new_txt = txt
            continue
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum+len(tkn)+1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn)+1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="vsinghetal_2018",
                          task="seq_tagging",
                          split_type=mode,
                          uid=_id,
                          text=txt,
                          langids=tags,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(tokens_dict), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_mt1_downloads(path1, path2, mode, dataset_name, task_name):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    FIELDS += [fieldname for fieldname in ["tgt", "tgt_pp", ] if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    txt_lines = [line.strip() for line in open(path1, "r")]
    tgt_lines = [line.strip() for line in open(path2, "r")]
    for i, (txt, tgt) in enumerate(zip(txt_lines, tgt_lines)):
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        new_tgt = clean_generic(tgt)
        if new_tgt.strip() == "":
            new_tgt = tgt
        if len(new_tgt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_tgt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_tgt = " ".join(newtokens)
        example = Example(dataset=dataset_name,
                          task=task_name,
                          split_type=mode,
                          uid=i,
                          text=txt,
                          text_pp=new_txt,
                          tgt=tgt,
                          tgt_pp=new_tgt)
        examples.append(example)
        progress_bar(len(examples), len(txt_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_mt2_downloads(path, mode, dataset_name, task_name):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    FIELDS += [fieldname for fieldname in ["tgt", "tgt_pp", ] if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    rows = read_csv_file(path)
    txt_lines = [line[0].strip() for line in rows]
    tgt_lines = [line[1].strip() for line in rows]
    for i, (txt, tgt) in enumerate(zip(txt_lines, tgt_lines)):
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        new_tgt = clean_generic(tgt)
        if new_tgt.strip() == "":
            new_tgt = tgt
        if len(new_tgt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_tgt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_tgt = " ".join(newtokens)
        example = Example(dataset=dataset_name,
                          task=task_name,
                          split_type=mode,
                          uid=i,
                          text=txt,
                          text_pp=new_txt,
                          tgt=tgt,
                          tgt_pp=new_tgt)
        examples.append(example)
        progress_bar(len(examples), len(txt_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_royetal2013_downloads(path, mode, standardizing_tags={}):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = [line.strip() for line in open(path, "r")]
    for i, line in enumerate(lines):
        tkns = line.split()
        txt = " ".join([tkn.split("\\")[0].strip() for tkn in tkns])
        tgs = " ".join([tkn.split("\\")[1].split("=")[0].strip() for tkn in tkns])
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="royetal2013_lid",
                          task="classification",
                          split_type=mode,
                          uid=i,
                          text=txt,
                          langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                            for lid in tgs.split(" ")]),
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_semeval2017_en_sa_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = [line.strip() for line in open(path, "r")]
    for i, line in enumerate(lines):
        try:
            uid, label, txt = line.split("\t")[:3]
        except Exception as e:
            print(path)
            print(line)
            print(line.split("\t"))
            # raise Exception
            continue
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="semeval2017_en_sa",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_iitp_product_reviews_hi_sa_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = [line.strip() for line in open(path, "r")]
    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        vals = line.split(",")
        label = vals[0]
        txt = ",".join(vals[1:])
        txt = trn_hin2eng.transform(txt)
        new_txt = "".join([char for char in txt])
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="iitp_product_reviews_hi_sa",
                          task="classification",
                          split_type=mode,
                          uid=len(examples),
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_sentimix2020_downloads(path, mode="train", test_labels: dict = {}, standardizing_tags={}):
    st_time = time()
    to_translate = []

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
        # "noeng": ['text_noeng'],
        # "nohin": ["text_nohin"],
        # "trt": ['text_trt'],
        # "trt_noeng": ["text_trt_noeng"],
        # "fd": ["text_fd"],
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    uid, label, tokens, lang_ids = None, None, None, None
    start_new = True
    lines = open(path, 'r').readlines()
    for idx, line in enumerate(lines):
        if line == "\n" or len(lines)-1 == idx and tokens:
            org_tokens = [token for token in tokens]
            org_tags = [langid for langid in lang_ids]
            assert len(org_tokens) == len(org_tags), print(len(org_tokens), len(org_tags))
            tokens, lang_ids = clean_sentimix2020_lines(org_tokens, org_tags)
            if " ".join(tokens).strip() == "":
                tokens, lang_ids = org_tokens, org_tags
            # if len(" ".join(tokens)) > MAX_CHAR_LEN:
            #     n_trimmed += 1
            #     newtokens, newlangids, currsum = [], [], 0
            #     for tkn, lgid in zip(tokens, lang_ids):  # 1 for space
            #         if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
            #             newtokens.append(tkn)
            #             newlangids.append(lgid)
            #             currsum += len(tkn) + 1
            #         else:
            #             break
            #     tokens, lang_ids = newtokens, newlangids
            example = Example(dataset="sentimix2020_hinglish",
                              task="classification",
                              split_type=mode,
                              uid=uid,
                              text=" ".join(org_tokens),
                              label=label,
                              langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                                for lid in org_tags]),
                              text_pp=" ".join(tokens),
                              langids_pp=" ".join(lang_ids))
            # if "noeng" in add_fields:
            #     text_noeng = [token for (token, lang_id) in zip(tokens, lang_ids) if lang_id.lower() != "eng"]
            #     example = example._replace(text_noeng=" ".join(text_noeng))
            # if "nohin" in add_fields:
            #     text_nohin = [token for (token, lang_id) in zip(tokens, lang_ids) if lang_id.lower() != "hin"]
            #     example = example._replace(text_nohin=" ".join(text_nohin))
            # if "trt" in add_fields:
            #     trt_tokens = [TRANSLITERATOR.transform(token) if lang_id.lower() == "hin" else token
            #                   for (token, lang_id) in zip(tokens, lang_ids)]
            #     example = example._replace(text_trt=" ".join(trt_tokens))
            # if "trt_noeng" in add_fields:
            #     text_trt_noeng = [trt_token for (trt_token, lang_id) in zip(trt_tokens, lang_ids)
            #                       if lang_id.lower() != "eng"]
            #     example = example._replace(text_trt_noeng=" ".join(text_trt_noeng))
            # if "fd" in add_fields:
            #     text_fd = []
            #     eng_phrase = []
            #     for (token, lang_id) in zip(trt_tokens, lang_ids):
            #         if lang_id.lower() == "hin":
            #             if len(eng_phrase) != 0:
            #                 to_translate.append(" ".join(eng_phrase))
            #                 text_fd.append("UNIQUE_TRANSLATION_ID_" + str(len(to_translate)))
            #                 eng_phrase.clear()
            #             text_fd.append(token)
            #         else:
            #             eng_phrase.append(token)
            #     if len(eng_phrase) != 0:
            #         to_translate.append(" ".join(eng_phrase))
            #         text_fd.append("UNIQUE_TRANSLATION_ID_" + str(len(to_translate)))
            #     example = example._replace(text_fd=" ".join(text_fd))
            # if "fe" in add_fields:
            #     text_fe = ""
            #     example = example._replace(text_fe=" ".join(text_fe))

            examples.append(example)
            progress_bar(len(examples), 14000 if mode == "train" else 3000, ["time"], [time()-st_time])
            start_new = True
            continue
        parts = line.strip().split("\t")
        if start_new:
            uid, label = (parts[1], parts[2]) if mode != "test" else (parts[1], test_labels[parts[1]])
            tokens, lang_ids = [], []
            start_new = False
        else:
            if len(parts) == 2:  # some lines are weird, e.g see uid `20134`
                tokens.append(parts[0])
                lang_ids.append(parts[1])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    def my_repl(matchobj):
        ans = response_translations[int(matchobj.group(0).split("_")[-1])-1]
        return ans

    if "fd" in add_fields:
        response_translations = []
        for i in range(int(len(to_translate)/1024)+1):
            response_translations.extend(google_translate(to_translate[i*1024: (i+1)*1024], "hi"))
        assert(len(to_translate) == len(response_translations))
        new_examples = []
        for example in examples:
            new_text_fd = re.sub("UNIQUE_TRANSLATION_ID_\d{1,}", my_repl, example.text_fd)
            example = example._replace(text_fd=new_text_fd)
            new_examples.append(example)
        examples = new_examples

    return examples


def read_sail2017_downloads(path, mode="train"):
    st_time = time()
    to_translate = []

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for uid, line in enumerate(all_lines):
        if line.strip() == "":
            continue
        txt, label = [part.strip() for part in line.strip().split("\t")][:2]
        new_txt = clean_sail2017_lines(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum+len(tkn)+1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn)+1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="sail_2017_hinglish",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          text_pp=new_txt,
                          label=label)
        examples.append(example)
        progress_bar(len(examples), len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_sail2017_downloads_new(path, labelled_dict, mode="train", standardizing_tags={}):
    st_time = time()
    to_translate = []
    label_dict = {"-1": "negative", "0": "neutral", "1": "positive"}

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)
    not_found = 0

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for line in all_lines:
        _id = line.strip()
        if _id not in labelled_dict:
            not_found += 1
            continue
        else:
            data = labelled_dict[_id]
            txt, txt_langids, label = data["text"], data["lang_tagged_text"], label_dict[str(data["sentiment"])]
        org_tokens, org_tags = [], []
        for x in txt_langids.split(" "):
            if x:
                vals = x.split("\\")
                org_tokens.append(vals[0])
                org_tags.append(vals[-1])
        assert len(org_tokens) == len(org_tags), print(len(org_tokens), len(org_tags))
        tokens, lang_ids = clean_sail2017_lines(org_tokens, org_tags)
        if " ".join(tokens).strip() == "":
            tokens, lang_ids = org_tokens, org_tags
        # if len(" ".join(tokens)) > MAX_CHAR_LEN:
        #     n_trimmed += 1
        #     newtokens, newlangids, currsum = [], [], 0
        #     for tkn, lgid in zip(tokens, lang_ids):  # 1 for space
        #         if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
        #             newtokens.append(tkn)
        #             newlangids.append(lgid)
        #             currsum += len(tkn) + 1
        #         else:
        #             break
        #     tokens, lang_ids = newtokens, newlangids
        example = Example(dataset="sail2017_hinglish",
                          task="classification",
                          split_type=mode,
                          uid=_id,
                          text=" ".join(org_tokens),
                          langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                            for lid in org_tags]),
                          text_pp=" ".join(tokens),
                          langids_pp=" ".join(lang_ids),
                          label=label)
        examples.append(example)
        progress_bar(len(examples), len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed} "
          f"and # of not found instances: {not_found}")
    return examples


def read_subwordlstm_downloads(path, mode="train"):
    st_time = time()
    to_translate = []
    label_dict = {"0": "negative", "1": "neutral", "2": "positive"}

    global FIELDS, MAX_CHAR_LEN
    add_fields = {
    }
    for k, v in add_fields.items():
        FIELDS += (vv for vv in v if vv not in FIELDS)

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    all_lines = open(path, "r").readlines()
    for uid, line in enumerate(all_lines):
        if line.strip() == "":
            continue
        txt, label = [part.strip() for part in line.strip().split("\t")][:2]
        label = label_dict[label]
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum+len(tkn)+1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn)+1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="subwordlstm_2016_hinglish",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          text_pp=new_txt,
                          label=label)
        examples.append(example)
        progress_bar(len(examples), len(all_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_hinglishpedia_downloads(path1, path2, mode, standardizing_tags={}):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    FIELDS += [fieldname for fieldname in ["tgt", ] if fieldname not in FIELDS]
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    txt_lines = [line.strip() for line in open(path1, "r")]
    tag_lines = [line.strip() for line in open(path2, "r")]

    for i, (txt, tags) in tqdm(enumerate(zip(txt_lines, tag_lines))):
        if not txt:
            continue
        txt = trn_hin2eng.transform(txt)
        example = Example(dataset="hinglishpedia",
                          task="classification",
                          split_type=mode,
                          uid=len(examples),
                          text=txt,
                          langids=" ".join([standardizing_tags[lid] if lid in standardizing_tags else "other"
                                            for lid in tags.split()]),
                          text_pp=txt
                          )
        examples.append(example)
        # progress_bar(len(examples), len(txt_lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")
    return examples


def read_kumaretal_2019_agg_downloads(path, mode, romanize=False):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    n_trimmed, n_romanized = 0, 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=False)
    for i, line in enumerate(lines):
        uid, txt, label = line[0], line[1], line[2]
        if not txt:
            continue
        if romanize:
            new_txt = trn_hin2eng.transform(txt)
            if txt != new_txt:
                n_romanized += 1
            txt = new_txt
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="kumaretal_2019_agg",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])

    if romanize:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed} "
              f"and # of romanized instances: {n_romanized}")
    else:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


def read_kumaretal_2020_agg_downloads(path, mode, test_labels_file=None, romanize=False):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    from indictrans import Transliterator
    trn_hin2eng = Transliterator(source='hin', target='eng')

    if mode == "test":
        test_label_lines = read_csv_file(test_labels_file, has_header=True)

    n_trimmed, n_romanized = 0, 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=True)
    for i, line in enumerate(lines):
        if mode == "test":
            uid, txt = line[0], line[1]
            assert test_label_lines[i][0] == uid
            label = test_label_lines[i][1]
        else:
            uid, txt, label = line[0], line[1], line[2]
        if not txt:
            continue
        if romanize:
            new_txt = trn_hin2eng.transform(txt)
            if txt != new_txt:
                n_romanized += 1
            txt = new_txt
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="kumaretal_2020_agg",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])

    if romanize:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed} "
              f"and # of romanized instances: {n_romanized}")
    else:
        print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


def read_vijayetal_2018_hatespeech_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=False, delimiter="\t")
    for i, line in enumerate(lines):
        uid, txt, label = len(examples), line[0].strip(), line[1].strip()
        if label == "n" or label == "on":
            label = "no"
        assert label in ["yes", "no"], print(label)
        if not txt:
            continue
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="vijayetal_2018_hatespeech",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])
    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


def read_kauretal_2019_reviews_downloads(path, mode):
    st_time = time()

    global FIELDS, MAX_CHAR_LEN

    label_def = {
        1: "Gratitude",
        2: "About the recipe",
        3: "About the video",
        4: "Praising",
        5: "Hybrid",
        6: "Undefined",
        7: "Suggestions and queries"
    }

    n_trimmed = 0
    Example = namedtuple(f"{mode}_example", FIELDS, defaults=(None,) * len(FIELDS))
    examples = []
    lines = read_csv_file(path, has_header=True)
    for i, line in enumerate(lines):
        uid, txt, label = line[0], line[1], label_def[int(line[2].strip())]
        if not txt:
            continue
        new_txt = clean_generic(txt)
        if new_txt.strip() == "":
            new_txt = txt
        if len(new_txt) > MAX_CHAR_LEN:
            n_trimmed += 1
            newtokens, currsum = [], 0
            for tkn in new_txt.split():  # 1 for space
                if currsum + len(tkn) + 1 <= MAX_CHAR_LEN:
                    newtokens.append(tkn)
                    currsum += len(tkn) + 1
                else:
                    break
            new_txt = " ".join(newtokens)
        example = Example(dataset="kauretal_2019_reviews",
                          task="classification",
                          split_type=mode,
                          uid=uid,
                          text=txt,
                          label=label,
                          text_pp=new_txt)
        examples.append(example)
        progress_bar(len(examples), len(lines), ["time"], [time() - st_time])

    print(f"len of {mode} data: {len(examples)} and # of trimmed instances: {n_trimmed}")

    return examples


if __name__ == "__main__":

    """ datasets with langids available """
    """ langids_pp also available """
    SENTIMIX2020 = False  # #processed  #msft_identified #tags_standardized
    SAIL2017_NEW = False  # #processed #tags_standardized
    """ langids_pp not available """
    ROYETAL2013_LID = False  # #processed  #tags_standardized
    GLUECOS_FG_POS = False   # #processed  #tags_standardized
    GLUECOS_UD_POS = False   # #processed  #tags_standardized
    GLUECOS_NER = False   # #processed # no tags are available for standardization
    LINCE_LID = False  # #processed  #msft_identified #tags_standardized
    LINCE_NER = False  # #processed #tags_standardized
    LINCE_POS = False  # #processed #tags_standardized
    HINGLISHPEDIA = False

    """ datasets with langids unavailable """
    SUBWORDLSTM2016 = False  # #processed  #msft_identified
    MT_1, MT_2 = False, False  # #processed
    VSINGH_NER = False  # #processed

    """ other datasets """
    SEMEVAL2017_EN_SA = False  # #processed
    IITP_PRODUCT_REVIEWS = False

    """ concatenate datasets """
    MLM_CONCAT_DATA_PROCESSED = False  # for CS-LM (Romanized)
    MLM_CONCAT_DATA = False  # for CS-LM (Romanized)
    LangInformed_MLM_CONCAT_DATA = False  # for CS-LM (Romanized)
    LID_CONCAT_DATA = False  # for LID tool building

    """ (NEW!) kumaretal_2019 for cs-agression """
    KUMARETAL_2019_AGG = False
    KUMARETAL_2020_AGG = False
    KAURETAL_2019_REVIEWS = False
    VIJAYETAL_2018_HATESPEECH = False

    ############################################
    ############################################
    ############################################

    """ sentimix 2020 hinglish """
    if SENTIMIX2020:
        MAX_CHAR_LEN = 300  # for text_pp and langids_pp
        src_path = "../downloads/cs-sa/sentimix2020/Hinglish"
        dest_path = "../datasets/sentimix2020/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train_14k_split_conll.txt"],
            "dev": ["dev_3k_split_conll.txt"],
            "test": ["Hindi_test_unalbelled_conll_updated.txt"]
        }
        test_labels = {line.strip().split(",")[0]: line.strip().split(",")[1]
                       for line in open(f"{src_path}/test_labels_hinglish.txt", "r")}
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_sentimix2020_downloads(os.path.join(src_path, filename), mode=key, test_labels=test_labels,
                                                  standardizing_tags={"Eng": "en", "Hin": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ sail 2017 hinglish """
    if SAIL2017_NEW:
        MAX_CHAR_LEN = 300  # for text_pp
        opfile = open("../downloads/cs-sa/SAIL_2017/Original Data/HI-EN.json", "r")
        loaded_data = json.load(opfile)
        json_labelled_data = {str(x["id"]): x for x in loaded_data}
        # count_different = 0
        # for x in loaded_data:
        #     try:
        #         txt = x["text"].strip()
        #     except AttributeError:
        #         print(x["text"])
        #     if txt in json_labelled_data:
        #         if not json_labelled_data[txt]["lang_tagged_text"] == x["lang_tagged_text"]:
        #             # print(json_labelled_data[x["text"]]["lang_tagged_text"])
        #             # print(x["lang_tagged_text"])
        #             # print()
        #             count_different += 1
        #         # print(json_labelled_data[x["text"]], x)
        #         continue
        #     else:
        #         json_labelled_data[txt] = x
        # print(count_different)
        # if not len(json_labelled_data) == len(loaded_data):
        #     print(len(json_labelled_data), len(loaded_data))
        opfile.close()
        src_path = "../downloads/cs-sa/SAIL_2017/Split IDs"
        dest_path = "../datasets/sail2017/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train_id.txt"],
            "dev": ["validation_id.txt"],
            "test": ["test_id.txt"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_sail2017_downloads_new(os.path.join(src_path, filename), json_labelled_data, mode=key,
                                                  standardizing_tags={"EN": "en", "HI": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ subword-lstm 2016 paper's hinglish """
    if SUBWORDLSTM2016:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-sa/Sub-word-LSTM/"
        dest_path = "../datasets/subwordlstm2016/Hinglish"
        dataset_files = {
            "train": ["IIITH_Codemixed_Processed.txt"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_subwordlstm_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        print("-------------------------------")

    """ lince lid hinglish """
    if LINCE_LID:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-benchmarks/lince/lid_hineng"
        dest_path = "../datasets/lince_lid/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train.conll"],
            "dev": ["dev.conll"],
            "test": ["test.conll"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_lince_downloads(os.path.join(src_path, filename), mode=key, dataset_name="lince_lid",
                                           task_name="seq_tagging", is_pos=False, is_ner=False,
                                           standardizing_tags={"lang1": "en", "lang2": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ lince ner hinglish """
    if LINCE_NER:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-benchmarks/lince/ner_hineng"
        dest_path = "../datasets/lince_ner/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train.conll"],
            "dev": ["dev.conll"],
            "test": ["test.conll"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_lince_downloads(os.path.join(src_path, filename), mode=key, dataset_name="lince_lid",
                                           task_name="seq_tagging", is_pos=False, is_ner=True,
                                           standardizing_tags={"en": "en", "hi": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ lince pos hinglish """
    if LINCE_POS:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-benchmarks/lince/pos_hineng"
        dest_path = "../datasets/lince_pos/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train.conll"],
            "dev": ["dev.conll"],
            "test": ["test.conll"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_lince_downloads(os.path.join(src_path, filename), mode=key, dataset_name="lince_lid",
                                           task_name="seq_tagging", is_pos=True, is_ner=False,
                                           standardizing_tags={"en": "en", "hi": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ mt hinglish (mrinaldhar et al. 2020) """
    if MT_1:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-mt/mrinaldhar/"
        dest_path = "../datasets/dharetal2018_mt/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": [("s-enhi.txt", "t-en.txt"), ]
        }
        data = {}
        for key, filenames_pair in dataset_files.items():
            data[f"{key}"] = []
            for pr in filenames_pair:
                exs = read_mt1_downloads(os.path.join(src_path, pr[0]), os.path.join(src_path, pr[1]),
                                         mode=key, dataset_name="dharetal2018_mt", task_name="seq2seq")
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        print("-------------------------------")

    """ mt hinglish (srivastava et al. 2020) """
    if MT_2:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-mt/srivastava/"
        dest_path = "../datasets/srivastavaetal2020_mt/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["English-Hindi code-mixed parallel corpus.csv"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_mt2_downloads(os.path.join(src_path, filename),
                                         mode=key, dataset_name="srivastavaetal2020_mt", task_name="seq2seq")
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        print("-------------------------------")

    """ LID (FIRE 2013 aka Roy et al. 2013) """
    if ROYETAL2013_LID:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-lid/royetal2013/"
        dest_path = "../datasets/royetal2013_lid/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "dev": ["dev.txt"],
            "test": ["test.txt"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_royetal2013_downloads(os.path.join(src_path, filename), mode=key,
                                                 standardizing_tags={"E": "en", "H": "hi", "en": "en", "hi": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ V-Singh et al. NER 2018 """
    if VSINGH_NER:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-ner/vsinghetal2018/"
        dest_path = "../datasets/vsinghetal2018_ner/Hinglish"
        dataset_files = {
            "train": ["annotatedData.csv"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_vsingh_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        print("-------------------------------")

    """ gluecos pos fg hinglish """
    if GLUECOS_FG_POS:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-benchmarks/GLUECoS/Data/Processed_Data/POS_EN_HI_FG/Romanized"
        dest_path = "../datasets/gluecos_pos_fg/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train.txt"],
            "dev": ["validation.txt"],
            "test": ["test.txt"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_gluecos_downloads(os.path.join(src_path, filename),
                                             mode=key, dataset_name="gluecos_pos_fg",
                                             task_name="seq_tagging", is_pos=True, is_ner=False,
                                             standardizing_tags={"en": "en", "hi": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ gluecos pos ud hinglish """
    if GLUECOS_UD_POS:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-benchmarks/GLUECoS/Data/Processed_Data/POS_EN_HI_UD/Romanized"
        dest_path = "../datasets/gluecos_pos_ud/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train.txt"],
            "dev": ["validation.txt"],
            "test": ["test.txt"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_gluecos_downloads(os.path.join(src_path, filename),
                                             mode=key, dataset_name="gluecos_pos_ud",
                                             task_name="seq_tagging", is_pos=True, is_ner=False,
                                             standardizing_tags={"EN": "en", "HI": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ gluecos ner hinglish """
    if GLUECOS_NER:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-benchmarks/GLUECoS/Data/Processed_Data/NER_EN_HI/Romanized"
        dest_path = "../datasets/gluecos_ner/Hinglish"
        dest_analysis_path = os.path.join(dest_path, "analysis")
        dataset_files = {
            "train": ["train.txt"],
            "dev": ["validation.txt"],
            "test": ["test.txt"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_gluecos_downloads(os.path.join(src_path, filename),
                                             mode=key, dataset_name="gluecos_ner",
                                             task_name="seq_tagging", is_pos=False, is_ner=True)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            create_path(dest_analysis_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_pp.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex.text_pp+"\n")
            opfile.close()
            dest_file_name = os.path.join(dest_analysis_path, f"{key}_tagged.txt")
            opfile = open(dest_file_name, "w")
            for ex in exs:
                if ex.langids is not None:
                    opfile.write(" ".join([f"{a}|||{b}" for a, b in zip(ex.text.split(" "),
                                                                        ex.langids.split(" "))])+"\n")
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ hinglishpedia (using indictrans) """
    if HINGLISHPEDIA:
        MAX_CHAR_LEN = 300  # for text_pp
        src_path = "../downloads/cs-hinglishpedia"
        dest_path = "../datasets/hinglishpedia/Hinglish"
        dataset_files = {
            "train": [("words/train.txt", "langs/train.txt"), ],
            "dev": [("words/valid.txt", "langs/valid.txt"), ],
            "test": [("words/test.txt", "langs/test.txt"), ]
        }
        data = {}
        for key, filenames_pair in dataset_files.items():
            data[f"{key}"] = []
            for pr in filenames_pair:
                exs = read_hinglishpedia_downloads(os.path.join(src_path, pr[0]), os.path.join(src_path, pr[1]),
                                                   mode=key, standardizing_tags={"0": "en", "1": "hi"})
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ semeval2017_en_sa """
    if SEMEVAL2017_EN_SA:
        MAX_CHAR_LEN = 300  # for text_pp and langids_pp
        src_path = "../downloads/en-sa/2017_English_final/GOLD/Subtask_A"
        dest_path = "../datasets/semeval2017_en_sa/English"
        dataset_files = {
            "train": ["twitter-2013train-A.txt", "twitter-2015train-A.txt", "twitter-2016train-A.txt"],
            "dev": ["twitter-2013dev-A.txt", "twitter-2016dev-A.txt", "twitter-2016devtest-A.txt"],
            "test": ["twitter-2013test-A.txt", "twitter-2014test-A.txt", "twitter-2015test-A.txt",
                     "twitter-2016test-A.txt"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_semeval2017_en_sa_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ iitp_product_reviews_hi_sa """
    if IITP_PRODUCT_REVIEWS:
        MAX_CHAR_LEN = 300  # for text_pp and langids_pp
        src_path = "../downloads/hi-sa/iitp-product-reviews/hi/"
        dest_path = "../datasets/iitp_product_reviews_hi_sa/Hinglish"
        dataset_files = {
            "train": ["hi-train.csv"],
            "dev": ["hi-valid.csv"],
            "test": ["hi-test.csv"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_iitp_product_reviews_hi_sa_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        print("-------------------------------")

    """ concatenate some datasets for training language models """
    if MLM_CONCAT_DATA_PROCESSED:
        MAX_LEN = 300
        concat_src = "../datasets"
        concat_dest = "../datasets/mlm_all_processed"
        concat_datasets = ["lince_lid/Hinglish", "lince_ner/Hinglish", "lince_pos/Hinglish",
                           "dharetal2018_mt/Hinglish", "srivastavaetal2020_mt/Hinglish",
                           "royetal2013_lid/Hinglish",
                           "sail2017/Hinglish", "sentimix2020/Hinglish", "subwordlstm2016/Hinglish",
                           "vsinghetal2018_ner/Hinglish",
                           "gluecos_pos_fg/Hinglish", "gluecos_pos_ud/Hinglish",
                           "hinglishpedia/Hinglish"]
        create_path(concat_dest)
        # train
        train_lines = []
        n_found_train, n_found_dev = 0, 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "train.jsonl" in filenames:
                n_found_train += 1
                train_examples = read_datasets_jsonl(os.path.join(check_dir, "train.jsonl"))
                train_lines.extend([ex.text_pp for ex in train_examples])
            if "dev.jsonl" in filenames:
                n_found_dev += 1
                dev_examples = read_datasets_jsonl(os.path.join(check_dir, "dev.jsonl"))
                train_lines.extend([ex.text_pp for ex in dev_examples])
        print("")
        print(f"train files found in {n_found_train} datasets from the given {len(concat_datasets)} datasets")
        print(f"dev files found in {n_found_dev} datasets from the given {len(concat_datasets)} datasets")
        print(f"total train lines obtained: {len(train_lines)}")
        opfile = open(os.path.join(concat_dest, "train.txt"), "w")
        c, cc = 0, 0
        for line in train_lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < MAX_LEN:
                opfile.write(f"{line}\n")
                c += 1
            else:
                curr_len, curr_tokens = 0, []
                tokens = [tkn+" " for tkn in line.split()]
                tokens[-1] = tokens[-1][:-1]
                for token in tokens:
                    if curr_len + len(token) > MAX_LEN:
                        sub_line = "".join(curr_tokens)
                        assert len(sub_line) <= MAX_LEN, print(len(sub_line), sub_line)
                        opfile.write(f"{sub_line}\n")
                        cc += 1
                        if len(token) > MAX_LEN:
                            break
                        curr_len, curr_tokens = 0, []
                    curr_tokens.append(token)
                    curr_len += len(token)
        opfile.close()
        print(f"total train lines written: {c}+{cc}={c+cc}")
        print("")
        # test
        test_lines = []
        n_found = 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "test.jsonl" in filenames:
                n_found += 1
                test_examples = read_datasets_jsonl(os.path.join(check_dir, "test.jsonl"))
                test_lines.extend([ex.text_pp for ex in test_examples])
        print("")
        print(f"test files found in {n_found} datasets from the given {len(concat_datasets)} datasets")
        print(f"total test lines obtained: {len(test_lines)}")
        opfile = open(os.path.join(concat_dest, "test.txt"), "w")
        c, cc = 0, 0
        for line in test_lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < MAX_LEN:
                opfile.write(f"{line}\n")
                c += 1
            # else:
            #     curr_len, curr_tokens = 0, []
            #     tokens = [tkn+" " for tkn in line.split()]
            #     tokens[-1] = tokens[-1][:-1]
            #     for token in tokens:
            #         if curr_len + len(token) > MAX_LEN:
            #             sub_line = "".join(curr_tokens)
            #             assert len(sub_line) <= MAX_LEN, print(len(sub_line), sub_line)
            #             opfile.write(f"{sub_line}\n")
            #             cc += 1
            #             if len(token) > MAX_LEN:
            #                 break
            #             curr_len, curr_tokens = 0, []
            #         curr_tokens.append(token)
            #         curr_len += len(token)
        opfile.close()
        print(f"total test lines written: {c}+{cc}={c+cc}")

    if MLM_CONCAT_DATA:
        MAX_LEN = 300
        concat_src = "../datasets"
        concat_dest = "../datasets/mlm_all_new"  # "../datasets/mlm_all"
        concat_datasets = ["lince_lid/Hinglish", "lince_ner/Hinglish", "lince_pos/Hinglish",
                           "dharetal2018_mt/Hinglish", "srivastavaetal2020_mt/Hinglish",
                           "royetal2013_lid/Hinglish",
                           "sail2017/Hinglish", "sentimix2020/Hinglish", "subwordlstm2016/Hinglish",
                           "vsinghetal2018_ner/Hinglish",
                           "gluecos_pos_fg/Hinglish", "gluecos_pos_ud/Hinglish",
                           "hinglishpedia/Hinglish",
                           # NEW!!
                           "kumaretal_2019_agg/Hinglish-R", "kumaretal_2020_agg/Hinglish-R",
                           "vijayetal_2018_hatespeech/Hinglish", "kauretal_2019_reviews/Hinglish"]
        create_path(concat_dest)
        # train
        train_lines = []
        n_found_train, n_found_dev = 0, 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "train.jsonl" in filenames:
                n_found_train += 1
                train_examples = read_datasets_jsonl(os.path.join(check_dir, "train.jsonl"))
                train_lines.extend([ex.text for ex in train_examples])
            if "dev.jsonl" in filenames:
                n_found_dev += 1
                dev_examples = read_datasets_jsonl(os.path.join(check_dir, "dev.jsonl"))
                train_lines.extend([ex.text for ex in dev_examples])
        print("")
        print(f"train files found in {n_found_train} datasets from the given {len(concat_datasets)} datasets")
        print(f"dev files found in {n_found_dev} datasets from the given {len(concat_datasets)} datasets")
        print(f"total train lines obtained: {len(train_lines)}")
        opfile = open(os.path.join(concat_dest, "train.txt"), "w")
        c, cc = 0, 0
        for line in train_lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < MAX_LEN:
                opfile.write(f"{line}\n")
                c += 1
            else:
                curr_len, curr_tokens = 0, []
                tokens = [tkn+" " for tkn in line.split()]
                tokens[-1] = tokens[-1][:-1]
                for token in tokens:
                    if curr_len + len(token) > MAX_LEN:
                        sub_line = "".join(curr_tokens)
                        assert len(sub_line) <= MAX_LEN, print(len(sub_line), sub_line)
                        opfile.write(f"{sub_line}\n")
                        cc += 1
                        if len(token) > MAX_LEN:
                            break
                        curr_len, curr_tokens = 0, []
                    curr_tokens.append(token)
                    curr_len += len(token)
        opfile.close()
        print(f"total train lines written: {c}+{cc}={c+cc}")
        print("")
        # test
        test_lines = []
        n_found = 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "test.jsonl" in filenames:
                n_found += 1
                test_examples = read_datasets_jsonl(os.path.join(check_dir, "test.jsonl"))
                test_lines.extend([ex.text for ex in test_examples])
        print("")
        print(f"test files found in {n_found} datasets from the given {len(concat_datasets)} datasets")
        print(f"total test lines obtained: {len(test_lines)}")
        opfile = open(os.path.join(concat_dest, "test.txt"), "w")
        c, cc = 0, 0
        for line in test_lines:
            line = line.strip()
            if not line:
                continue
            if len(line) < MAX_LEN:
                opfile.write(f"{line}\n")
                c += 1
            # else:
            #     curr_len, curr_tokens = 0, []
            #     tokens = [tkn+" " for tkn in line.split()]
            #     tokens[-1] = tokens[-1][:-1]
            #     for token in tokens:
            #         if curr_len + len(token) > MAX_LEN:
            #             sub_line = "".join(curr_tokens)
            #             assert len(sub_line) <= MAX_LEN, print(len(sub_line), sub_line)
            #             opfile.write(f"{sub_line}\n")
            #             cc += 1
            #             if len(token) > MAX_LEN:
            #                 break
            #             curr_len, curr_tokens = 0, []
            #         curr_tokens.append(token)
            #         curr_len += len(token)
        opfile.close()
        print(f"total test lines written: {c}+{cc}={c+cc}")

    if LangInformed_MLM_CONCAT_DATA:
        MAX_LEN = 200
        concat_src = "../datasets"
        concat_dest = "../datasets/li-mlm_all"
        concat_datasets = ["hinglishpedia/Hinglish",
                           "lince_lid/Hinglish", "lince_ner/Hinglish", "lince_pos/Hinglish",
                           "royetal2013_lid/Hinglish",
                           "sail2017/Hinglish", "sentimix2020/Hinglish",
                           "gluecos_pos_ud/Hinglish", "gluecos_pos_fg/Hinglish", ]
        create_path(concat_dest)
        # train
        train_lines = []
        n_found_train = 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "train.jsonl" in filenames:
                n_found_train += 1
                train_examples = read_datasets_jsonl(os.path.join(check_dir, "train.jsonl"))
                train_lines.extend([(ex.text, ex.langids) for ex in train_examples if ex.langids is not None])
                # for ex in train_examples:
                #     if ex.langids is None:
                #         print(ex)
                #         print(os.path.join(check_dir, "train.jsonl"))
                #         raise Exception
                for ex in train_examples:
                    if len(ex.text.split(" ")) != len(ex.langids.split(" ")):
                        print(len(ex.text.split(" ")), len(ex.langids.split(" ")))
                        print(ex)
                        print(os.path.join(check_dir, "train.jsonl"))
                        raise Exception
        print("")
        print(f"train files found in {n_found_train} datasets from the given {len(concat_datasets)} datasets")
        print(f"total train lines obtained: {len(train_lines)}")
        opfile = jsonlines.open(os.path.join(concat_dest, "train.jsonl"), "w")
        c, cc = 0, 0
        for line in train_lines:
            if len(line[0]) < MAX_LEN:
                try:
                    if not len(line[0].split(" ")) == len(line[1].split(" ")):
                        print(len(line[0].split(" ")), line[0])
                        print(len(line[1].split(" ")), line[1])
                        raise Exception
                except AttributeError:
                    print(line)
                    raise Exception
                opfile.write({"text": line[0], "langids": line[1]})
                c += 1
            else:
                curr_len, curr_tokens, curr_tags = 0, [], []
                for token, tag in zip(line[0].split(" "), line[1].split(" ")):
                    if curr_len + len(token) > MAX_LEN:
                        sub_line = " ".join(curr_tokens)
                        sub_tags = " ".join(curr_tags)
                        if not len(sub_line.split(" ")) == len(sub_tags.split(" ")):
                            print(len(line[0].split(" ")), len(sub_line.split(" ")), sub_line)
                            print(len(line[1].split(" ")), len(sub_tags.split(" ")), sub_tags)
                            raise Exception
                        opfile.write({"text": sub_line, "langids": sub_tags})
                        cc += 1
                        if len(token) > MAX_LEN:
                            break
                        curr_len, curr_tokens, curr_tags = 0, [], []
                    curr_tokens.append(token)
                    curr_tags.append(tag)
                    curr_len += len(token)
        opfile.close()
        print(f"total train lines written: {c}+{cc}={c+cc}")
        print("")

        # dev
        dev_lines = []
        n_found_dev = 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "dev.jsonl" in filenames:
                n_found_dev += 1
                dev_examples = read_datasets_jsonl(os.path.join(check_dir, "dev.jsonl"))
                dev_lines.extend([(ex.text, ex.langids) for ex in dev_examples if ex.langids is not None])
        print("")
        print(f"dev files found in {n_found_dev} datasets from the given {len(concat_datasets)} datasets")
        print(f"total dev lines obtained: {len(dev_lines)}")
        opfile = jsonlines.open(os.path.join(concat_dest, "test.jsonl"), "w")
        c, cc = 0, 0
        for line in dev_lines:
            if len(line[0]) < MAX_LEN:
                assert len(line[0].split(" ")) == len(line[1].split(" "))
                opfile.write({"text": line[0], "langids": line[1]})
                c += 1
            else:
                curr_len, curr_tokens, curr_tags = 0, [], []
                for token, tag in zip(line[0].split(" "), line[1].split(" ")):
                    if curr_len + len(token) > MAX_LEN:
                        sub_line = " ".join(curr_tokens)
                        sub_tags = " ".join(curr_tags)
                        assert len(sub_line.split(" ")) == len(sub_tags.split(" "))
                        opfile.write({"text": sub_line, "langids": sub_tags})
                        cc += 1
                        if len(token) > MAX_LEN:
                            break
                        curr_len, curr_tokens, curr_tags = 0, [], []
                    curr_tokens.append(token)
                    curr_tags.append(tag)
                    curr_len += len(token)
        opfile.close()
        print(f"total dev lines written: {c}+{cc}={c+cc}")
        print("")

    """ concatenate some datasets for training LID tool """
    if LID_CONCAT_DATA:
        MAX_LEN = 200
        concat_src = "../datasets"
        concat_dest = "../datasets/lid_all"
        concat_datasets = ["hinglishpedia/Hinglish",
                           "lince_lid/Hinglish", "lince_ner/Hinglish", "lince_pos/Hinglish",
                           "royetal2013_lid/Hinglish",
                           "sail2017/Hinglish", "sentimix2020/Hinglish",
                           "gluecos_pos_ud/Hinglish", "gluecos_pos_fg/Hinglish", ]
        create_path(concat_dest)
        # train
        train_lines = []
        n_found_train = 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "train.jsonl" in filenames:
                n_found_train += 1
                train_examples = read_datasets_jsonl(os.path.join(check_dir, "train.jsonl"))
                train_lines.extend([(ex.text, ex.langids) for ex in train_examples if ex.langids is not None])
                # for ex in train_examples:
                #     if ex.langids is None:
                #         print(ex)
                #         print(os.path.join(check_dir, "train.jsonl"))
                #         raise Exception
                for ex in train_examples:
                    if len(ex.text.split(" ")) != len(ex.langids.split(" ")):
                        print(len(ex.text.split(" ")), len(ex.langids.split(" ")))
                        print(ex)
                        print(os.path.join(check_dir, "train.jsonl"))
                        raise Exception
        print("")
        print(f"train files found in {n_found_train} datasets from the given {len(concat_datasets)} datasets")
        print(f"total train lines obtained: {len(train_lines)}")
        opfile = jsonlines.open(os.path.join(concat_dest, "train.jsonl"), "w")
        c, cc = 0, 0
        for line in train_lines:
            if len(line[0]) < MAX_LEN:
                try:
                    if not len(line[0].split(" ")) == len(line[1].split(" ")):
                        print(len(line[0].split(" ")), line[0])
                        print(len(line[1].split(" ")), line[1])
                        raise Exception
                except AttributeError:
                    print(line)
                    raise Exception
                opfile.write({"text": line[0], "langids": line[1]})
                c += 1
            else:
                curr_len, curr_tokens, curr_tags = 0, [], []
                for token, tag in zip(line[0].split(" "), line[1].split(" ")):
                    if curr_len + len(token) > MAX_LEN:
                        sub_line = " ".join(curr_tokens)
                        sub_tags = " ".join(curr_tags)
                        if not len(sub_line.split(" ")) == len(sub_tags.split(" ")):
                            print(len(line[0].split(" ")), len(sub_line.split(" ")), sub_line)
                            print(len(line[1].split(" ")), len(sub_tags.split(" ")), sub_tags)
                            raise Exception
                        opfile.write({"text": sub_line, "langids": sub_tags})
                        cc += 1
                        if len(token) > MAX_LEN:
                            break
                        curr_len, curr_tokens, curr_tags = 0, [], []
                    curr_tokens.append(token)
                    curr_tags.append(tag)
                    curr_len += len(token)
        opfile.close()
        print(f"total train lines written: {c}+{cc}={c+cc}")
        print("")

        # dev
        dev_lines = []
        n_found_dev = 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "dev.jsonl" in filenames:
                n_found_dev += 1
                dev_examples = read_datasets_jsonl(os.path.join(check_dir, "dev.jsonl"))
                dev_lines.extend([(ex.text, ex.langids) for ex in dev_examples if ex.langids is not None])
        print("")
        print(f"dev files found in {n_found_dev} datasets from the given {len(concat_datasets)} datasets")
        print(f"total dev lines obtained: {len(dev_lines)}")
        opfile = jsonlines.open(os.path.join(concat_dest, "dev.jsonl"), "w")
        c, cc = 0, 0
        for line in dev_lines:
            if len(line[0]) < MAX_LEN:
                assert len(line[0].split(" ")) == len(line[1].split(" "))
                opfile.write({"text": line[0], "langids": line[1]})
                c += 1
            else:
                curr_len, curr_tokens, curr_tags = 0, [], []
                for token, tag in zip(line[0].split(" "), line[1].split(" ")):
                    if curr_len + len(token) > MAX_LEN:
                        sub_line = " ".join(curr_tokens)
                        sub_tags = " ".join(curr_tags)
                        assert len(sub_line.split(" ")) == len(sub_tags.split(" "))
                        opfile.write({"text": sub_line, "langids": sub_tags})
                        cc += 1
                        if len(token) > MAX_LEN:
                            break
                        curr_len, curr_tokens, curr_tags = 0, [], []
                    curr_tokens.append(token)
                    curr_tags.append(tag)
                    curr_len += len(token)
        opfile.close()
        print(f"total dev lines written: {c}+{cc}={c+cc}")
        print("")

        # test
        test_lines = []
        n_found = 0
        for dataset in concat_datasets:
            check_dir = os.path.join(concat_src, dataset)
            filenames = os.listdir(check_dir)
            if "test.jsonl" in filenames:
                n_found += 1
                test_examples = read_datasets_jsonl(os.path.join(check_dir, "test.jsonl"))
                test_lines.extend([(ex.text, ) for ex in test_examples])
        print("")
        print(f"test files found in {n_found} datasets from the given {len(concat_datasets)} datasets")
        print(f"total test lines obtained: {len(test_lines)}")
        opfile = jsonlines.open(os.path.join(concat_dest, "test.jsonl"), "w")
        c, cc = 0, 0
        for line in test_lines:
            if len(line[0]) < MAX_LEN:
                opfile.write({"text": line[0]})
                c += 1
            else:
                curr_len, curr_tokens, curr_tags = 0, [], []
                for token in line[0].split(" "):
                    if curr_len + len(token) > MAX_LEN:
                        sub_line = " ".join(curr_tokens)
                        opfile.write({"text": sub_line})
                        cc += 1
                        if len(token) > MAX_LEN:
                            break
                        curr_len, curr_tokens, curr_tags = 0, [], []
                    curr_tokens.append(token)
                    curr_len += len(token)
        opfile.close()
        print(f"total test lines written: {c}+{cc}={c+cc}")

    """ agression """
    if KUMARETAL_2019_AGG:
        MAX_CHAR_LEN = float("inf")
        src_path = "../downloads/cs-aggression/trac1-dataset"

        # Hinglish
        dest_path = "../datasets/kumaretal_2019_agg/Hinglish"
        dataset_files = {
            "train": ["hindi/agr_hi_train.csv"],
            "dev": ["hindi/agr_hi_dev.csv"],
            "test": ["trac-gold-set/agr_hi_fb_gold.csv", "trac-gold-set/agr_hi_tw_gold.csv"],
            "test_fb": ["trac-gold-set/agr_hi_fb_gold.csv"],
            "test_tw": ["trac-gold-set/agr_hi_tw_gold.csv"],
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kumaretal_2019_agg_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        test_fb_examples = read_datasets_jsonl(f"{dest_path}/test_fb.jsonl", "test_fb")
        test_tw_examples = read_datasets_jsonl(f"{dest_path}/test_tw.jsonl", "test_tw")

        # Hinglish (Romanized)
        dest_path = "../datasets/kumaretal_2019_agg/Hinglish-R"
        dataset_files = {
            "train": ["hindi/agr_hi_train.csv"],
            "dev": ["hindi/agr_hi_dev.csv"],
            "test": ["trac-gold-set/agr_hi_fb_gold.csv", "trac-gold-set/agr_hi_tw_gold.csv"],
            "test_fb": ["trac-gold-set/agr_hi_fb_gold.csv"],
            "test_tw": ["trac-gold-set/agr_hi_tw_gold.csv"],
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kumaretal_2019_agg_downloads(os.path.join(src_path, filename), mode=key, romanize=True)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        test_fb_examples = read_datasets_jsonl(f"{dest_path}/test_fb.jsonl", "test_fb")
        test_tw_examples = read_datasets_jsonl(f"{dest_path}/test_tw.jsonl", "test_tw")

        # English
        dest_path = "../datasets/kumaretal_2019_agg/English"
        dataset_files = {
            "train": ["english/agr_en_train.csv"],
            "dev": ["english/agr_en_dev.csv"],
            "test": ["trac-gold-set/agr_en_fb_gold.csv", "trac-gold-set/agr_en_tw_gold.csv"],
            "test_fb": ["trac-gold-set/agr_en_fb_gold.csv"],
            "test_tw": ["trac-gold-set/agr_en_tw_gold.csv"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kumaretal_2019_agg_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        test_fb_examples = read_datasets_jsonl(f"{dest_path}/test_fb.jsonl", "test_fb")
        test_tw_examples = read_datasets_jsonl(f"{dest_path}/test_tw.jsonl", "test_tw")

        # English (Romanized, as and when required the conversion happens)
        dest_path = "../datasets/kumaretal_2019_agg/English-R"
        dataset_files = {
            "train": ["english/agr_en_train.csv"],
            "dev": ["english/agr_en_dev.csv"],
            "test": ["trac-gold-set/agr_en_fb_gold.csv", "trac-gold-set/agr_en_tw_gold.csv"],
            "test_fb": ["trac-gold-set/agr_en_fb_gold.csv"],
            "test_tw": ["trac-gold-set/agr_en_tw_gold.csv"]
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kumaretal_2019_agg_downloads(os.path.join(src_path, filename), mode=key, romanize=True)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
        test_fb_examples = read_datasets_jsonl(f"{dest_path}/test_fb.jsonl", "test_fb")
        test_tw_examples = read_datasets_jsonl(f"{dest_path}/test_tw.jsonl", "test_tw")
        print("-------------------------------")

    if KUMARETAL_2020_AGG:
        MAX_CHAR_LEN = float("inf")
        src_path = "../downloads/cs-aggression/trac2-dataset"

        # English
        dest_path = "../datasets/kumaretal_2020_agg/English"
        test_labels_file = os.path.join(src_path, "gold/trac2_eng_gold_a.csv")
        dataset_files = {
            "train": ["eng/trac2_eng_train.csv"],
            "dev": ["eng/trac2_eng_dev.csv"],
            "test": ["test/trac2_eng_test.csv"],
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kumaretal_2020_agg_downloads(os.path.join(src_path, filename), mode=key,
                                                        test_labels_file=test_labels_file, romanize=False)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")

        # Hinglish
        dest_path = "../datasets/kumaretal_2020_agg/Hinglish"
        test_labels_file = os.path.join(src_path, "gold/trac2_hin_gold_a.csv")
        dataset_files = {
            "train": ["hin/trac2_hin_train.csv"],
            "dev": ["hin/trac2_hin_dev.csv"],
            "test": ["test/trac2_hin_test.csv"],
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kumaretal_2020_agg_downloads(os.path.join(src_path, filename), mode=key,
                                                        test_labels_file=test_labels_file, romanize=False)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")

        # Hinglish-R
        dest_path = "../datasets/kumaretal_2020_agg/Hinglish-R"
        test_labels_file = os.path.join(src_path, "gold/trac2_hin_gold_a.csv")
        dataset_files = {
            "train": ["hin/trac2_hin_train.csv"],
            "dev": ["hin/trac2_hin_dev.csv"],
            "test": ["test/trac2_hin_test.csv"],
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kumaretal_2020_agg_downloads(os.path.join(src_path, filename), mode=key,
                                                        test_labels_file=test_labels_file, romanize=True)
                data[f"{key}"].extend(exs)
        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")

    """ kaur et al. 2019 youtube reviews classification"""
    if KAURETAL_2019_REVIEWS:
        MAX_CHAR_LEN = float("inf")
        src_path = "../downloads/cs-reviews/Cooking Data"

        # Hinglish
        dest_path = "../datasets/kauretal_2019_reviews/Hinglish"
        dataset_files = {
            "train": ["kabita_preprocessing.csv", "nisha_preprocessing.csv"],
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_kauretal_2019_reviews_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)

        examples = data["train"]
        split_sizes = [0.6, 0.2, 0.2]
        if split_sizes:
            assert sum(split_sizes) == 1.0
            n_examples = len(examples)
            split_sizes_int = []
            for val in split_sizes[:-1]:
                split_sizes_int.append(int(val * n_examples))
            split_sizes_int.append(n_examples - sum(split_sizes_int))
            random.seed(SEED)
            random.shuffle(examples)
            split_examples = []
            start = 0
            for num in split_sizes_int:
                split_examples.append(examples[start:start + num])
                start += num
            print(f"examples split sizes: {[len(x) for x in split_examples]}")
            assert sum([len(x) for x in split_examples]) == len(examples)
            data["train"] = split_examples[0]
            data["dev"] = split_examples[1]
            data["test"] = split_examples[2]

        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")

    """ vijay et al. 2018 hate speech """
    if VIJAYETAL_2018_HATESPEECH:
        MAX_CHAR_LEN = float("inf")
        src_path = "../downloads/cs-hatespeech/vijayetal2018"

        # Hinglish
        dest_path = "../datasets/vijayetal_2018_hatespeech/Hinglish"
        dataset_files = {
            "train": ["hate_speech.tsv"],
        }
        data = {}
        for key, filenames in dataset_files.items():
            data[f"{key}"] = []
            for filename in filenames:
                exs = read_vijayetal_2018_hatespeech_downloads(os.path.join(src_path, filename), mode=key)
                data[f"{key}"].extend(exs)

        examples = data["train"]
        split_sizes = [0.6, 0.2, 0.2]
        if split_sizes:
            assert sum(split_sizes) == 1.0
            n_examples = len(examples)
            split_sizes_int = []
            for val in split_sizes[:-1]:
                split_sizes_int.append(int(val * n_examples))
            split_sizes_int.append(n_examples - sum(split_sizes_int))
            random.seed(SEED)
            random.shuffle(examples)
            split_examples = []
            start = 0
            for num in split_sizes_int:
                split_examples.append(examples[start:start + num])
                start += num
            print(f"examples split sizes: {[len(x) for x in split_examples]}")
            assert sum([len(x) for x in split_examples]) == len(examples)
            data["train"] = split_examples[0]
            data["dev"] = split_examples[1]
            data["test"] = split_examples[2]

        for key, exs in data.items():
            create_path(dest_path)
            dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
            opfile = jsonlines.open(dest_file_name, "w")
            for ex in exs:
                opfile.write(ex._asdict())
            opfile.close()
        print("")
        train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
        test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")

    print("")

    # """ create vocab and plt freq """
    # from vocab import create_vocab
    # label_vocab = create_vocab([ex.label for ex in train_examples])
    # print(label_vocab.token2idx)
    # vocab = create_vocab([ex.text_pp for ex in train_examples])
    # print(vocab.token2idx)
    # print("")
    # import matplotlib.pyplot as plt
    # freq_values = [tpl[1] for tpl in vocab.token_freq]
    # plt.hist(freq_values, bins=20)
    # plt.grid()
    # plt.show()

    # """ sail 2017 hinglish """
    # if SAIL2017:
    #     MAX_CHAR_LEN = 300  # for text_pp
    #     src_path = "../downloads/cs-sa/SAIL_2017/Processed Data/Romanized"
    #     dest_path = "../datasets/sail2017/Hinglish"
    #     dataset_files = {
    #         "train": ["train.txt"],
    #         "dev": ["validation.txt"],
    #         "test": ["test.txt"]
    #     }
    #     data = {}
    #     for key, filenames in dataset_files.items():
    #         data[f"{key}"] = []
    #         for filename in filenames:
    #             exs = read_sail2017_downloads(os.path.join(src_path, filename),
    #                                       mode=key)
    #             data[f"{key}"].extend(exs)
    #     for key, exs in data.items():
    #         create_path(dest_path)
    #         dest_file_name = os.path.join(dest_path, f"{key}.jsonl")
    #         opfile = jsonlines.open(dest_file_name, "w")
    #         for ex in exs:
    #             opfile.write(ex._asdict())
    #         opfile.close()
    #     print("")
    #     train_examples = read_datasets_jsonl(f"{dest_path}/train.jsonl", "train")
    #     dev_examples = read_datasets_jsonl(f"{dest_path}/dev.jsonl", "dev")
    #     test_examples = read_datasets_jsonl(f"{dest_path}/test.jsonl", "test")
    #     print("-------------------------------")
