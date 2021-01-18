import json
from tqdm import tqdm
from typing import List
from collections import namedtuple

import torch
from torch.nn.utils.rnn import pad_sequence


def load_vocab(path) -> namedtuple:
    return_dict = json.load(open(path))
    # for idx2token, idx2chartoken, have to change keys from strings to ints
    #   https://stackoverflow.com/questions/45068797/how-to-convert-string-int-json-into-real-int-with-json-loads
    if "token2idx" in return_dict:
        return_dict.update({"idx2token": {v: k for k, v in return_dict["token2idx"].items()}})
    if "chartoken2idx" in return_dict:
        return_dict.update({"idx2chartoken": {v: k for k, v in return_dict["chartoken2idx"].items()}})

    # NEW
    # vocab: dict to named tuple
    vocab = namedtuple('vocab', sorted(return_dict))
    return vocab(**return_dict)


def create_vocab(data: List[str],
                 keep_simple=False,
                 min_max_freq: tuple = (1, float("inf")),
                 topk=None,
                 intersect: List = None,
                 load_char_tokens: bool = False,
                 is_label: bool = False,
                 labels_data_split_at_whitespace: bool = False) -> namedtuple:
    """
    :param data: list of sentences from which tokens are obtained as whitespace seperated
    :param keep_simple: retain tokens that have ascii and do not have digits (for preprocessing)
    :param min_max_freq: retain tokens whose count satisfies >min_freq and <max_freq
    :param topk: retain only topk tokens (specify either topk or min_max_freq)
    :param intersect: retain tokens that are at intersection with a custom token list
    :param load_char_tokens: if true, character tokens will also be loaded
    :param is_label: when the inouts are list of labels
    :return: a vocab namedtuple
    """

    if topk is None and (min_max_freq[0] > 1 or min_max_freq[1] < float("inf")):
        raise Exception("both min_max_freq and topk should not be provided at once !")

    # if is_label
    if is_label:

        def split_(txt: str):
            if labels_data_split_at_whitespace:
                return txt.split(" ")
            else:
                return [txt, ]

        # get all tokens
        token_freq, token2idx, idx2token = {}, {}, {}
        for example in tqdm(data):
            for token in split_(example):
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1
        print(f"Total tokens found: {len(token_freq)}")
        print(f"token_freq:\n{token_freq}\n")

        # create token2idx and idx2token
        for token in token_freq:
            idx = len(token2idx)
            idx2token[idx] = token
            token2idx[token] = idx

        token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        return_dict = {"token2idx": token2idx,
                       "idx2token": idx2token,
                       "token_freq": token_freq,
                       "n_tokens": len(token2idx),
                       "n_all_tokens": len(token2idx)}

    else:

        # get all tokens
        token_freq, token2idx, idx2token = {}, {}, {}
        for example in tqdm(data):
            for token in example.split(" "):
                if token not in token_freq:
                    token_freq[token] = 0
                token_freq[token] += 1
        print(f"Total tokens found: {len(token_freq)}")

        # retain only simple tokens
        if keep_simple:
            isascii = lambda s: len(s) == len(s.encode())
            hasdigits = lambda s: len([x for x in list(s) if x.isdigit()]) > 0
            tf = [(t, f) for t, f in [*token_freq.items()] if (isascii(t) and not hasdigits(t))]
            token_freq = {t: f for (t, f) in tf}
            print(f"After removing non-ascii and tokens with digits, total tokens retained: {len(token_freq)}")

        # retain only tokens with specified min and max range
        if min_max_freq[0] > 1 or min_max_freq[1] < float("inf"):
            sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
            tf = [(i[0], i[1]) for i in sorted_ if (min_max_freq[0] <= i[1] <= min_max_freq[1])]
            token_freq = {t: f for (t, f) in tf}
            print(f"After min_max_freq selection, total tokens retained: {len(token_freq)}")

        # retain only topk tokens
        if topk is not None:
            sorted_ = sorted(token_freq.items(), key=lambda item: item[1], reverse=True)
            token_freq = {t: f for (t, f) in list(sorted_)[:topk]}
            print(f"After topk selection, total tokens retained: {len(token_freq)}")

        # retain only interection of tokens
        if intersect is not None and len(intersect) > 0:
            tf = [(t, f) for t, f in [*token_freq.items()] if (t in intersect or t.lower() in intersect)]
            token_freq = {t: f for (t, f) in tf}
            print(f"After intersection, total tokens retained: {len(token_freq)}")

        # create token2idx and idx2token
        for token in token_freq:
            idx = len(token2idx)
            idx2token[idx] = token
            token2idx[token] = idx

        # add <<PAD>> special token
        ntokens = len(token2idx)
        pad_token = "<<PAD>>"
        token_freq.update({pad_token: -1})
        token2idx.update({pad_token: ntokens})
        idx2token.update({ntokens: pad_token})

        # add <<UNK>> special token
        ntokens = len(token2idx)
        unk_token = "<<UNK>>"
        token_freq.update({unk_token: -1})
        token2idx.update({unk_token: ntokens})
        idx2token.update({ntokens: unk_token})

        # new
        # add <<EOS>> special token
        ntokens = len(token2idx)
        eos_token = "<<EOS>>"
        token_freq.update({eos_token: -1})
        token2idx.update({eos_token: ntokens})
        idx2token.update({ntokens: eos_token})

        # new
        # add <<SOS>> special token
        ntokens = len(token2idx)
        sos_token = "<<SOS>>"
        token_freq.update({sos_token: -1})
        token2idx.update({sos_token: ntokens})
        idx2token.update({ntokens: sos_token})

        # return dict
        token_freq = list(sorted(token_freq.items(), key=lambda item: item[1], reverse=True))
        return_dict = {"token2idx": token2idx,
                       "idx2token": idx2token,
                       "token_freq": token_freq,
                       "pad_token": pad_token,
                       "pad_token_idx": token2idx[pad_token],
                       "unk_token": unk_token,
                       "unk_token_idx": token2idx[unk_token],
                       "eos_token": eos_token,
                       "eos_token_idx": token2idx[eos_token],
                       "sos_token": sos_token,
                       "sos_token_idx": token2idx[sos_token],
                       "n_tokens": len(token2idx) - 4,
                       "n_special_tokens": 4,
                       "n_all_tokens": len(token2idx)
                       }

        # load_char_tokens
        if load_char_tokens:
            print("loading character tokens as well")
            char_return_dict = create_char_vocab(use_default=True, data=data)
            return_dict.update(char_return_dict)

    # NEW
    # vocab: dict to named tuple
    vocab = namedtuple('vocab', sorted(return_dict))
    return vocab(**return_dict)


def create_char_vocab(use_default: bool, data=None) -> dict:
    if not use_default and data is None:
        raise Exception("data is None")

    # reset char token utils
    chartoken2idx, idx2chartoken = {}, {}
    char_unk_token, char_pad_token, char_start_token, char_end_token = \
        "<<CHAR_UNK>>", "<<CHAR_PAD>>", "<<CHAR_START>>", "<<CHAR_END>>"
    special_tokens = [char_unk_token, char_pad_token, char_start_token, char_end_token]
    for char in special_tokens:
        idx = len(chartoken2idx)
        chartoken2idx[char] = idx
        idx2chartoken[idx] = char

    if use_default:
        chars = list(
            """ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        for char in chars:
            if char not in chartoken2idx:
                idx = len(chartoken2idx)
                chartoken2idx[char] = idx
                idx2chartoken[idx] = char
    else:
        # helper funcs
        # isascii = lambda s: len(s) == len(s.encode())
        """
        # load batches of lines and obtain unique chars
        nlines = len(data)
        bsize = 5000
        nbatches = int( np.ceil(nlines/bsize) )
        for i in tqdm(range(nbatches)):
            blines = " ".join( [ex for ex in data[i*bsize:(i+1)*bsize]] )
            #bchars = set(list(blines))
            for char in bchars:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char
        """
        # realized the method above doesn't preserve order!!
        for line in tqdm(data):
            for char in line:
                if char not in chartoken2idx:
                    idx = len(chartoken2idx)
                    chartoken2idx[char] = idx
                    idx2chartoken[idx] = char

    print(f"number of unique chars found: {len(chartoken2idx)}")
    return_dict = {"chartoken2idx": chartoken2idx,
                   "idx2chartoken": idx2chartoken,
                   "char_unk_token": char_unk_token,
                   "char_pad_token": char_pad_token,
                   "char_start_token": char_start_token,
                   "char_end_token": char_end_token,
                   "char_unk_token_idx": chartoken2idx[char_unk_token],
                   "char_pad_token_idx": chartoken2idx[char_pad_token],
                   "char_start_token_idx": chartoken2idx[char_start_token],
                   "char_end_token_idx": chartoken2idx[char_end_token],
                   "n_tokens": len(chartoken2idx) - 4,
                   "n_special_tokens": 4}
    return return_dict


def char_tokenize(batch_sentences, vocab):
    """
    :returns List[pad_sequence], Tensor[int]
    """
    chartoken2idx = vocab.chartoken2idx
    char_unk_token = vocab.char_unk_token
    char_pad_token = vocab.char_pad_token
    char_start_token = vocab.char_start_token
    char_end_token = vocab.char_end_token

    func_word2charids = lambda word: [chartoken2idx[char_start_token]] + \
                                     [chartoken2idx[char] if char in chartoken2idx else chartoken2idx[char_unk_token]
                                      for char in list(word)] + \
                                     [chartoken2idx[char_end_token]]

    char_idxs = [[func_word2charids(word) for word in sent.split(" ")] for sent in batch_sentences]
    char_padding_idx = chartoken2idx[char_pad_token]
    tokenized_output = [pad_sequence(
        [torch.as_tensor(list_of_wordidxs).long() for list_of_wordidxs in list_of_lists],
        batch_first=True,
        padding_value=char_padding_idx
    )
        for list_of_lists in char_idxs]
    # dim [nsentences,nwords_per_sentence]
    nchars = [torch.as_tensor([len(wordlevel) for wordlevel in sentlevel]).long() for sentlevel in char_idxs]
    # dim [nsentences]
    nwords = torch.tensor([len(sentlevel) for sentlevel in tokenized_output]).long()
    return tokenized_output, nchars, nwords


def sclstm_tokenize(batch_sentences, vocab):
    """
    return List[pad_sequence], Tensor[int]
    """
    chartoken2idx = vocab.chartoken2idx
    char_unk_token_idx = vocab.char_unk_token_idx

    def sc_vector(word):
        a = [0]*len(chartoken2idx)
        if word[0] in chartoken2idx: a[ chartoken2idx[word[0]] ] = 1
        else: a[ char_unk_token_idx ] = 1
        b = [0]*len(chartoken2idx)
        for char in word[1:-1]:
            if char in chartoken2idx: b[ chartoken2idx[char] ] += 1
            #else: b[ char_unk_token_idx ] = 1
        c = [0]*len(chartoken2idx)
        if word[-1] in chartoken2idx: c[ chartoken2idx[word[-1]] ] = 1
        else: c[ char_unk_token_idx ] = 1
        return a+b+c

    # return list of tesnors and we don't need to pad these unlike cnn-lstm case!
    tensor_output =  [ torch.tensor([sc_vector(word) for word in sent.split(" ")]).float() for sent in batch_sentences]
    nwords = torch.tensor([len(sentlevel) for sentlevel in tensor_output]).long()
    return tensor_output, nwords
