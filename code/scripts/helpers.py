import sys
import io
import torch
import numpy as np
from typing import List
from torch.nn.utils.rnn import pad_sequence
try:
    import fasttext
except ImportError:
    print("install fasttext if you plan to use models based on it")



def get_model_nparams(model):
    ntotal, n_gradrequired = 0, 0
    for param in list(model.parameters()):
        temp = 1
        for sz in list(param.size()):
            temp *= sz
        ntotal += temp
        if param.requires_grad:
            n_gradrequired += temp
    return ntotal, n_gradrequired


def train_validation_split(data, train_ratio, seed=11927):
    len_ = len(data)
    train_len_ = int(np.ceil(train_ratio*len_))
    inds_shuffled = np.arange(len_)
    np.random.seed(seed)
    np.random.shuffle(inds_shuffled)
    train_data = []
    for ind in inds_shuffled[:train_len_]:
        train_data.append(data[ind])
    validation_data = []
    for ind in inds_shuffled[train_len_:]:
        validation_data.append(data[ind])
    return train_data, validation_data


def batch_iter(data, batch_size, shuffle):
    """
    each data item is a tuple of labels and text
    """
    n_batches = int(np.ceil(len(data) / batch_size))
    indices = list(range(len(data)))
    if shuffle:
        np.random.shuffle(indices)

    for i in range(n_batches):
        batch_indices = indices[i * batch_size: (i + 1) * batch_size]
        yield [data[i] for i in batch_indices]


def progress_bar(value, endvalue, names=[], values=[], bar_length=15):
    assert(len(names) == len(values))
    percent = float(value) / endvalue
    arrow = '-' * int(round(percent * bar_length)-1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    string = ''
    for name, val in zip(names, values):
        temp = '|| {0}: {1:.4f} '.format(name, val) if val != None else '|| {0}: {1} '.format(name, None)
        string += temp
    sys.stdout.write("\rPercent: [{0}] {1}% {2}".format(arrow + spaces, int(round(percent * 100)), string))
    sys.stdout.flush()
    if value >= endvalue-1:
        print()
    return


class FastTextVecs(object):
    def __init__(self, langauge, dim=300, path=None):

        path = path if path is not None else f'../fasttext_models/cc.{langauge}.{dim}.bin'
        if path.endswith(".bin"):  # a model instead of word vectors
            self.ft = fasttext.load_model(path)
            self.word_vectors = None
            self.words = None
            self.ft_dim = self.ft.get_dimension()
        elif path.endswith(".vec"):
            self.ft = None
            self.word_vectors = self.load_vectors(path)
            self.words = [*self.word_vectors.keys()]
            self.ft_dim = len(self.word_vectors[self.words[0]])
        else:
            raise Exception(f"Invalid extension for the FASTTEXT_MODEL_PATH: {path}")
        print(f'fasttext model loaded from: {path} with dim: {self.ft_dim}')

    def get_dimension(self):
        return self.ft_dim

    def get_word_vector(self, word):
        if self.ft is not None:
            return self.ft.get_word_vector(word)
        try:
            word_vector = self.word_vectors[word]
        except KeyError:
            word_vector = np.array([0.0] * self.ft_dim)
        return word_vector

    def get_phrase_vector(self, phrases: List[str]):
        if isinstance(phrases, str):
            phrases = [phrases]
        assert isinstance(phrases, list) or isinstance(phrases, tuple), print(type(phrases))
        batch_array = np.array([np.mean([self.get_word_vector(word) for word in sentence.split()], axis=0)
                                if sentence else self.get_word_vector("")
                                for sentence in phrases])
        return batch_array

    def get_pad_vectors(self, batch_tokens: "list[list[tokens]]", token_pad_idx=0.0, return_lengths=False):
        assert isinstance(batch_tokens[0], list)
        tensors_list = [torch.tensor([self.get_word_vector(token) for token in line]) for line in batch_tokens]
        batch_vectors = pad_sequence(tensors_list, batch_first=True, padding_value=token_pad_idx)
        if return_lengths:
            batch_lengths = torch.tensor([len(x) for x in tensors_list]).long()
            return batch_vectors, batch_lengths
        return batch_vectors


    @staticmethod
    def load_vectors(fname):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = np.array([*map(float, tokens[1:])])
        return data