"""
notes
-----
list of models: https://huggingface.co/transformers/pretrained_models.html
"""

import os
import time
import json
import torch
import argparse
import datetime
import jsonlines
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
from pytorch_pretrained_bert import BertAdam
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from datasets import read_datasets_jsonl, create_path, EXAMPLE
from vocab import create_vocab, load_vocab, char_tokenize, sclstm_tokenize
from models import WholeWordBertMLP, WholeWordBertForSeqClassificationAndTagging, FusedBertMLP, SentenceBert
from models import SimpleMLP, CharLstmLstmMLP, ScLstmMLP
from models import WholeWordBertLstmMLP, WholeWordBertScLstmMLP, WholeWordBertCharLstmLstmMLP
from models import WholeWordBertXXXInformedMLP
from models import SentenceBertForSemanticSimilarity
from models import _custom_bert_tokenize
from helpers import get_model_nparams, batch_iter, progress_bar, FastTextVecs

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"


""" re-usable methods """


def load_required_vocab(CHECKPOINT_PATH):

    vocab_names = ["label_vocab", "lid_label_vocab", "pos_label_vocab", "word_vocab"]
    vocabs = {nm: None for nm in vocab_names}
    something_exists = False

    for nm in vocab_names:
        pth = os.path.join(CHECKPOINT_PATH, f"{nm}.json")
        if os.path.exists(pth):
            vocabs[nm] = load_vocab(pth)
            something_exists = True

    if not something_exists:
        raise Exception(f"no vocab files exist in the path specified: {CHECKPOINT_PATH}")

    return vocabs


def load_required_model(vocabs, CHECKPOINT_PATH):
    label_vocab = vocabs["label_vocab"]
    model = WholeWordBertMLP(out_dim=label_vocab.n_all_tokens, pretrained_path="xlm-roberta-base", finetune_bert=False)
    print(f"number of parameters (all, trainable) in your model: {get_model_nparams(model)}")
    model.to(DEVICE)
    print(f"in interactive inference mode...loading model.pth.tar from {CHECKPOINT_PATH}")
    model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"),
                                     map_location=torch.device(DEVICE))['model_state_dict'])
    return model


def load_required_vocab_and_model(CHECKPOINT_PATH):
    vocabs = load_required_vocab(CHECKPOINT_PATH)
    model = load_required_model(vocabs, CHECKPOINT_PATH)
    return vocabs, model


def get_predictions(input_sentences: Union[str, List[str]], label_vocab, model, DEV_BATCH_SIZE=8, verbose=False)->List[str]:

    if isinstance(input_sentences, str):
        input_sentences = [input_sentences, ]

    test_examples = []
    for sent in input_sentences:
        sent = sent.strip()
        new_example = EXAMPLE(dataset=None, task=None, split_type=None, uid=None, text=sent, text_pp=None,
                              label=None, langids=None, seq_labels=None, langids_pp=None, meta_data=None)
        test_examples.append(new_example)

    test_exs, test_preds, test_probs, test_true, targets = [], [], [], [], None
    selected_examples = test_examples
    n_batches = int(np.ceil(len(selected_examples) / DEV_BATCH_SIZE))
    selected_examples_batch_iter = batch_iter(selected_examples, DEV_BATCH_SIZE, shuffle=False)
    print(f"len of data: {len(selected_examples)}")
    print(f"n_batches of data: {n_batches}")
    for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
        st_time = time.time()
        # forward
        targets = [label_vocab.token2idx[ex.label] if ex.label is not None else ex.label for ex in batch_examples]
        targets = None if any([x is None for x in targets]) else targets

        batch_sentences = [getattr(ex, "text") for ex in batch_examples]
        output_dict = model.predict(text_batch=batch_sentences, targets=targets)

        test_exs.extend(batch_examples)
        test_preds.extend(output_dict["preds"])
        test_probs.extend(output_dict["probs"])
        if targets is not None:
            test_true.extend(targets)
        # update progress
        if verbose:
            progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])

    results = [label_vocab.idx2token[y] for y in test_preds]
    if verbose:
        print(results)

    return results


""" main """

if __name__ == "__main__":


    """ args """
    # import argparse
    parser = argparse.ArgumentParser(description="main.py")
    parser.add_argument(
        "--mode",
        type=str,
        # train -> to train and validate the specified model
        # dev, test -> specifies the data splits that are to be evaluated, requires --eval-ckpt-path specification
        choices=["train", "dev", "test", "train_test", "train_dev", "interactive", "train_dev_test",
                 "test_fb", "test_tw", "test.paws", "test.xnli", "test.csmt", "test.ted",
                 "dev_kannada", "dev_tamil", "dev_malayalam"
                 ],
        help="to train plus infer or only infer. `train` implicitly means `train` plus `dev`",
    )
    parser.add_argument(
        "--dataset-name",  # eg. "sentimix2020/Hinglish" or "sail2017/Hinglish", etc.
        type=str,
        default="",
        help="path where to load train.txt, test.txt and dev.txt from",
    )
    parser.add_argument(
        "--augment-train-datasets",  # eg. "sail2017/Hinglish,subwordlstm2016/Hinglish" or "semeval2017_en_sa/English", etc.
        type=str,
        default="",
        help="training data from these locations will be augmented; directories inputted as comma seperated",
    )
    parser.add_argument(
        "--text-type",
        type=str,
        default="",
        # ""-> code-switched, "trt"-> transliterated, "pp"->preprocessing
        # choices=["", "pp", "pp+trt", "msftlid", "+noisy_11927+noisy_2020",
        #          "trt", "hi", "en", "D", "non_english", "non_hindi", "non_english_D",
        #          "parul"],
        help="change this if you plan to train models with processed text; applies to augmented data too",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        # choices=["xlm-roberta-base", "bert-base-cased", "bert-base-multilingual-cased", other hgface models,
        #           "fasttext-vanilla", "fasttext-lstm", "charlstmlstm", "sclstm"
        #           "bert-lstm", "bert-fasttext-lstm", "bert-sc-lstm", "bert-charlstm-lstm", "bert-charlstm-lstm-v2",
        #           "li-bert-base-cased", "li-xlm-roberta-base"
        #           "posi-xlm-roberta-base",
        #           "bert-semantic-similarity"],
        help="name of the model architecture you want to load with transformers lib; eg. xlm-roberta-base, bert-base-cased."
             "Must contain the word bert or fasttext or lstm in it",
    )
    parser.add_argument(
        "--sentence-bert",
        help="if False, [CLS] is taken, else avg. sentence embeddings",
        action='store_true'  # if you pass --sentence-bert in cmd, this will be True and SentenceBert model is loaded
    )
    parser.add_argument(
        "--custom-pretrained-path",
        type=str,
        default="",
        help="specify the path from which a custom pretrained model needs to be loaded",
    )
    parser.add_argument(
        "--multitask-lid-sa",
        help="Triggers multitask learning of both language identification and sentiment analysis",
        action='store_true'  # if you pass --multitask-lid-sa in cmd, this will be True and MultiTask model is loaded
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="max number of epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="max number of epochs without any accuracy improvements before terminating training",
    )
    parser.add_argument(
        "--eval-ckpt-path",
        type=str,
        default="",
        help="path to load mode for testing/inference",
    )
    parser.add_argument(
        "--save-errors-path",
        type=str,
        default="",
        help="path to save errors when args.mode is dev or test; if empty, reults will be saved with temp-date-time",
    )
    # TODO: we don't require langids-type if text-type is specified
    parser.add_argument(
        "--langids-type",
        type=str,
        default="",
        help="change this if you plan to train models with different langids",
    )
    # TODO: Expects full text name. ex: --fusion-text-types text text_hi text_trt text_en
    parser.add_argument(
        "--fusion-text-types",
        nargs="+",
        help="All data types' representations we want to be fused"
    )
    parser.add_argument(
        "--fusion-strategy",
        type=str,
        default="concat",
        choices=["concat", "mean_pool", "max_pool"],
        help="method of combining fusion representations",
    )
    parser.add_argument(
        "--checkpoint-save-root-dir",
        type=str,
        default="../checkpoints",
        help="root dir where further directories are created to save checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="batch size for compiling batches when training models",
    )
    args = parser.parse_args()


    """ checks """
    args.dataset_name = args.dataset_name.strip("\\")
    if "train" not in args.mode and args.model_name is not None:
        assert "bert" in args.model_name or "fasttext" in args.model_name or "lstm" in args.model_name
    if args.model_name in ["fasttext-vanilla", "bert-semantic-similarity"]:
        print(f"dropping text_type as it is irrelevant with model_name=={args.model_name}")
        args.text_type = None
    if args.fusion_text_types:
        print("dropping text_type as it is irrelevant with fusion_text_types")
        args.text_type = ""
    if args.multitask_lid_sa and args.fusion_text_types:
        raise Exception
    if args.mode == "interactive":
        args.save_errors_path = None  # due to interactive mode
    args.langids_type = args.text_type
    print("dropping your inputted info about langids_type and setting it to same as text_type")


    """ settings """
    START_EPOCH, N_EPOCHS = 0, args.max_epochs
    if "train" in args.mode and START_EPOCH > 0:
        raise NotImplementedError(f"unable to continue training (NotImplemented; set START_EPOCH=0 for training). "
                                  f"Now performing inference...")
    if args.batch_size is None:
        TRAIN_BATCH_SIZE, DEV_BATCH_SIZE = (16, 16) if "bert" in args.model_name else (64, 64)
    else:
        assert args.batch_size > 0, print(f"args.batch_size: {args.batch_size} must be a positive integer")
        TRAIN_BATCH_SIZE, DEV_BATCH_SIZE = (args.batch_size, args.batch_size)
    GRADIENT_ACC = 2 if "bert" in args.model_name else 1
    if args.fusion_text_types:
        FUSION_N = len(args.fusion_text_types)
        TRAIN_BATCH_SIZE, DEV_BATCH_SIZE = int(TRAIN_BATCH_SIZE/FUSION_N), int(DEV_BATCH_SIZE/FUSION_N)
        GRADIENT_ACC *= FUSION_N
    if args.model_name == "bert-semantic-similarity":
        FUSION_N = 2
        TRAIN_BATCH_SIZE, DEV_BATCH_SIZE = int(TRAIN_BATCH_SIZE/FUSION_N), int(DEV_BATCH_SIZE/FUSION_N)
        GRADIENT_ACC *= FUSION_N
    if args.eval_ckpt_path:
        assert any([ii in args.mode for ii in ["dev", "test", "interactive"]])
        CHECKPOINT_PATH = args.eval_ckpt_path
    else:
        assert "train" in args.mode, print("--mode must contain `train` if no eval ckpt path is specified")
        CHECKPOINT_PATH = f"{args.checkpoint_save_root_dir}/{args.dataset_name}/{args.model_name}/"
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, f"text_raw") if args.text_type == "" \
            else os.path.join(CHECKPOINT_PATH, f"text_{args.text_type}")
        if os.path.exists(CHECKPOINT_PATH):
            subparts = [x for x in CHECKPOINT_PATH.split("/") if x]
            subparts[-1] = subparts[-1]+"--"+str(datetime.datetime.now()).replace(" ", "_")
            CHECKPOINT_PATH = "/".join(subparts)
        create_path(CHECKPOINT_PATH)
    print("****")
    print(CHECKPOINT_PATH)
    print("****")


    """ load dataset """
    if "train" in args.mode:
        train_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/train.jsonl", "train")
        dev_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/dev.jsonl", "dev")
        if args.text_type is not None and "+" in args.text_type:  # eg. "pp+trt", "+noisy"
            new_train_examples = []
            types = args.text_type.split("+")
            for ex in train_examples:
                for typ in types:
                    new_example = EXAMPLE(dataset=ex.dataset, task=ex.task, split_type=ex.split_type,
                                          uid=ex.uid+(f"_{typ}" if typ != "" else f"_raw"),
                                          text=getattr(ex, f"text_{typ}" if typ != "" else "text"),
                                          text_pp=ex.text_pp, label=ex.label, langids=ex.langids,
                                          seq_labels=None, langids_pp=None, meta_data=None)
                    new_train_examples.append(new_example)
            train_examples = new_train_examples
            print(f"train examples increased to {len(train_examples)} due to --text-type {args.text_type}")
            args.text_type = ""  # due to already inclusion of other data in training
        if args.augment_train_datasets:
            train_augment_examples = []
            dnames = [x.strip() for x in args.augment_train_datasets.split(",")]
            for name in dnames:
                temp_examples = read_datasets_jsonl(f"../datasets/{name}/train.jsonl", "train")
                train_augment_examples.extend(temp_examples)
            print(f"obtained {len(train_augment_examples)} additional train examples")
            train_examples.extend(train_augment_examples)

    if "dev" in args.mode:
        try:
            dev_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/{args.mode}.jsonl", f"dev")
        except FileNotFoundError:
            dev_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/dev.jsonl", f"dev")

    if "test" in args.mode:
        try:
            test_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/{args.mode}.jsonl", f"test")
        except FileNotFoundError:
            test_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/test.jsonl", f"test")


    """ check dataset """
    # TODO: Move this condition to dataset creation time
    if args.multitask_lid_sa:
        if args.mode == 'train':
            examples = train_examples + dev_examples
        elif args.mode == 'test':
            examples = test_examples
        for ex in examples:
            if len(getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text").split(" ")) != len(getattr(ex, f"text_{args.langids_type}" if args.langids_type != "" else "langids").split(" ")):
                raise AssertionError


    """ obtain vocab(s) """
    label_vocab, lid_label_vocab, pos_label_vocab, word_vocab = None, None, None, None
    if "train" in args.mode:
        label_vocab = create_vocab([ex.label for ex in train_examples], is_label=True)
        if args.multitask_lid_sa or args.model_name.startswith("li-"):
            lid_label_vocab = create_vocab([i for ex in train_examples for i in getattr(ex, f"langids_{args.langids_type}"
                                            if args.langids_type != "" else "langids").split(" ")], is_label=False)
        if args.model_name.startswith("posi-"):
            pos_label_vocab = create_vocab([i for ex in train_examples for i in getattr(ex, f"postags_{args.text_type}"
                                            if args.text_type != "" else "postags").split(" ")], is_label=False)
        if any([term in args.model_name for term in ["lstm", ]]):
            word_vocab = create_vocab(
                [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text") for ex in train_examples],
                is_label=False,
                load_char_tokens=True)
    else:
        label_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "label_vocab.json"))
        if args.multitask_lid_sa or args.model_name.startswith("li-"):
            lid_label_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"))
        if args.model_name.startswith("posi-"):
            pos_label_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"))
        if any([term in args.model_name for term in ["lstm", ]]):
            word_vocab = load_vocab(os.path.join(CHECKPOINT_PATH, "word_vocab.json"))

    # train_examples = train_examples[:8]
    # dev_examples = dev_examples[:6]
    # test_examples = test_examples[:10]


    """ define and initialize model """
    model = None
    if args.model_name == "bert-lstm":
        print("bert variant used in bert-lstm is xlm-roberta-base")
        model = WholeWordBertLstmMLP(out_dim=label_vocab.n_all_tokens, pretrained_path="xlm-roberta-base", finetune_bert=True)
    elif args.model_name == "bert-sc-lstm":
        print("bert variant used in bert-sc-lstm is xlm-roberta-base")
        model = WholeWordBertScLstmMLP(screp_dim=3*len(word_vocab.chartoken2idx), out_dim=label_vocab.n_all_tokens,
                                       pretrained_path="xlm-roberta-base", finetune_bert=True)
    elif args.model_name == "bert-charlstm-lstm":
        print("bert variant used in bert-charlstm-lstm is xlm-roberta-base")
        model = WholeWordBertCharLstmLstmMLP(nchars=len(word_vocab.chartoken2idx),
                                             char_emb_dim=128,
                                             char_padding_idx=word_vocab.char_pad_token_idx,
                                             out_dim=label_vocab.n_all_tokens,
                                             pretrained_path="xlm-roberta-base")
    elif args.model_name == "bert-charlstm-lstm-v2":
        print("bert variant used in bert-charlstm-lstm-v2 is xlm-roberta-base")
        assert os.path.exists(args.custom_pretrained_path)
        model = WholeWordBertCharLstmLstmMLP(nchars=len(word_vocab.chartoken2idx),
                                             char_emb_dim=128,
                                             char_padding_idx=word_vocab.char_pad_token_idx,
                                             out_dim=label_vocab.n_all_tokens,
                                             pretrained_path="xlm-roberta-base",
                                             freezable_pretrained_path=args.custom_pretrained_path,
                                             device=DEVICE)
        args.custom_pretrained_path = ""
    elif args.model_name == "bert-fasttext-lstm":
        # load pretrained
        fst_english = FastTextVecs("en")
        print("Loaded en fasttext model")
        print("bert variant used in bert-fasttext-lstm is xlm-roberta-base")
        model = WholeWordBertScLstmMLP(screp_dim=fst_english.ft_dim, out_dim=label_vocab.n_all_tokens,
                                       pretrained_path="xlm-roberta-base", finetune_bert=True)
    elif "bert" in args.model_name and args.model_name.startswith("li-"):
        assert os.path.exists(args.custom_pretrained_path)
        model = WholeWordBertXXXInformedMLP(out_dim=label_vocab.n_all_tokens,
                                            pretrained_path=args.custom_pretrained_path,
                                            n_lang_ids=lid_label_vocab.n_all_tokens,
                                            device=DEVICE,
                                            token_type_pad_idx=lid_label_vocab.pad_token_idx)
        args.custom_pretrained_path = ""
    elif "bert" in args.model_name and args.model_name.startswith("posi-"):
        assert os.path.exists(args.custom_pretrained_path)
        model = WholeWordBertXXXInformedMLP(out_dim=label_vocab.n_all_tokens, pretrained_path=args.custom_pretrained_path,
                                            n_lang_ids=pos_label_vocab.n_all_tokens, device=DEVICE,
                                            token_type_pad_idx=pos_label_vocab.pad_token_idx)
        args.custom_pretrained_path = ""
    elif args.model_name == "bert-semantic-similarity":
        print("bert variant used in bert-semantic-similarity is xlm-roberta-base")
        model = SentenceBertForSemanticSimilarity(out_dim=label_vocab.n_all_tokens, pretrained_path="xlm-roberta-base",
                                                  finetune_bert=True)
    elif "bert" in args.model_name:
        pretrained_path = args.model_name
        if args.multitask_lid_sa:
            model = WholeWordBertForSeqClassificationAndTagging(sent_out_dim=label_vocab.n_all_tokens,
                                                                lang_out_dim=lid_label_vocab.n_all_tokens,
                                                                pretrained_path=pretrained_path)
        elif args.fusion_text_types:
            model = FusedBertMLP(out_dim=label_vocab.n_all_tokens, pretrained_path=pretrained_path,
                                 finetune_bert=True, fusion_n=FUSION_N, fusion_strategy=args.fusion_strategy)
        elif args.sentence_bert:
            model = SentenceBert(out_dim=label_vocab.n_all_tokens, pretrained_path=pretrained_path,
                                 finetune_bert=True)
        else:
            model = WholeWordBertMLP(out_dim=label_vocab.n_all_tokens, pretrained_path=pretrained_path,
                                     finetune_bert=True)
    elif args.model_name == "fasttext-vanilla":
        # load pretrained
        fst_english = FastTextVecs("en")
        print("Loaded en fasttext model")
        fst_hindi = FastTextVecs("hi")
        print("Loaded hi fasttext model")
        # define model
        #   choose model based on if you want to pass en and hi token details seperately, or just only one of en or hi
        model = SimpleMLP(out_dim=label_vocab.n_all_tokens, input_dim1=fst_english.ft_dim)  # input_dim2=fst_hindi.ft_dim)
    elif args.model_name == "charlstmlstm":
        model = CharLstmLstmMLP(nchars=len(word_vocab.chartoken2idx),
                                char_emb_dim=128,
                                char_padding_idx=word_vocab.char_pad_token_idx,
                                padding_idx=word_vocab.pad_token_idx,
                                output_dim=label_vocab.n_all_tokens)
    elif args.model_name == "sclstm":
        model = ScLstmMLP(screp_dim=3*len(word_vocab.chartoken2idx),
                          padding_idx=word_vocab.pad_token_idx,
                          output_dim=label_vocab.n_all_tokens)
    elif args.model_name == "fasttext-lstm":
        # load pretrained
        fst_english = FastTextVecs("en")
        print("Loaded en fasttext model")
        model = ScLstmMLP(screp_dim=fst_english.ft_dim,
                          padding_idx=word_vocab.pad_token_idx,
                          output_dim=label_vocab.n_all_tokens)

    if "bert" in args.model_name and "train" in args.mode and args.custom_pretrained_path:
        print(f"\nLoading weights from args.custom_pretrained_path:{args.custom_pretrained_path}")
        pretrained_dict = torch.load(f"{args.custom_pretrained_path}/pytorch_model.bin",
                                     map_location=torch.device(DEVICE))
        model_dict = model.state_dict()
        # 1. filter out unnecessary keys
        used_dict = {}
        for k, v in model_dict.items():
            if "classifier.weight" in k or "classifier.bias" in k:
                print(k)
                continue
            if k in pretrained_dict and v.shape == pretrained_dict[k].shape:
                used_dict[k] = pretrained_dict[k]
            elif ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict[".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict[".".join(k.split(".")[1:])]
            elif "bert." + ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict["bert." + ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[1:])]
            elif "roberta." + ".".join(k.split(".")[1:]) in pretrained_dict and v.shape == pretrained_dict["roberta." + ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict["roberta." + ".".join(k.split(".")[1:])]
            elif "bert." + k in pretrained_dict and v.shape == pretrained_dict["bert." + k].shape:
                used_dict[k] = pretrained_dict["bert." + k]
            elif "roberta." + k in pretrained_dict and v.shape == pretrained_dict["roberta." + k].shape:
                used_dict[k] = pretrained_dict["roberta." + k]
        unused_dict = {k: v for k, v in model_dict.items() if k not in used_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(used_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. print unused_dict
        print("WARNING !!!")
        print(f"Following {len([*unused_dict.keys()])} keys are not updated from {args.custom_pretrained_path}/pytorch_model.bin")
        print(f"  →→ {[*unused_dict.keys()]}")

    print(f"number of parameters (all, trainable) in your model: {get_model_nparams(model)}")
    model.to(DEVICE)


    """ define optimizer """
    if "train" in args.mode:
        if "bert" in args.model_name and "lstm" in args.model_name:
            bert_model_params_names = ["bert_model."+x[0] for x in model.bert_model.named_parameters()]
            # others
            other_params = [param[1] for param in list(model.named_parameters()) if param[0] not in bert_model_params_names]
            print(f"{len(other_params)} number of params are being optimized with Adam")
            optimizer = torch.optim.Adam(other_params, lr=0.001)
            # bert
            bert_params = [param for param in list(model.named_parameters()) if param[0] in bert_model_params_names]
            param_optimizer = bert_params
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = int(len(train_examples) / TRAIN_BATCH_SIZE / GRADIENT_ACC * N_EPOCHS)
            lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
            bert_optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
            print(f"{len(bert_params)} number of params are being optimized with BertAdam")
        elif "bert" in args.model_name:
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
            t_total = int(len(train_examples) / TRAIN_BATCH_SIZE / GRADIENT_ACC * N_EPOCHS)
            lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
            optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


    """ training and validation """
    if "train" in args.mode:
        best_dev_loss, best_dev_loss_epoch = 0., -1
        best_dev_acc, best_dev_acc_epoch = 0., -1
        for epoch_id in range(START_EPOCH, N_EPOCHS):

            if epoch_id-best_dev_acc_epoch > args.patience:
                print(f"set patience of {args.patience} epochs reached; terminating train process")
                break

            print("\n\n################")
            print("epoch: ", epoch_id)

            """ training """
            train_loss, train_acc, train_preds = 0., -1, []
            n_batches = int(np.ceil(len(train_examples) / TRAIN_BATCH_SIZE))
            train_examples_batch_iter = batch_iter(train_examples, TRAIN_BATCH_SIZE, shuffle=True)
            print(f"len of train data: {len(train_examples)}")
            print(f"n_batches of train data: {n_batches}")
            model.zero_grad()
            model.train()
            for batch_id, batch_examples in enumerate(train_examples_batch_iter):
                st_time = time.time()
                # forward
                targets = [label_vocab.token2idx[ex.label] for ex in batch_examples]
                if args.model_name == "bert-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-sc-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_screps, _ = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                        batch_screps=batch_screps, targets=targets)
                elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                        batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
                                        batch_lengths=batch_lengths, targets=targets)
                elif args.model_name == "bert-fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                        batch_screps=batch_embs, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("li-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
                                      for ex in batch_examples]
                    # adding "other" at ends
                    batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
                                               [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
                                               [str(lid_label_vocab.eos_token_idx)])
                                      for lang_ids in batch_lang_ids]
                    output_dict = model(batch_sentences, batch_lang_ids, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("posi-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
                                     for ex in batch_examples]
                    # adding "other" at ends
                    batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
                                              [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
                                              [str(pos_label_vocab.eos_token_idx)])
                                     for pos_ids in batch_pos_ids]
                    output_dict = model(batch_sentences, batch_pos_ids, targets=targets)
                elif args.model_name == "bert-semantic-similarity":
                    batch_sentences = []
                    for text_type in ["src", "tgt"]:
                        batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                    output_dict = model(text_batch=batch_sentences, targets=targets)
                elif "bert" in args.model_name:
                    if args.fusion_text_types:
                        batch_sentences = []
                        for text_type in args.fusion_text_types:
                            batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                        output_dict = model(text_batch=batch_sentences, targets=targets)
                    elif args.multitask_lid_sa:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        lid_targets = [[lid_label_vocab.token2idx[token] for token in
                                        getattr(ex, f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(" ")]
                                       for ex in batch_examples]
                        output_dict = model(text_batch=batch_sentences, sa_targets=targets, lid_targets=lid_targets)
                    elif args.sentence_bert:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model(text_batch=batch_sentences, targets=targets)
                    else:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "fasttext-vanilla":
                    # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "),
                    #                                                        ex.langids_pp.split(" "))
                    #                            if tag.lower() != "hin"]) for ex in batch_examples]
                    # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                          if tag.lower() == "hin"]) for ex in batch_examples]
                    # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
                    # output_dict = model(input1=input_english, input2=input_hindi, targets=targets)
                    batch_english = [" ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
                                     for ex in batch_examples]
                    input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    output_dict = model(input1=input_english, targets=targets)
                elif args.model_name == "charlstmlstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
                elif args.model_name == "sclstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_screps, batch_lengths = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model(batch_screps, batch_lengths, targets=targets)
                elif args.model_name == "fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model(batch_embs, batch_lengths, targets=targets)
                loss = output_dict["loss"]
                batch_loss = loss.cpu().detach().numpy()
                train_loss += batch_loss
                # backward
                if GRADIENT_ACC > 1:
                    loss = loss / GRADIENT_ACC
                loss.backward()
                # optimizer step
                if (batch_id + 1) % GRADIENT_ACC == 0 or batch_id >= n_batches - 1:
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if args.model_name in ["bert-lstm", "bert-sc-lstm", "bert-charlstm-lstm"]:
                        bert_optimizer.step()
                    optimizer.step()
                    model.zero_grad()
                # update progress
                progress_bar(batch_id + 1, n_batches, ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc"],
                             [time.time() - st_time, batch_loss, train_loss / (batch_id + 1), train_acc])
                # break
            print("")

            """ validation """
            dev_loss, dev_acc, dev_preds, dev_true = 0., 0., [], []
            n_batches = int(np.ceil(len(dev_examples) / DEV_BATCH_SIZE))
            dev_examples_batch_iter = batch_iter(dev_examples, DEV_BATCH_SIZE, shuffle=False)
            print(f"len of dev data: {len(dev_examples)}")
            print(f"n_batches of dev data: {n_batches}")
            for batch_id, batch_examples in enumerate(dev_examples_batch_iter):
                st_time = time.time()
                # forward
                targets = [label_vocab.token2idx[ex.label] for ex in batch_examples]
                if args.model_name == "bert-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-sc-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_screps, _ = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_screps, targets=targets)
                elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
                                                batch_lengths=batch_lengths, targets=targets)
                elif args.model_name == "bert-fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_embs, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("li-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
                                      for ex in batch_examples]
                    # adding "other" at ends
                    batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
                                               [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
                                               [str(lid_label_vocab.eos_token_idx)])
                                      for lang_ids in batch_lang_ids]
                    output_dict = model.predict(batch_sentences, batch_lang_ids, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("posi-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
                                     for ex in batch_examples]
                    # adding "other" at ends
                    batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
                                              [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
                                              [str(pos_label_vocab.eos_token_idx)])
                                     for pos_ids in batch_pos_ids]
                    output_dict = model.predict(batch_sentences, batch_pos_ids, targets=targets)
                elif args.model_name == "bert-semantic-similarity":
                    batch_sentences = []
                    for text_type in ["src", "tgt"]:
                        batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif "bert" in args.model_name:
                    if args.fusion_text_types:
                        batch_sentences = []
                        for text_type in args.fusion_text_types:
                            batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    elif args.multitask_lid_sa:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        lid_targets = [[lid_label_vocab.token2idx[token] for token in
                                        getattr(ex, f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(" ")]
                                       for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, sa_targets=targets, lid_targets=lid_targets)
                    elif args.sentence_bert:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    else:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "fasttext-vanilla":
                    # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "),
                    #                                                        ex.langids_pp.split(" "))
                    #                            if tag.lower() != "hin"]) for ex in batch_examples]
                    # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                          if tag.lower() == "hin"]) for ex in batch_examples]
                    # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
                    # output_dict = model.predict(input1=input_english, input2=input_hindi, targets=targets)
                    batch_english = [" ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
                                     for ex in batch_examples]
                    input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    output_dict = model.predict(input1=input_english, targets=targets)
                elif args.model_name == "charlstmlstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
                elif args.model_name == "sclstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_screps, batch_lengths = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_screps, batch_lengths, targets=targets)
                elif args.model_name == "fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_embs, batch_lengths, targets=targets)
                batch_loss = output_dict["loss"].cpu().detach().numpy()
                dev_loss += batch_loss
                dev_acc += output_dict["acc_num"]
                dev_preds.extend(output_dict["preds"])
                dev_true.extend(targets)
                # update progress
                progress_bar(batch_id + 1, n_batches,
                             ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", 'avg_batch_acc'],
                             [time.time() - st_time, batch_loss, dev_loss / (batch_id + 1),
                              output_dict["acc_num"]/DEV_BATCH_SIZE, dev_acc / ((batch_id + 1)*DEV_BATCH_SIZE)])
                # break
            dev_acc /= len(dev_examples)  # exact
            dev_loss /= n_batches  # approximate
            print("\n Validation Complete")
            print(f"Validation avg_loss: {dev_loss:.4f} and acc: {dev_acc:.4f}")
            print(classification_report(dev_true, dev_preds, digits=4))

            """ model saving """
            # name = "model-epoch{}.pth.tar".format(epoch_id)
            name = "model.pth.tar"
            if (START_EPOCH == 0 and epoch_id == START_EPOCH) or best_dev_acc < dev_acc:
                best_dev_acc, best_dev_acc_epoch = dev_acc, epoch_id
                torch.save({
                    'epoch_id': epoch_id,
                    'max_dev_acc': best_dev_acc,
                    'argmax_dev_acc': best_dev_acc_epoch,
                    'model_state_dict': model.state_dict(),
                    # 'optimizer_state_dict': optimizer.state_dict()
                    },
                    os.path.join(CHECKPOINT_PATH, name))
                print("Model saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, name), epoch_id))
                if label_vocab is not None:
                    json.dump(label_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "label_vocab.json"), "w"), indent=4)
                    print("label_vocab saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "label_vocab.json"), epoch_id))
                if word_vocab is not None:
                    json.dump(word_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "word_vocab.json"), "w"), indent=4)
                    print("word_vocab saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "word_vocab.json"), epoch_id))
                    opfile = open(os.path.join(CHECKPOINT_PATH, "vocab.txt"), "w")
                    for word in word_vocab.token2idx.keys():
                        opfile.write(word+"\n")
                    opfile.close()
                    print("vocab words saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "vocab.txt"), epoch_id))
                    opfile = open(os.path.join(CHECKPOINT_PATH, "vocab_char.txt"), "w")
                    for word in word_vocab.chartoken2idx.keys():
                        opfile.write(word+"\n")
                    opfile.close()
                    print("vocab chars saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "vocab_char.txt"), epoch_id))
                if lid_label_vocab is not None:
                    json.dump(lid_label_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"), "w"), indent=4)
                    print("lid_label_vocab saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "lid_label_vocab.json"), epoch_id))
                if pos_label_vocab is not None:
                    json.dump(pos_label_vocab._asdict(), open(os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"), "w"), indent=4)
                    print("pos_label_vocab saved at {} in epoch {}".format(os.path.join(CHECKPOINT_PATH, "pos_label_vocab.json"), epoch_id))
            else:
                print("no improvements in results to save a checkpoint")
                print(f"checkpoint previously saved during epoch {best_dev_acc_epoch}(0-base) at: "
                      f"{os.path.join(CHECKPOINT_PATH, name)}")

        # if "test" in args.mode:
        #     args.mode = "test"

    """ testing """
    for mode_type in ["dev", "test"]:
        if mode_type in args.mode:
            """ doing inference on dev and/or test set """
            print("\n\n################")
            print(f"doing inference on {mode_type} set")
            print(f"in inference...loading model.pth.tar from {CHECKPOINT_PATH}")
            model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"),
                                             map_location=torch.device(DEVICE))['model_state_dict'])
            if not args.save_errors_path:
                args.save_errors_path = str(datetime.datetime.now()).replace(" ", "_")
            args.save_errors_path = os.path.join(CHECKPOINT_PATH, args.save_errors_path)
            test_exs, test_preds, test_probs, test_true, targets = [], [], [], [], None
            selected_examples = test_examples if mode_type == "test" else dev_examples
            n_batches = int(np.ceil(len(selected_examples) / DEV_BATCH_SIZE))
            selected_examples_batch_iter = batch_iter(selected_examples, DEV_BATCH_SIZE, shuffle=False)
            print(f"len of {mode_type} data: {len(selected_examples)}")
            print(f"n_batches of {mode_type} data: {n_batches}")
            for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
                st_time = time.time()
                # forward
                targets = [label_vocab.token2idx[ex.label] if ex.label is not None else ex.label for ex in batch_examples]
                targets = None if any([x is None for x in targets]) else targets
                if args.model_name == "bert-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-sc-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_screps, _ = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_screps, targets=targets)
                elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
                                                batch_lengths=batch_lengths, targets=targets)
                elif args.model_name == "bert-fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_embs, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("li-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
                                      for ex in batch_examples]
                    # adding "other" at ends
                    batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
                                               [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
                                               [str(lid_label_vocab.eos_token_idx)])
                                      for lang_ids in batch_lang_ids]
                    output_dict = model.predict(batch_sentences, batch_lang_ids, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("posi-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
                                     for ex in batch_examples]
                    # adding "other" at ends
                    batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
                                              [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
                                              [str(pos_label_vocab.eos_token_idx)])
                                     for pos_ids in batch_pos_ids]
                    output_dict = model.predict(batch_sentences, batch_pos_ids, targets=targets)
                elif args.model_name == "bert-semantic-similarity":
                    batch_sentences = []
                    for text_type in ["src", "tgt"]:
                        batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif "bert" in args.model_name:
                    if args.fusion_text_types:
                        batch_sentences = []
                        for text_type in args.fusion_text_types:
                            batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    elif args.multitask_lid_sa:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        lid_targets = [[lid_label_vocab.token2idx[token] for token in
                                        getattr(ex,
                                                f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(
                                            " ")]
                                       for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, sa_targets=targets, lid_targets=lid_targets)
                    elif args.sentence_bert:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    else:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "fasttext-vanilla":
                    # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                            if tag.lower() != "hin"]) for ex in batch_examples]
                    # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                          if tag.lower() == "hin"]) for ex in batch_examples]
                    # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
                    # output_dict = model.predict(input1=input_english, input2=input_hindi, targets=targets)
                    batch_english = [" ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
                                     for ex in batch_examples]
                    input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    output_dict = model.predict(input1=input_english, targets=targets)
                elif args.model_name == "charlstmlstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
                elif args.model_name == "sclstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_screps, batch_lengths = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_screps, batch_lengths, targets=targets)
                elif args.model_name == "fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_embs, batch_lengths, targets=targets)
                test_exs.extend(batch_examples)
                test_preds.extend(output_dict["preds"])
                test_probs.extend(output_dict["probs"])
                if targets is not None:
                    test_true.extend(targets)
                # update progress
                progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])
            print("")

            print(f"\n(NEW!) saving predictions in the folder: {args.save_errors_path}")
            create_path(args.save_errors_path)
            opfile = jsonlines.open(os.path.join(args.save_errors_path, "predictions.jsonl"), "w")
            for i, (x, y, z) in enumerate(zip(test_exs, test_preds, test_probs)):
                dt = x._asdict()
                dt.update({"prediction": label_vocab.idx2token[y]})
                dt.update({"pred_probs": z})
                opfile.write(dt)
            opfile.close()
            opfile = open(os.path.join(args.save_errors_path, "predictions.txt"), "w")
            for i, (x, y) in enumerate(zip(test_exs, test_preds)):
                try:
                    opfile.write(f"{label_vocab.idx2token[y]} ||| {x.text}\n")
                except AttributeError:
                    opfile.write(f"{label_vocab.idx2token[y]}\n")
            opfile.close()

            # if targets is not None and len(targets) > 0:
            #     print(f"\n(NEW!) saving errors files in the folder: {args.save_errors_path}")
            #     # report
            #     report = classification_report(test_true, test_preds, digits=4,
            #                                    target_names=[label_vocab.idx2token[idx]
            #                                                  for idx in range(0, label_vocab.n_all_tokens)])
            #     print("\n"+report)
            #     opfile = open(os.path.join(args.save_errors_path, "report.txt"), "w")
            #     opfile.write(report+"\n")
            #     opfile.close()
            #     # errors
            #     opfile = jsonlines.open(os.path.join(args.save_errors_path, "errors.jsonl"), "w")
            #     for i, (x, y, z) in enumerate(zip(test_exs, test_preds, test_true)):
            #         if y != z:
            #             dt = x._asdict()
            #             dt.update({"prediction": label_vocab.idx2token[y]})
            #             opfile.write(dt)
            #     opfile.close()
            #     for idx_i in label_vocab.idx2token:
            #         for idx_j in label_vocab.idx2token:
            #             opfile = jsonlines.open(os.path.join(args.save_errors_path,
            #                                                  f"errors_pred-{idx_i}_target-{idx_j}.jsonl"), "w")
            #             temp_test_exs = [x for x, y, z in zip(test_exs, test_preds, test_true)
            #                                       if (y == idx_i and z == idx_j)]
            #             for x in temp_test_exs:
            #                 dt = x._asdict()
            #                 dt.update({"prediction": label_vocab.idx2token[idx_i]})
            #                 opfile.write(dt)
            #             opfile.close()
            #     # confusion matrix
            #     cm = confusion_matrix(y_true=test_true, y_pred=test_preds, labels=list(set(test_true)))
            #     disp = ConfusionMatrixDisplay(confusion_matrix=cm,
            #                                   display_labels=[label_vocab.idx2token[ii] for ii in list(set(test_true))])
            #     disp = disp.plot(values_format="d")
            #     plt.savefig(os.path.join(args.save_errors_path, "confusion_matrix.png"))
            #     # plt.show()


    """ interactive """
    if args.mode == "interactive":
        print("\n\n################")
        print(f"in interactive inference mode...loading model.pth.tar from {CHECKPOINT_PATH}")
        model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, "model.pth.tar"),
                                         map_location=torch.device(DEVICE))['model_state_dict'])
        while True:
            text_input = input("enter your text here: ")
            if text_input == "-1":
                break
            new_example = EXAMPLE(dataset=None, task=None, split_type=None, uid=None, text=text_input, text_pp=None,
                                  label=None, langids=None, seq_labels=None, langids_pp=None, meta_data=None)
            test_examples = [new_example]

            # -----------> left unedited from previous
            test_exs, test_preds, test_probs, test_true, targets = [], [], [], [], None
            selected_examples = test_examples
            n_batches = int(np.ceil(len(selected_examples) / DEV_BATCH_SIZE))
            selected_examples_batch_iter = batch_iter(selected_examples, DEV_BATCH_SIZE, shuffle=False)
            print(f"len of {args.mode} data: {len(selected_examples)}")
            print(f"n_batches of {args.mode} data: {n_batches}")
            for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
                st_time = time.time()
                # forward
                targets = [label_vocab.token2idx[ex.label] if ex.label is not None else ex.label for ex in batch_examples]
                targets = None if any([x is None for x in targets]) else targets
                if args.model_name == "bert-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "bert-sc-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_screps, _ = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_screps, targets=targets)
                elif args.model_name in ["bert-charlstm-lstm", "bert-charlstm-lstm-v2"]:
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_idxs=batch_idxs, batch_char_lengths=batch_char_lengths,
                                                batch_lengths=batch_lengths, targets=targets)
                elif args.model_name == "bert-fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_sentences, batch_bert_dict, batch_splits = _custom_bert_tokenize(
                        batch_sentences, model.bert_tokenizer, max_len=200)
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_bert_dict=batch_bert_dict, batch_splits=batch_splits,
                                                batch_screps=batch_embs, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("li-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_lang_ids = [getattr(ex, f"langids_{args.text_type}" if args.text_type != "" else "langids")
                                      for ex in batch_examples]
                    # adding "other" at ends
                    batch_lang_ids = [" ".join([str(lid_label_vocab.sos_token_idx)] +
                                               [str(lid_label_vocab.token2idx[lang]) for lang in lang_ids.split(" ")] +
                                               [str(lid_label_vocab.eos_token_idx)])
                                      for lang_ids in batch_lang_ids]
                    output_dict = model.predict(batch_sentences, batch_lang_ids, targets=targets)
                elif "bert" in args.model_name and args.model_name.startswith("posi-"):
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_pos_ids = [getattr(ex, f"postags_{args.text_type}" if args.text_type != "" else "postags")
                                     for ex in batch_examples]
                    # adding "other" at ends
                    batch_pos_ids = [" ".join([str(pos_label_vocab.sos_token_idx)] +
                                              [str(pos_label_vocab.token2idx[lang]) for lang in pos_ids.split(" ")] +
                                              [str(pos_label_vocab.eos_token_idx)])
                                     for pos_ids in batch_pos_ids]
                    output_dict = model.predict(batch_sentences, batch_pos_ids, targets=targets)
                elif args.model_name == "bert-semantic-similarity":
                    batch_sentences = []
                    for text_type in ["src", "tgt"]:
                        batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                    output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif "bert" in args.model_name:
                    if args.fusion_text_types:
                        batch_sentences = []
                        for text_type in args.fusion_text_types:
                            batch_sentences.extend([getattr(ex, text_type) for ex in batch_examples])
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    elif args.multitask_lid_sa:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        lid_targets = [[lid_label_vocab.token2idx[token] for token in
                                        getattr(ex,
                                                f"langids_{args.langids_type}" if args.langids_type != "" else "langids").split(
                                            " ")]
                                       for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, sa_targets=targets, lid_targets=lid_targets)
                    elif args.sentence_bert:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                    else:
                        batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                           for ex in batch_examples]
                        output_dict = model.predict(text_batch=batch_sentences, targets=targets)
                elif args.model_name == "fasttext-vanilla":
                    # batch_english = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                            if tag.lower() != "hin"]) for ex in batch_examples]
                    # input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    # batch_hindi = [" ".join([token for token, tag in zip(ex.text_trt.split(" "), ex.langids_pp.split(" "))
                    #                          if tag.lower() == "hin"]) for ex in batch_examples]
                    # input_hindi = torch.tensor(fst_hindi.get_phrase_vector(batch_english))
                    # output_dict = model.predict(input1=input_english, input2=input_hindi, targets=targets)
                    batch_english = [" ".join([token for token, tag in zip(ex.text_pp.split(" "), ex.langids_pp.split(" "))])
                                     for ex in batch_examples]
                    input_english = torch.tensor(fst_english.get_phrase_vector(batch_english))
                    output_dict = model.predict(input1=input_english, targets=targets)
                elif args.model_name == "charlstmlstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_idxs, batch_char_lengths, batch_lengths = char_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_idxs, batch_char_lengths, batch_lengths, targets=targets)
                elif args.model_name == "sclstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_screps, batch_lengths = sclstm_tokenize(batch_sentences, word_vocab)
                    output_dict = model.predict(batch_screps, batch_lengths, targets=targets)
                elif args.model_name == "fasttext-lstm":
                    batch_sentences = [getattr(ex, f"text_{args.text_type}" if args.text_type != "" else "text")
                                       for ex in batch_examples]
                    batch_embs, batch_lengths = fst_english.get_pad_vectors(
                        batch_tokens=[line.split(" ") for line in batch_sentences],
                        return_lengths=True)
                    output_dict = model.predict(batch_embs, batch_lengths, targets=targets)
                test_exs.extend(batch_examples)
                test_preds.extend(output_dict["preds"])
                test_probs.extend(output_dict["probs"])
                if targets is not None:
                    test_true.extend(targets)
                # update progress
                # progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])
            # <-----------
            print([label_vocab.idx2token[y] for y in test_preds])


    print("complete")
