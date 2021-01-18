import os
import time
import json
import torch
import random
import argparse
import numpy as np
from typing import Union, List
from pytorch_pretrained_bert import BertAdam
from sklearn.metrics import classification_report
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score

from datasets import read_datasets_jsonl, create_path, EXAMPLE
from vocab import create_vocab, load_vocab
from models import WholeWordBertForSeqClassificationAndTagging
from helpers import get_model_nparams, batch_iter, progress_bar

DEVICE = 'cuda' if torch.cuda.is_available() else "cpu"


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
    model = WholeWordBertForSeqClassificationAndTagging(
        sent_out_dim=2,  # Any random number because we don't care about classification loss
        lang_out_dim=label_vocab.n_tokens, 
        pretrained_path="xlm-roberta-base")
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


def get_predictions(input_sentences: Union[str, List[str]], label_vocab, model, DEV_BATCH_SIZE=1, verbose=False)->List[str]:

    assert DEV_BATCH_SIZE == 1, print("this constraint enforced due to the way the predict_lid() method is written")

    if isinstance(input_sentences, str):
        input_sentences = [input_sentences, ]

    test_examples = []
    for sent in input_sentences:
        sent = sent.strip()
        new_example = EXAMPLE(dataset=None, task=None, split_type=None, uid=None, text=sent, text_pp=None,
                              label=None, langids=None, seq_labels=None, langids_pp=None, meta_data=None)
        test_examples.append(new_example)

    test_exs, results = [], []
    selected_examples = test_examples
    n_batches = int(np.ceil(len(selected_examples) / DEV_BATCH_SIZE))
    selected_examples_batch_iter = batch_iter(selected_examples, DEV_BATCH_SIZE, shuffle=False)
    print(f"len of data: {len(selected_examples)}")
    print(f"n_batches of data: {n_batches}")
    for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
        st_time = time.time()

        batch_sentences = [getattr(ex, "text") for ex in batch_examples]
        output_dict = model.predict_lid(text_batch=batch_sentences)

        test_exs.extend(batch_examples)

        # update progress
        if verbose:
            progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])

        results_ = " ".join([label_vocab.idx2token[y] for y in output_dict["preds"]])
        if verbose:
            print(results_)
        results.append(results_)

    return results


def load_model_from_args(label_vocab, args):
    """ define and initialize model """

    model = WholeWordBertForSeqClassificationAndTagging(
        sent_out_dim=2,  # Any random number because we don't care about classification loss
        lang_out_dim=label_vocab.n_tokens, 
        pretrained_path=args.model_name)

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
            elif ".".join(k.split(".")[1:]) in pretrained_dict and \
                    v.shape == pretrained_dict[".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict[".".join(k.split(".")[1:])]
            elif "roberta." + ".".join(k.split(".")[1:]) in pretrained_dict and \
                    v.shape == pretrained_dict["roberta." + ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict["roberta." + ".".join(k.split(".")[1:])]
            elif "bert." + ".".join(k.split(".")[1:]) in pretrained_dict and \
                    v.shape == pretrained_dict["bert." + ".".join(k.split(".")[1:])].shape:
                used_dict[k] = pretrained_dict["bert." + ".".join(k.split(".")[1:])]
            elif "roberta." + k in pretrained_dict and v.shape == pretrained_dict["roberta." + k].shape:
                used_dict[k] = pretrained_dict["roberta." + k]
            elif "bert." + k in pretrained_dict and v.shape == pretrained_dict["bert." + k].shape:
                used_dict[k] = pretrained_dict["bert." + k]
            elif k.replace(".roberta.", ".") in pretrained_dict and v.shape == pretrained_dict[
                k.replace(".roberta.", ".")].shape:
                used_dict[k] = pretrained_dict[k.replace(".roberta.", ".")]
            elif k.replace(".bert.", ".") in pretrained_dict and v.shape == pretrained_dict[
                k.replace(".bert.", ".")].shape:
                used_dict[k] = pretrained_dict[k.replace(".bert.", ".")]
        unused_dict = {k: v for k, v in model_dict.items() if k not in used_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(used_dict)
        # 3. load the new state dict
        model.load_state_dict(model_dict)
        # 4. print unused_dict
        print("WARNING !!!")
        print(
            f"Following {len([*unused_dict.keys()])} keys are not updated from {args.custom_pretrained_path}/pytorch_model.bin")
        print(f"  →→ {[*unused_dict.keys()]}")

    return model


def get_optimizer():
    """ define optimizer """
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = int(len(train_examples) / TRAIN_BATCH_SIZE / GRADIENT_ACC * N_EPOCHS)
    lr = 2e-5  # 1e-4 or 2e-5 or 5e-5
    optimizer = BertAdam(optimizer_grouped_parameters, lr=lr, warmup=0.1, t_total=t_total)
    return optimizer


if __name__ == "__main__":

    """ args """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        # train -> to train and validate the specified model
        # dev, test -> specifies the data splits that are to be evaluated, requires --eval-ckpt-path specification
        # infer -> infers on the test set and saves results in --infer-results-path
        choices=["train", "dev", "test", "infer"],
        help="to train plus infer or only infer",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-multilingual-cased",
        choices=["xlm-roberta-base", "bert-base-cased", "bert-base-multilingual-cased"],
        help="name of the architecture you want to load with transformers lib; eg. xlm-roberta-base, bert-base-cased."
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=5,
        help="max number of epochs",
    )
    parser.add_argument(
        "--custom-pretrained-path",
        type=str,
        default="",
        help="specify the path from which a custom pretrained model needs to be loaded",
    )
    parser.add_argument(
        "--eval-ckpt-path",
        type=str,
        default="",
        help="path to load mode for testing/inference",
    )
    parser.add_argument(
        "--infer-results-path",
        type=str,
        default="",
        help="path to save predictions",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        help="path where to load train.txt, test.txt and dev.txt from",
    )
    parser.add_argument(
        "--text-type",
        type=str,
        help="type of text in each jsonl line - text or text_pp or src, etc.",
        default="text",
        required=True
    )
    parser.add_argument(
        "--tag-type",
        type=str,
        help="type of output - langids or postags etc",
        default="postags",
        required=True
    )
    parser.add_argument(
        "--use-seqeval",
        type=bool,
        help="Make true for using metrics from seqeval",
        default=False
    )
    parser.add_argument(
        "--cross-validate",
        type=bool,
        help="5 fold cross validation implemented",
        default=False
    )
    args = parser.parse_args()
    
    """ checks """
    args.dataset_name = args.dataset_name.strip("\\")
    
    """ settings """
    START_EPOCH, N_EPOCHS = 0, args.max_epochs
    TRAIN_BATCH_SIZE, DEV_BATCH_SIZE = (16, 16)
    GRADIENT_ACC = 2
    if args.eval_ckpt_path:
        assert args.mode in ["dev", "test", "infer"]
        CHECKPOINT_PATH = args.eval_ckpt_path
    else:
        assert args.mode in ["train"]
        CHECKPOINT_PATH = f"../checkpoints/{args.dataset_name}/{args.model_name}/"
    create_path(CHECKPOINT_PATH)
    print(CHECKPOINT_PATH)
    
    """ load dataset """
    train_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/train.jsonl", "train")
    dev_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/dev.jsonl", "dev")
    test_examples = read_datasets_jsonl(f"../datasets/{args.dataset_name}/test.jsonl", "test")
    
    """ obtain vocab """
    label_vocab, lid_label_vocab, pos_label_vocab, word_vocab = None, None, None, None
    label_vocab = create_vocab([getattr(ex, args.tag_type) for ex in train_examples],
                               is_label=True, labels_data_split_at_whitespace=True)
    print(f"label_vocab: {label_vocab}")
    
    """check dataset"""
    examples = None
    if args.mode == 'train':
        examples = train_examples + dev_examples
    elif args.mode == 'test':
        examples = test_examples
    elif args.mode == 'dev':
        examples = dev_examples
    for ex in examples:
        if len(getattr(ex, "text").split(" ")) != len(getattr(ex, args.tag_type).split(" ")):
            raise AssertionError
    
    if args.cross_validate:
        random.seed(11927)
        random.shuffle(train_examples)
        fold_len = int(len(train_examples)/5)
        folds = [train_examples[:fold_len], train_examples[fold_len:2*fold_len], train_examples[2*fold_len:3*fold_len],
                 train_examples[3*fold_len:4*fold_len], train_examples[4*fold_len:]]
        
    """ training and validation """
    if args.mode == "train":
        for fold_num in range(5 if args.cross_validate else 1):
            model = load_model_from_args(label_vocab, args)
            model.to(DEVICE)
            optimizer = get_optimizer()
            if args.cross_validate:
                train_examples = [item for num, sublist in enumerate(folds) if num != fold_num for item in sublist]
                dev_examples = folds[fold_num]
    
            best_dev_loss, best_dev_loss_epoch = 0., -1
            best_dev_acc, best_dev_acc_epoch = 0., -1
            for epoch_id in range(START_EPOCH, N_EPOCHS):
                print("################")
                print("epoch: ", epoch_id)
                if args.cross_validate:
                    print("fold: ", fold_num)
    
                """ training """
                train_loss, train_acc, train_preds = 0., -1, []
                n_batches = int(np.ceil(len(train_examples) / TRAIN_BATCH_SIZE))
                train_examples_batch_iter = batch_iter(train_examples, TRAIN_BATCH_SIZE, shuffle=True)
                print(f"len of train data: {len(train_examples)}")
                print(f"n_batches of train data: {n_batches}")
                optimizer.zero_grad()
                model.train()
                for batch_id, batch_examples in enumerate(train_examples_batch_iter):
                    st_time = time.time()
                    lid_targets = [[label_vocab.token2idx[token] for token in getattr(ex, args.tag_type).split(" ")] for ex
                                   in batch_examples]
                    batch_sentences = [getattr(ex, "text") for ex in batch_examples]
                    output_dict = model(text_batch=batch_sentences, sa_targets=None, lid_targets=lid_targets)
    
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
                        optimizer.step()
                        optimizer.zero_grad()
                    # accuracy (not on full dataset; only to understand trends while training)
                    if batch_id % 1000 == 0:
                        output_dict = model.predict_lid(text_batch=batch_sentences, targets=lid_targets)
                        train_acc = output_dict["acc"]
                    # update progress
                    progress_bar(batch_id + 1, n_batches, ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc"],
                                 [time.time() - st_time, batch_loss, train_loss / (batch_id + 1), train_acc])
                print("")
    
                """ validation """
                dev_loss, dev_acc, dev_preds, dev_true, dev_seqeval_preds, dev_seqeval_true = 0., 0., [], [], [], []
                n_batches = int(np.ceil(len(dev_examples) / DEV_BATCH_SIZE))
                dev_examples_batch_iter = batch_iter(dev_examples, DEV_BATCH_SIZE, shuffle=False)
                print(f"len of dev data: {len(dev_examples)}")
                print(f"n_batches of dev data: {n_batches}")
                for batch_id, batch_examples in enumerate(dev_examples_batch_iter):
                    st_time = time.time()
                    # forward
                    lid_targets = [[label_vocab.token2idx[token] for token in getattr(ex, args.tag_type).split(" ")] for ex
                                   in batch_examples]
                    batch_sentences = [getattr(ex, "text") for ex in batch_examples]
                    output_dict = model.predict_lid(text_batch=batch_sentences, targets=lid_targets)
                    batch_loss = output_dict["loss"].cpu().detach().numpy()
                    dev_loss += batch_loss
                    dev_acc += output_dict["acc_num"]
                    dev_preds.extend(output_dict["preds"])
                    dev_true.extend(output_dict["targets"])
                    if args.use_seqeval:
                        dev_seqeval_preds.append(output_dict["preds"])
                        dev_seqeval_true.append(output_dict["targets"])
                    # update progress
                    progress_bar(batch_id + 1, n_batches,
                                 ["batch_time", "batch_loss", "avg_batch_loss", "batch_acc", 'avg_batch_acc'],
                                 [time.time() - st_time, batch_loss, dev_loss / (batch_id + 1), output_dict["acc_num"],
                                  dev_acc / ((batch_id + 1)*DEV_BATCH_SIZE)])
                dev_acc /= len(dev_examples)  # exact
                dev_loss /= n_batches  # approximate
                print("\n Validation Complete")
                print(f"Validation avg_loss: {dev_loss:.4f} and acc: {dev_acc:.4f}")
                if not args.use_seqeval:
                    print(classification_report(dev_true, dev_preds, digits=4))
                else:
                    print(f"Precision : {precision_score(dev_seqeval_true, dev_seqeval_preds)}")
                    print(f"Recall : {recall_score(dev_seqeval_true, dev_seqeval_preds)}")
                    print(f"F1 : {f1_score(dev_seqeval_true, dev_seqeval_preds)}")
                    print(f"Accuracy : {accuracy_score(dev_seqeval_true, dev_seqeval_preds)}")
    
                """ model saving """
                # name = "model-epoch{}.pth.tar".format(epoch_id)
                if (START_EPOCH == 0 and epoch_id == START_EPOCH) or best_dev_acc < dev_acc:
                    # re-assign
                    best_dev_acc, best_dev_acc_epoch = dev_acc, epoch_id
                    # save
                    if args.cross_validate:
                        name = f"model{fold_num}.pth.tar"
                    else:
                        name = "model.pth.tar"
                    torch.save({
                        'epoch_id': epoch_id,
                        'max_dev_acc': best_dev_acc,
                        'argmax_dev_acc': best_dev_acc_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
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
    elif args.mode == "dev" or args.mode == "test":
        model = load_model_from_args(label_vocab, args)
        model.to(DEVICE)
        test_preds, test_true, test_seqeval_preds, test_seqeval_true = [], [], [], []
        print("in testing ...")
    
        for fold_num in range(5 if args.cross_validate else 1):
            if args.cross_validate:
                name = f"model{fold_num}.pth.tar"
            else:
                name = "model.pth.tar"
            """ testing on dev and test set """
            model.load_state_dict(torch.load(os.path.join(CHECKPOINT_PATH, name),
                                             map_location=torch.device(DEVICE))['model_state_dict'])
            print("################")
            if args.cross_validate:
                print("fold: ", fold_num)
    
            selected_examples = test_examples if args.mode == "test" else dev_examples
            n_batches = int(np.ceil(len(selected_examples) / DEV_BATCH_SIZE))
            selected_examples_batch_iter = batch_iter(selected_examples, DEV_BATCH_SIZE, shuffle=False)
            print(f"len of {args.mode} data: {len(selected_examples)}")
            print(f"n_batches of {args.mode} data: {n_batches}")
            for batch_id, batch_examples in enumerate(selected_examples_batch_iter):
                st_time = time.time()
                lid_targets = [[label_vocab.token2idx[token] for token in getattr(ex, args.tag_type).split(" ")] for ex
                               in batch_examples]
                batch_sentences = [getattr(ex, "text") for ex in batch_examples]
                output_dict = model.predict_lid(text_batch=batch_sentences, targets=lid_targets)
                test_preds.extend(output_dict["preds"])
                test_true.extend(output_dict["targets"])
                if args.use_seqeval:
                    test_seqeval_preds.append(output_dict["preds"])
                    test_seqeval_true.append(output_dict["targets"])
                # update progress
                progress_bar(batch_id + 1, n_batches, ["batch_time"], [time.time() - st_time])
        print("")
        if not args.use_seqeval:
            print(classification_report(test_true, test_preds, digits=4,
                                        labels=list(range(0, label_vocab.n_all_tokens)),
                                        target_names=[label_vocab.idx2token[idx] for idx in range(0, label_vocab.n_all_tokens)]))
        else:
            print(f"Precision : {precision_score(test_seqeval_true, test_seqeval_preds)}")
            print(f"Recall : {recall_score(test_seqeval_true, test_seqeval_preds)}")
            print(f"F1 : {f1_score(test_seqeval_true, test_seqeval_preds)}")
            print(f"Accuracy : {accuracy_score(test_seqeval_true, test_seqeval_preds)}")
    elif args.mode == "infer":
        # TODO: predict lables for test set and save them in --infer-results-path
        raise NotImplementedError