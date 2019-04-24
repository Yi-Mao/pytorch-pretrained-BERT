# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_gpt_summarization.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --save_root_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import csv
import datetime
import logging
import os
import random

import numpy as np
import torch
from path import Path
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from tqdm import tqdm, trange

from pytorch_pretrained_bert import (CONFIG_NAME, WEIGHTS_NAME, OpenAIAdam,
                                     OpenAIGPTDoubleHeadsModel,
                                     OpenAIGPTTokenizer, cached_path)

ROCSTORIES_URL = "https://s3.amazonaws.com/datasets.huggingface.co/ROCStories.tar.gz"


def setup_logger(log_file):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # create console handler with a higher log level
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(asctime)s %(levelname)s] %(message)s',
                                  '%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(sh)


def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)


def load_rocstories_dataset(dataset_path):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as f:
        f = csv.reader(f)
        output = []
        next(f)  # skip the first line
        for line in tqdm(f):
            output.append((' '.join(line[1:5]), line[5], line[6],
                           int(line[-1]) - 1))
    return output


def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token,
                         delimiter_token, clf_token):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, 2, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, 2), dtype=np.int64)
        lm_labels = np.full(
            (n_batch, 2, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch, ), dtype=np.int64)
        for i, (story, cont1, cont2, mc_label), in enumerate(dataset):
            with_cont1 = [start_token] + story[:cap_length] + [
                delimiter_token
            ] + cont1[:cap_length] + [clf_token]
            with_cont2 = [start_token] + story[:cap_length] + [
                delimiter_token
            ] + cont2[:cap_length] + [clf_token]
            input_ids[i, 0, :len(with_cont1)] = with_cont1
            input_ids[i, 1, :len(with_cont2)] = with_cont2
            mc_token_ids[i, 0] = len(with_cont1) - 1
            mc_token_ids[i, 1] = len(with_cont2) - 1
            lm_labels[i, 0, :len(with_cont1) - 1] = with_cont1[1:]
            lm_labels[i, 1, :len(with_cont2) - 1] = with_cont2[1:]
            mc_labels[i] = mc_label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name',
        type=str,
        help='Folder that contains pretrained model and config.')
    parser.add_argument(
        "--do_train", action='store_true', help="Whether to run fine tuning.")
    parser.add_argument(
        "--do_eval",
        action='store_true',
        help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--save_root_dir",
        type=str,
        required=True,
        help=
        "Root output directory where model predictions and checkpoints will be written."
    )
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.9)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir_prefix', type=str, default='run')
    parser.add_argument('--log_file', type=str, default='log.txt')

    args = parser.parse_args()
    if not args.do_train and not args.do_eval:
        raise ValueError(
            'At least one of `do_train` or `do_eval` must be True.')

    # Add datetime to save dir name.
    args.save_dir = os.path.join(
        args.save_root_dir, args.save_dir_prefix +
        datetime.datetime.now().strftime('_%Y%m%d_%H%M%S'))
    Path(args.save_dir).mkdir_p()

    setup_logger(os.path.join(args.save_dir, args.log_file))
    logger = logging.getLogger()
    logger.info('Arguments: {0}'.format(args))

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Use GPU if possible
    if torch.cuda.is_available():
        device = torch.device('cuda:' + str(args.gpu))
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device('cpu')

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ['_start_', '_delimiter_', '_classify_']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(
        args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(
        tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(
        args.model_name, num_special_tokens=len(special_tokens))
    model.to(device)

    # Load and encode the datasets
    if not args.train_dataset and not args.eval_dataset:
        roc_stories = cached_path(ROCSTORIES_URL)

    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    logger.info("Encoding dataset...")
    train_dataset = load_rocstories_dataset(args.train_dataset)
    eval_dataset = load_rocstories_dataset(args.eval_dataset)
    datasets = (train_dataset, eval_dataset)
    encoded_datasets = tokenize_and_encode(datasets)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions // 2 - 2
    input_length = max(len(story[:max_length]) + max(len(cont1[:max_length]), len(cont2[:max_length])) + 3  \
                           for dataset in encoded_datasets for story, cont1, cont2, _ in dataset)
    input_length = min(input_length, model.config.n_positions
                       )  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length,
                                           max_length, *special_tokens_ids)
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[
        0], tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(
        eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{
        'params':
        [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
        'weight_decay':
        0.01
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    num_train_optimization_steps = len(
        train_data) * args.num_train_epochs // args.train_batch_size
    optimizer = OpenAIAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mc_token_ids, lm_labels, mc_labels = batch
                losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                loss = args.lm_coef * losses[0] + losses[1]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item(
                ) if exp_average_loss is None else 0.7 * exp_average_loss + 0.3 * loss.item(
                )
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(
                    exp_average_loss, optimizer.get_lr()[0])

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(
            model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.save_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.save_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.save_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.save_dir)
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.save_dir)
        model.to(device)

    if args.do_eval:
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            with torch.no_grad():
                _, mc_loss = model(input_ids, mc_token_ids, lm_labels,
                                   mc_labels)
                _, mc_logits = model(input_ids, mc_token_ids)

            mc_logits = mc_logits.detach().cpu().numpy()
            mc_labels = mc_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

            eval_loss += mc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        train_loss = tr_loss / nb_tr_steps if args.do_train else None
        result = {
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'train_loss': train_loss
        }

        output_eval_file = os.path.join(args.save_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


if __name__ == '__main__':
    main()
