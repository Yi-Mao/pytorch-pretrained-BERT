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
"""
OpenAI GPT model fine-tuning for summarization.

Adapted from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_openai_gpt.py

CMD:
python run_gpt_summarization.py \
    --model_folder <model_folder> \
    --train_dataset <train_dataset> \
    --eval_dataset <eval_dataset> \
    --src_seq_length_trunc 400 \
    --tgt_seq_length_trunc 100 \
    --save_root_dir <save_root_dir> \
    --save_checkpoint_steps 1000 \
    --report_steps 50 \
    --train_steps 50000

train_dataset (and eval_dataset) was preprocessed by run_gpt_summarization.py.
"""
import argparse
import datetime
import logging
import os
import random

import numpy as np
import torch
from path import Path
from torch.nn.parallel.scatter_gather import gather
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from parallel import DataParallelCriterion, DataParallelModel
from pytorch_pretrained_bert import (CONFIG_NAME, WEIGHTS_NAME, OpenAIAdam,
                                     OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                                     cached_path)
from report import Statistics, accuracy

DELIMITER_TOKEN = '_delimiter_'
CLF_TOKEN = '_clf_'
MASK_VALUE = -1


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


def load_summarization_dataset(dataset_path, max_src_seq_length,
                               max_tgt_seq_length, src_seq_length_trunc,
                               tgt_seq_length_trunc, delimiter_token_id,
                               clf_token_id):
    """Read summarization dataset preprocessed by preprocess_cnndm.py

    Original document and its summary are separated by special delimiter token.
    """
    output = None
    if dataset_path:
        original = torch.load(dataset_path)

        original = [
            t for t in original
            if len(t[0]) <= max_src_seq_length
            and len(t[1]) <= max_tgt_seq_length
        ]

        # Get sequence length.
        src_seq_length = min(
            max(len(t[0]) for t in original), src_seq_length_trunc)
        tgt_seq_length = min(
            max(len(t[1]) for t in original), tgt_seq_length_trunc)
        seq_length = src_seq_length + tgt_seq_length + 2

        output = np.full(
            (len(original), seq_length), fill_value=MASK_VALUE, dtype=np.int64)
        for i, (src, tgt) in enumerate(original):
            seq = src[:src_seq_length] + [
                delimiter_token_id
            ] + tgt[:tgt_seq_length] + [clf_token_id]
            output[i, :len(seq)] = seq

        output = torch.from_numpy(output)
    return output


def save_checkpoint(save_dir, model, tokenizer):
    """Save checkpoint."""
    Path(save_dir).mkdir_p()
    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(save_dir, CONFIG_NAME)
    model_to_save = model.module if hasattr(
        model, 'module') else model  # Only save the model itself
    torch.save(model_to_save.state_dict(), output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(save_dir)


class LMLoss(torch.nn.CrossEntropyLoss):
    """Language modeling loss."""

    def __init__(self,
                 weight=None,
                 size_average=None,
                 ignore_index=-100,
                 reduce=None,
                 reduction='mean'):
        super(LMLoss, self).__init__(weight, size_average, ignore_index,
                                     reduce, reduction)

    def forward(self, *inputs):
        logits, labels = tuple(inputs)
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss = super(LMLoss, self).forward(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1))
        return loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_folder',
        type=str,
        help='Folder that contains pretrained model and config.')
    parser.add_argument(
        '--train_dataset',
        type=str,
        required=True,
        help='Tokenizer is in the same folder as train_dataset.')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument(
        '--max_src_seq_length',
        type=int,
        default=10000,
        help='Maximum source sequence length.')
    parser.add_argument(
        '--max_tgt_seq_length',
        type=int,
        default=10000,
        help='Maximum target sequence length.')
    parser.add_argument(
        '--src_seq_length_trunc',
        type=int,
        default=100,
        help='Truncate source sequence length.')
    parser.add_argument(
        '--tgt_seq_length_trunc',
        type=int,
        default=20,
        help='Truncate target sequence length.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_steps', type=int, default=100000)
    parser.add_argument(
        '--report_steps',
        type=int,
        default=50,
        help='Print stats every X steps')
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument(
        "--save_root_dir",
        type=str,
        required=True,
        help=
        "Root experiment directory where model predictions and checkpoints will be written to."
    )
    parser.add_argument('--save_dir_prefix', type=str, default='run')
    parser.add_argument('--log_file', type=str, default='log.txt')
    parser.add_argument('--save_checkpoint_steps', type=int, default=5000)
    args = parser.parse_args()

    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # Add datetime to save dir name.
    args.save_dir = os.path.join(
        args.save_root_dir, args.save_dir_prefix +
        datetime.datetime.now().strftime('_%Y%m%d_%H%M%S'))
    Path(args.save_dir).mkdir_p()

    # Set up logger.
    setup_logger(os.path.join(args.save_dir, args.log_file))
    logger = logging.getLogger()
    logger.info('Arguments: {0}'.format(args))

    # Use GPU if possible
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = False
    if torch.cuda.is_available():
        use_cuda = True
        torch.cuda.manual_seed_all(args.seed)

    # Load tokenizer and model
    tokenizer_dir = os.path.dirname(args.train_dataset)
    tokenizer = OpenAIGPTTokenizer.from_pretrained(tokenizer_dir)
    assert DELIMITER_TOKEN in tokenizer.special_tokens
    assert CLF_TOKEN in tokenizer.special_tokens
    model = OpenAIGPTLMHeadModel.from_pretrained(
        args.model_folder, num_special_tokens=len(tokenizer.special_tokens))
    criterion = LMLoss(ignore_index=MASK_VALUE, reduction='sum')
    if use_cuda:
        model = DataParallelModel(model).cuda()
        # Losses from multiple GPUs are summed. Need to apply appropriate normalization.
        criterion = DataParallelCriterion(criterion).cuda()

    n_positions = model.module.config.n_positions if use_cuda else model.config.n_positions
    if args.src_seq_length_trunc + args.tgt_seq_length_trunc + 2 > n_positions:
        raise ValueError('Exceeds maximum allowed sequence length.')

    logger.info('Encoding dataset...')
    train_dataset = load_summarization_dataset(
        args.train_dataset, args.max_src_seq_length, args.max_tgt_seq_length,
        args.src_seq_length_trunc, args.tgt_seq_length_trunc,
        tokenizer.special_tokens[DELIMITER_TOKEN],
        tokenizer.special_tokens[CLF_TOKEN])
    eval_dataset = load_summarization_dataset(
        args.eval_dataset, args.max_src_seq_length, args.max_tgt_seq_length,
        args.src_seq_length_trunc, args.tgt_seq_length_trunc,
        tokenizer.special_tokens[DELIMITER_TOKEN],
        tokenizer.special_tokens[CLF_TOKEN])
    eval_dataset = eval_dataset[:48, :]

    train_data = TensorDataset(train_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(
        train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_dataloader = None
    if eval_dataset is not None:
        eval_data = TensorDataset(eval_dataset)
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
        args.weight_decay
    }, {
        'params':
        [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
        'weight_decay':
        0.0
    }]
    train_epochs = max(args.train_steps //
                       (len(train_data) // args.train_batch_size), 1)
    optimizer = OpenAIAdam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        warmup=args.warmup_proportion,
        max_grad_norm=args.max_grad_norm,
        weight_decay=args.weight_decay,
        t_total=args.train_steps)

    # Multi-GPU Training
    model.train()
    tr_steps = 0
    tr_loss = Statistics()
    for _ in range(train_epochs):
        for _, data_batch in enumerate(train_dataloader):
            data_batch = data_batch[0].cuda()
            n_elements = torch.sum(data_batch[..., 1:] != MASK_VALUE).item()

            predictions = model(data_batch)
            loss = criterion(predictions, data_batch) / n_elements

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tr_steps += 1
            tr_loss.update(loss.item(), 0, n_elements)

            if tr_steps % args.report_steps == 0:
                logger.info('step {:6d}; loss {:8.4f}; lr: {:8.2e}'.format(
                    tr_steps, tr_loss.loss, optimizer.get_lr()[0]))

            if tr_steps % args.save_checkpoint_steps == 0:
                save_checkpoint(
                    os.path.join(args.save_dir, str(tr_steps)), model,
                    tokenizer)

                # Validation
                if eval_dataloader is not None:
                    model.eval()
                    valid_loss = Statistics()
                    for data_batch in eval_dataloader:
                        data_batch = data_batch[0].cuda()

                        with torch.no_grad():
                            predictions = model(data_batch)

                            token_accuracy, n_elements = accuracy(
                                torch.argmax(
                                    gather(predictions, 0, dim=0)[..., :-1, :],
                                    -1), data_batch[..., 1:], MASK_VALUE)
                            loss = criterion(predictions,
                                             data_batch) / n_elements
                            valid_loss.update(loss.item(), token_accuracy,
                                              n_elements)
                    logger.info(
                        'Eval at step {:6d}; loss {:8.4f};  accuracy {:8.4f}'.
                        format(tr_steps, valid_loss.loss, valid_loss.accuracy))
                    model.train()

            # Stop
            if tr_steps >= args.train_steps:
                break
