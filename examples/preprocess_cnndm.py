import os

import torch
from path import Path

from pytorch_pretrained_bert import OpenAIGPTTokenizer

data_dir = r'E:\data\cnndm'
split = 'train'
GPT_version = 1

DELIMITER_TOKEN = '_delimiter_'
CLF_TOKEN = '_clf_'

# Load tokenizer
# This loading functions also add new tokens and embeddings called `special_tokens`
special_tokens = [DELIMITER_TOKEN, CLF_TOKEN]
tokenizer = OpenAIGPTTokenizer.from_pretrained(
    'openai-gpt', special_tokens=special_tokens)

# Read data
with open(os.path.join(data_dir, split + '.txt.src'), encoding='utf_8') as f:
    src = [line.rstrip('\n') for line in f]
with open(
        os.path.join(data_dir, split + '.txt.tgt.tagged'),
        encoding='utf_8') as f:
    tgt = [line.rstrip('\n') for line in f]
assert len(src) == len(tgt)

# Run BPE
data = []
for s, t in zip(src, tgt):
    data.append((tokenizer.encode(s), tokenizer.encode(t)))
print(data[0])

# Save
save_dir = os.path.join(data_dir, 'GPT-' + str(GPT_version))
Path(save_dir).mkdir_p()
tokenizer.save_vocabulary(save_dir)
torch.save(data, os.path.join(save_dir, split + '.pt'))
