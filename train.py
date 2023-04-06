#!/usr/bin/env python

import os.path

import torch
import torch.nn as nn

import safetensors.torch

from tokenizer import encode, decode
from model import SmolLM, block_size

import time
import random

# training parameters
batch_size = 64
learning_rate = 3e-4
eval_interval = 100
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "weights.pth"
data_path = 'data/all.txt'

# data
with open(data_path, 'r', errors='ignore') as f:
    lines = f.readlines()
    train_data = lines[:int(0.9 * len(lines))]
    test_data = lines[len(train_data):]

# block/batch configuration
def random_block(split, size):
    assert(split == "train" or split == "test")
    data = train_data if split == "train" else test_data

    # at least one token per line => guarantees no oob access
    line = random.randrange(len(data) - 2 * size)
    result = []

    while len(result) < 2 * size:
        result += encode(data[line])
        line += 1

    # random offset within the block so that we don't always start at line start
    off = random.randrange(len(result) - size)

    return result[off:off + size]

def get_batch(split):
    blocks = [torch.tensor(random_block(split, block_size + 1), dtype=torch.long) for _ in range(batch_size)]
    x = torch.stack([block[:block_size] for block in blocks])
    y = torch.stack([block[1:block_size+1] for block in blocks])
    return x.to(device), y.to(device)

# loss estimation
@torch.no_grad()
def estimate_loss(m, split):
    m.eval()
    losses = torch.zeros(eval_iters)
    for k in range(eval_iters):
        xb, yb = get_batch(split)
        logits, loss = m(xb, yb)
        losses[k] = loss.item()
    m.train()
    return losses.mean()

# train
m = SmolLM().to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")

if os.path.exists(model_path):
    m.load_state_dict(torch.load(model_path, weights_only=True), strict=False)

cpt = time.time()

for iter in range(1000000000):
    if iter % eval_interval == 0:
        elt = time.time() - cpt
        torch.save(m.state_dict(), model_path)
        print(f"step {iter}: train loss {estimate_loss(m, 'train'):.4f}, test loss {estimate_loss(m, 'test'):.4f}, time {elt:.1f} sec")
        cpt = time.time()
    # sample a batch of data
    xb, yb = get_batch('train')
    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()