#!/usr/bin/env python

import os.path

import torch
import torch.nn as nn

import safetensors.torch

from tokenizer import encode, decode
from model import SmolLM, block_size

import time

# training parameters
batch_size = 64
learning_rate = 3e-4
eval_interval = 100
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model_path = "weights.pth"
data_path = 'data/linux.txt'

# data
if os.path.exists(data_path + ".tokens"):
    data = safetensors.torch.load_file(data_path + ".tokens")["data"]
else:
    print("Tokenizing data... (will take a while)")
    with open(data_path, 'r') as f: text = f.read()
    data = torch.tensor(encode(text), dtype=torch.long)
    safetensors.torch.save_file({ "data": data }, data_path + ".tokens")

# train & test
n = int(0.9 * len(data))
train_data = data[:n]
test_data = data[n:]

# block/batch configuration
def get_batch(split):
    assert(split == "train" or split == "test")
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
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
    m.load_state_dict(torch.load(model_path, weights_only=True))

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