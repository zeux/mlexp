#!/usr/bin/env python

import sys
import os.path

import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import encode, decode
from model import SmolLM, block_size

model_path = "weights.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# generate new tokens by repeatedly getting the next prediction from the model
def generate(model, idx, max_new_tokens):
    # idx is (B,T) array of indices in the current context
    for _ in range(max_new_tokens):
        # get prediction from last context
        pidx = idx[:, -block_size:]
        logits, _ = model(pidx)
        # focus on last time step
        logits = logits[:, -1, :] # becomes (B,C)
        # apply softmax to get probabilities
        probs = F.softmax(logits, dim=-1) # (B,C)
        # sample from distribution
        idx_next = torch.multinomial(probs, num_samples=1) # (B,1)
        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1) # (B,T+1)

    return idx

# infer
m = SmolLM().to(device)
print(f"{sum(p.numel() for p in m.parameters())/1e6:.2f} M parameters")

mtime = None

for prompt in sys.stdin:
    if mtime != os.path.getmtime(model_path):
        mtime = os.path.getmtime(model_path)
        print(f"Loading model from {model_path}")
        m.load_state_dict(torch.load(model_path, weights_only=True))

    idx = torch.tensor([encode(prompt.rstrip())], dtype=torch.long, device=device)
    print(decode(generate(m, idx, max_new_tokens=500)[0].tolist()))