# Note: this file is *not* meant to be executable by itself (although maybe it will become a module one day)
# Instead, it provides snippets to copy & paste into notebooks that are useful for my ML experiments and occur in many of them.

# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as D

import torchvision.datasets
import torchvision.transforms as T
import torchvision.models

import time

# Notebook imports
import matplotlib.pyplot as plt
%matplotlib inline

# Device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Data loaders
batch_size = 4
train_cutoff = int(len(dataset) * 0.9)
train_data = D.DataLoader(D.Subset(dataset_aug, range(train_cutoff)), batch_size=batch_size, num_workers=4)
val_data = D.DataLoader(D.Subset(dataset, range(train_cutoff, len(dataset))), batch_size=batch_size, num_workers=4)

# Train/eval loops
def train(data, model, optimizer):
    losses = []
    model.train()
    for x, y in data:
        x, y = x.to(device), y.to(device)

        pred = model.forward(x)
        loss = model.loss(pred, y)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return torch.tensor(losses).mean().item()

@torch.no_grad()
def eval(data, model):
    loss = 0
    model.eval()
    for x, y in data:
        x, y = x.to(device), y.to(device)
        pred = model.forward(x)
        loss += model.loss(pred, y).item()
    return loss / len(data)

def trainloop(model, epochs=5, optimizer=torch.optim.AdamW, **optargs):
    optimizer = optimizer(model.parameters(), **optargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    start = time.time()
    train_losses, val_losses = [], []
    for e in range(epochs):
        epoch_start = time.time()
        train_loss = train(train_data, model, optimizer)
        val_loss = eval(val_data, model)
        print(f"\rEpoch {e}/{epochs}: {time.time()-epoch_start:.2f} sec, train loss {train_loss:>8f}, val loss {val_loss:>8f}, lr {optimizer.param_groups[0]['lr']}", end="")
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    nparams = sum(p.numel() for p in model.parameters())
    print(f"\rTrain [{nparams/1e6:.3f}M params]: {epochs} epochs took {time.time()-start:.2f} sec, train loss {train_loss:>8f}, val loss {val_loss:>8f}")
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.show()