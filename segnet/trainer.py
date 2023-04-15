import torch
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt

import time

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def eval(data, model):
    loss = 0
    model.eval()
    for x, y in data:
        x, y = x.to(device), y.to(device)
        pred = model.forward(x)
        loss += model.loss(pred, y).item()
    return loss / len(data)

def trainiter(data, model, optimizer):
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

def train(model, train_data, val_data, epochs=5, optimizer=torch.optim.AdamW, plot=True, **optargs):
    optimizer = optimizer(model.parameters(), **optargs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    start = time.time()
    train_losses, val_losses = [], []
    
    for e in range(epochs):
        epoch_start = time.time()
        train_loss = trainiter(train_data, model, optimizer)
        val_loss = eval(val_data, model)
        print(f"\rEpoch {e}/{epochs}: {time.time()-epoch_start:.2f} sec, train loss {train_loss:>8f}, val loss {val_loss:>8f}, lr {optimizer.param_groups[0]['lr']}", end="")
        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
    nparams = sum(p.numel() for p in model.parameters())    
    print(f"\rTrain [{nparams/1e6:.3f}M params]: {epochs} epochs took {time.time()-start:.2f} sec, train loss {train_loss:>8f}, val loss {val_loss:>8f}")
    
    if plot:
        # skip first epoch as it usually has an overly high train loss since it aggregates very early samples
        plt.plot(train_losses[1:], label="train")
        plt.plot(val_losses[1:], label="val")
        plt.legend()
        plt.show()