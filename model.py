import torch
import torch.nn as nn
from torch.nn import functional as F

from tokenizer import vocab_size

# parameters
block_size = 256
n_embed = 768
n_head = 8 # can't push to 12 without OOM
n_layers = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu' # TODO

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # attention key/query/value for each token
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=dropout, is_causal=True)
        return out

class Heads(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        assert(n_embed % num_heads == 0)
        self.sa = Heads(num_heads, n_embed // num_heads)
        self.ffwd = FeedForward()
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class SmolLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.pos_embedding_table = nn.Embedding(block_size, n_embed)
        assert(n_embed % 4 == 0)
        self.blocks = nn.Sequential(*[Block(n_head) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.tok_embedding_table(idx) # (B,T,C)
        pos_emb = self.pos_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B,T,vocab)
        
        if targets is None:
            loss = None
        else:
            # cross_entropy requires batch, channel order
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss