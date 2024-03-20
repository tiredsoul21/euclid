""" Model Components """
import torch
from torch import nn

class Head(nn.Module):
    """ Single head of self-attention """
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, context):
        """
        Forward Pass of Head
        param: context: torch.tensor (batch, time-step, channels)
        return: torch.tensor (batch, time-step, head size)
        """
        _,time,_ = context.shape

        # Compute the key, query (B,T,hs)
        k = self.key(context)
        q = self.query(context)

        # Compute attention scores ("affinities") - (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 

        # Void future context (B, T, T)
        wei = wei.masked_fill(self.tril[:time, :time] == 0, float('-inf')) 

        # Normalize the attention scores
        wei = torch.nn.functional.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Perform the weighted aggregation of the values (B, T, T) @ (B, T, hs) -> (B, T, hs)
        out = wei @ self.value(context)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, n_embd, head_size, block_size, dropout=0.0):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout=dropout) 
                                    for _ in range(num_heads)])
        # self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, context):
        """
        Forward Pass of MultiHeadAttention
        param: x: torch.tensor (batch, time-step, channels)
        return: torch.tensor (batch, time-step, channels)
        """
        out = torch.cat([head(context) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4* n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, context):
        """ Forward Pass of FeedFoward """
        return self.net(context)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, num_heads, n_embd, block_size, dropout=0.0):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // num_heads
        self.sa = MultiHeadAttention(num_heads, n_embd, head_size, block_size, dropout=dropout)
        self.ffwd = FeedFoward(n_embd, dropout=dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, context):
        """ Forward Pass of Block """
        # Pre Norm (LayerNorm before the attention)
        x = context + self.sa(self.ln1(context))
        x = x + self.ffwd(self.ln2(x))
        return x
