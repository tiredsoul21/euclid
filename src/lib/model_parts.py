""" Model Components """
import torch
from torch import nn

from dataclasses import dataclass

@dataclass
class GPTConfig:
    """ GPT Configuration - passes through to the model components """
    block_size: int = 1024
    vocab_size: int = 50304
    num_layers: int = 12
    num_heads: int = 12
    num_embd: int = 768
    dropout: float = 0.0
    bias: bool = True

class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias.
    PyTorch doesn't support simply bias=False
    """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, context):
        """ Forward pass of Layer Norm """
        return nn.functional.layer_norm(context, self.weight.shape, self.weight, self.bias, 1e-5)

class Head(nn.Module):
    """ Single head of self-attention """
    def __init__(self, gptConfig):
        super().__init__()
        assert gptConfig.num_embd % gptConfig.num_heads == 0
        head_size = gptConfig.num_embd // gptConfig.num_heads
        self.config = gptConfig

        self.key   = nn.Linear(self.config.num_embd, head_size, bias=False)
        self.query = nn.Linear(self.config.num_embd, head_size, bias=False)
        self.value = nn.Linear(self.config.num_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(self.config.block_size,
                                                           self.config.block_size)))

        self.dropout = nn.Dropout(self.config.dropout)

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

    def __init__(self, gptConfig):
        super().__init__()
        assert gptConfig.num_embd % gptConfig.num_heads == 0 and gptConfig.num_embd > 0
        assert gptConfig.num_heads > 0
        self.config = gptConfig

        self.heads = nn.ModuleList([Head(self.config) for _ in range(self.config.num_heads)])
        # self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.proj = nn.Linear(self.config.num_embd, self.config.num_embd)
        self.dropout = nn.Dropout(self.config.dropout)

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

    def __init__(self, gptConfig):
        super().__init__()
        assert gptConfig.num_embd is not None and gptConfig.num_embd > 0
        assert gptConfig.dropout is not None and 0.0 <= gptConfig.dropout < 1.0
        self.config = gptConfig
        self.net = nn.Sequential(
            nn.Linear(self.config.num_embd, 4* self.config.num_embd),
            nn.ReLU(),
            nn.Linear(4 * self.config.num_embd, self.config.num_embd),
            nn.Dropout(self.config.dropout),
        )

    def forward(self, context):
        """ Forward Pass of FeedFoward """
        return self.net(context)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, gptConfig):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        assert gptConfig.num_embd % gptConfig.num_heads == 0 and gptConfig.num_embd > 0
        assert gptConfig.block_size > 0
        assert gptConfig.num_heads > 0
        self.config = gptConfig
        self.sa = MultiHeadAttention(self.config)
        self.ffwd = FeedFoward(self.config)
        self.ln1 = nn.LayerNorm(self.config.num_embd)
        self.ln2 = nn.LayerNorm(self.config.num_embd)

    def forward(self, context):
        """ Forward Pass of Block """
        # Pre Norm (LayerNorm before the attention)
        x = context + self.sa(self.ln1(context))
        x = x + self.ffwd(self.ln2(x))
        return x
