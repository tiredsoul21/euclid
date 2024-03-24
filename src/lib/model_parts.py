""" Model Components """
import math
from dataclasses import dataclass

import torch
from torch import nn
# Import scaled_dot_product_attention as sdp_attention
from torch.nn.functional import scaled_dot_product_attention as sdp_attention


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

        self.key   = nn.Linear(gptConfig.num_embd, head_size, bias=False)
        self.query = nn.Linear(gptConfig.num_embd, head_size, bias=False)
        self.value = nn.Linear(gptConfig.num_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(gptConfig.block_size,
                                                           gptConfig.block_size)))

        self.dropout = nn.Dropout(gptConfig.dropout)

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

        self.heads = nn.ModuleList([Head(gptConfig) for _ in range(gptConfig.num_heads)])
        # self.proj = nn.Linear(head_size * num_heads, num_embd)
        self.proj = nn.Linear(gptConfig.num_embd, gptConfig.num_embd)
        self.dropout = nn.Dropout(gptConfig.dropout)

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

class CausalSelfAttention(nn.Module):
    """ Causal Self-Attention -- combining the multi-head attention with the causal mask """
    def __init__(self, gptConfig):
        super().__init__()
        assert gptConfig.num_embd is not None and gptConfig.num_embd > 0
        assert gptConfig.num_heads is not None and gptConfig.num_heads > 0
        assert gptConfig.num_embd % gptConfig.num_heads == 0
        assert gptConfig.dropout is not None and 0.0 <= gptConfig.dropout < 1.0
        assert gptConfig.block_size is not None and gptConfig.block_size > 0
        self.config = gptConfig

        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(gptConfig.num_embd, 3 * gptConfig.num_embd, bias=gptConfig.bias)
        # output projection
        self.c_proj = nn.Linear(gptConfig.num_embd, gptConfig.num_embd, bias=gptConfig.bias)
        # regularization
        self.attn_dropout = nn.Dropout(gptConfig.dropout)
        self.resid_dropout = nn.Dropout(gptConfig.dropout)

        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias",
                                 torch.tril(torch.ones(gptConfig.block_size, gptConfig.block_size))
                                 .view(1, 1, gptConfig.block_size, gptConfig.block_size))

    def forward(self, context):
        """ Forward Pass of CausalSelfAttention """
        batch, time, channel = context.size()

        # Calc query, key, values for all heads in batch and move head forward as the batch dim
        q, k, v  = self.c_attn(context).split(self.config.num_embd, dim=2)
        head_size = channel // self.config.num_heads
        # K, Q, V: (B, nh, T, hs)
        k = k.view(batch, time, self.config.num_heads, head_size).transpose(1, 2)
        q = q.view(batch, time, self.config.num_heads, head_size).transpose(1, 2)
        v = v.view(batch, time, self.config.num_heads, head_size).transpose(1, 2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = sdp_attention(q, k, v, attn_mask=None, 
                              dropout_p=self.config.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:time,:time] == 0, float('-inf'))
            att = nn.functional.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(batch, time, channel)

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, gptConfig):
        super().__init__()
        assert gptConfig.num_embd is not None and gptConfig.num_embd > 0
        assert gptConfig.dropout is not None and 0.0 <= gptConfig.dropout < 1.0
        self.net = nn.Sequential(
            nn.Linear(gptConfig.num_embd, 4* gptConfig.num_embd),
            nn.ReLU(),
            nn.Linear(4 * gptConfig.num_embd, gptConfig.num_embd),
            nn.Dropout(gptConfig.dropout),
        )

    def forward(self, context):
        """ Forward Pass of FeedFoward """
        return self.net(context)

class MLP(nn.Module):
    """Multi-Layer Perceptron to aid in complex spatial reasoning"""

    def __init__(self, gptConfig):
        super().__init__()
        assert gptConfig.num_embd is not None and gptConfig.num_embd > 0
        assert gptConfig.dropout is not None and 0.0 <= gptConfig.dropout < 1.0
        self.net = nn.Sequential(
            nn.Linear(gptConfig.num_embd, 4 * gptConfig.num_embd, bias=gptConfig.bias),
            nn.GELU(),
            nn.Linear(4 * gptConfig.num_embd, gptConfig.num_embd, bias=gptConfig.bias),
            nn.Dropout(gptConfig.dropout)
        )

    def forward(self, context):
        """ Forward Pass of FeedFoward """
        return self.net(context)

class OldBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, gptConfig):
        # num_embd: embedding dimension, num_heads: the number of heads we'd like
        super().__init__()
        assert gptConfig.num_embd % gptConfig.num_heads == 0 and gptConfig.num_embd > 0
        assert gptConfig.block_size > 0
        assert gptConfig.num_heads > 0

        self.sa = MultiHeadAttention(gptConfig)
        self.ffwd = FeedFoward(gptConfig)
        self.ln1 = nn.LayerNorm(gptConfig.num_embd)
        self.ln2 = nn.LayerNorm(gptConfig.num_embd)

    def forward(self, context):
        """ Forward Pass of Block """
        # Pre Norm (LayerNorm before the attention)
        x = context + self.sa(self.ln1(context))
        x = x + self.ffwd(self.ln2(x))
        return x

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, gptConfig):
        super().__init__()
        assert gptConfig.num_embd % gptConfig.num_heads == 0 and gptConfig.num_embd > 0
        assert gptConfig.block_size > 0
        assert gptConfig.num_heads > 0
        self.ln1 = LayerNorm(gptConfig.num_embd, bias=gptConfig.bias)
        self.attn = CausalSelfAttention(gptConfig)
        self.ln2 = LayerNorm(gptConfig.num_embd, bias=gptConfig.bias)
        self.mlp = MLP(gptConfig)

    def forward(self, context):
        """ Forward Pass of Block """
        # Pre Norm (LayerNorm before the attention)
        x = context + self.attn(self.ln1(context))
        x = x + self.mlp(self.ln2(x))
        return x
