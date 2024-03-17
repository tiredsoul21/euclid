#!/usr/bin/env python3
""" This script performs training of the nano gpt model """
import argparse

import torch
from torch import nn, device as hardware,  functional as Func
from torch.nn import functional as Func

from src.lib.data import TextStore

# python3 -m src.train.nano_gpt_model
BLOCK_SIZE = 16
BATCH_SIZE = 20
MAX_ITERS = 5000
LEARNING_RATE = 1e-3

EVAL_INTERVAL = 100
EVAL_ITERS = 200
N_EMBD = 64
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.0

SEED = 42

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        """ head_size: the dimension of the key, query, and value vectors """
        super().__init__()
        self.key =   nn.Linear(N_EMBD, head_size, bias=False)
        self.query = nn.Linear(N_EMBD, head_size, bias=False)
        self.value = nn.Linear(N_EMBD, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))

        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """ apply self-attention to a minibatch of input """
        B,T,C = x.shape
        k = self.key(x)   # (B,T,C)
        q = self.query(x) # (B,T,C)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = Func.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        """ apply attention to a minibatch of input """
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, N_EMBD):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.ReLU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        """ apply the feedforward to a minibatch of input """
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, n_embd, n_head):
        """
        n_embd: embedding dimension,
        n_head: the number of heads we'd like
        """
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """ apply the block to a minibatch of input """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module):
    """ a simple autoregressive language model """
    def __init__(self, n_words, process_device):
        super().__init__()
        self.device = process_device
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(n_words, N_EMBD)
        self.position_embedding_table = nn.Embedding(BLOCK_SIZE, N_EMBD)
        self.blocks = nn.Sequential(*[Block(N_EMBD, n_head=N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD) # final layer norm
        self.lm_head = nn.Linear(N_EMBD, n_words)

    def forward(self, idx, targets=None):
        """ compute the logits for the next token from a minibatch of input tokens """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        model_logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            model_loss = None
        else:
            B, T, C = model_logits.shape
            model_logits = model_logits.view(B*T, C)
            targets = targets.view(B*T)
            model_loss = Func.cross_entropy(model_logits, targets)

        return model_logits, model_loss

    def generate(self, idx, max_new_tokens):
        """ generate a sequence of tokens from the model """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last BLOCK_SIZE tokens
            idx_cond = idx[:, -BLOCK_SIZE:]
            # get the predictions
            model_logits, _ = self(idx_cond)
            # focus only on the last time step
            model_logits = model_logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = Func.softmax(model_logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(      "--cuda", default=False, help="Enable cuda",  action="store_true")
    parser.add_argument("-p", "--path", required=True, help="Directory or file of text data")
    # parser.add_argument("-v", "--val",                 help="Validation data, default=path/val/")
    # parser.add_argument("-t", "--test",                help="Test data, default=path/test/")
    # parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()
    device = hardware("cuda" if args.cuda else "cpu")

    # create a TextStore object
    text_store = TextStore(args.path)

    # create a model
    model = BigramLanguageModel(len(text_store.tokens), device)
    m = model.to(device)

    @torch.no_grad()
    def estimate_loss():
        """ estimate the loss on the train and val sets """
        out = {}
        model.eval()
        for split in ['train', 'val']:
            losses = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                X, Y = text_store.get_batch(BLOCK_SIZE, BATCH_SIZE, split)
                _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        model.train()
        return out

    # print the number of parameters in the model
    print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for iteration in range(MAX_ITERS):

        # every once in a while evaluate the loss on train and val sets
        if iteration % EVAL_INTERVAL == 0 or iteration == MAX_ITERS - 1:
            eval_loss = estimate_loss()
            print(f"step {iteration}: train \
                  loss {eval_loss['train']:.4f}, \
                  val loss {eval_loss['val']:.4f}")

        # sample a batch of data
        xb, yb = text_store.get_batch(BLOCK_SIZE, BATCH_SIZE)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(text_store.decode(m.generate(context, max_new_tokens=2000)[0].tolist()))

# SentencePiece, tiktoken tokenizers
