#!/usr/bin/env python3
""" This script performs training of the nano gpt model """
# SentencePiece, tiktoken tokenizers
import argparse

import torch
# from torch import nn, device, functional as Func
# from torch.nn import functional as Func

from ..lib.models import CharacterGPT

# python3 -m src.char_gpt_model  -p data/tinyshakespear.txt

torch.manual_seed(1337)
HARDWARE = 'cuda' if torch.cuda.is_available() else 'cpu'

TRAIN_RATIO = 0.9    # what fraction of the data will be used for training?
BLOCK_SIZE = 256     # what is the maximum context length for predictions?
BATCH_SIZE = 64      # how many independent sequences will we process in parallel?
EVAL_ITERS = 200     # how many iterations to average for loss estimation?
EVAL_STEPS = 500     # how many evaluation steps to take?
MAX_STEPS = 5000     # how many iterations to train for?
VOCAB_SIZE = 65      # how many unique tokens are in the data?
N_EMBD = 384         # how many dimensions to use for the token embeddings?
N_LAYERS = 6         # how many layers to use in the model?
LEARNING_RATE = 3e-4 # how large of a step to take when adjusting the model?
NUM_HEADS = 6        # how many independent attention heads to use?
DROP_RATE = 0.2      # how much dropout to apply in the model?

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(      "--cuda", default=False, help="Enable cuda",  action="store_true")
    parser.add_argument("-p", "--path", required=True, help="Directory or file of text data")
    # parser.add_argument("-v", "--val",                 help="Validation data, default=path/val/")
    # parser.add_argument("-t", "--test",                help="Test data, default=path/test/")
    # parser.add_argument("-r", "--run",  required=True, help="Run name")
    args = parser.parse_args()
    #HARDWARE = device("cuda" if args.cuda else "cpu")

    # Open the file
    with open(args.path, 'r', encoding='utf-8') as file:
        input_data = file.read()
    print(f"Length of the dataset is: {len(input_data)}")

    # Build the vocabulary
    vocab = sorted(list(set(input_data)))
    print(''.join(vocab))
    print(f"Vocabulary size is: {VOCAB_SIZE}")

    # Create a dictionary to map characters to indices
    tokens_dict = { char: idx for idx, char in enumerate(vocab) }
    tokens_lookup_dict = dict(enumerate(vocab))

    def encode(tokens_list: str) -> list[int]:
        """ Convert a list of tokens to a list of indices """
        return [tokens_dict[char] for char in tokens_list]
    def decode(indices: list[int]) -> str:
        """ Convert a list of indices to a list of tokens """
        return ''.join([tokens_lookup_dict[idx] for idx in indices])

    # Split the data into training and validation sets
    input_data_tokens = torch.tensor(encode(input_data), dtype=torch.long)
    data_count = len(input_data_tokens)
    train_data = input_data_tokens[:int(TRAIN_RATIO*data_count)]
    val_data = input_data_tokens[int(TRAIN_RATIO*data_count):]

    model = CharacterGPT(VOCAB_SIZE, NUM_HEADS, N_EMBD, N_LAYERS, BLOCK_SIZE, DROP_RATE)
    model.to(HARDWARE)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

    @torch.no_grad()
    def estimate_loss():
        """ Estimate the loss on the train and validation sets """
        out = {}
        model.eval()
        for split in ['train', 'val']:
            loss_list = torch.zeros(EVAL_ITERS)
            for k in range(EVAL_ITERS):
                eval_context, eval_targets = get_batch(split)
                _, eval_loss = model(eval_context, eval_targets)
                loss_list[k] = eval_loss.item()
            out[split] = loss_list.mean()
        model.train()
        return out

    def get_batch(split):
        """ generate a small batch of data of inputs x and targets y """
        data_split = train_data if split == 'train' else val_data
        ix = torch.randint(len(data_split) - BLOCK_SIZE, (BATCH_SIZE,))
        batch_context = torch.stack([data_split[i:i+BLOCK_SIZE]     for i in ix])
        batch_target  = torch.stack([data_split[i+1:i+BLOCK_SIZE+1] for i in ix])
        batch_context, batch_target = batch_context.to(HARDWARE), batch_target.to(HARDWARE)
        return batch_context, batch_target

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for step in range(MAX_STEPS):
        # every once in a while evaluate the loss on train and val sets
        if step % EVAL_STEPS == 0 or step == MAX_STEPS - 1:
            losses = estimate_loss()
            print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=HARDWARE)
    print(decode(model.generate(context, max_new_tokens=500)[0].tolist()))
