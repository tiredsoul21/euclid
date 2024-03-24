"""
Train the NanoGPT model.  Is trained as a sub-word language model.
This is the same / similar to the GPT-2 model, but is a smaller version.
Correlates to: https://github.com/karpathy/nanoGPT
"""
import os
import time
import math
import argparse
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from ..lib.models import NanoGPT
from ..lib.model_parts import GPTConfig

# python3 -m src.train.nano_gpt_model --init scratch

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
OUT_DIR = 'out'        # output directory
LOG_INTERVAL = 10      # how many steps to log metrics
EVAL_INTERVAL = 1000   # how often to run eval
EVAL_ITERS = 200       # how many iters to average for loss estimation
EVAL_ONLY = False      # Script exits right after the first eval?
ALWAYS_SAVE = True     # Save a checkpoint each eval?

# data
SEED = 42                           # random seed for reproducibility
DATASET  = 'openwebtext'            # which dataset to use huggerface/openwebtext
DATA_DIR = os.path.join('/home/derrick/data/', DATASET) # path to where the data is
GRAD_ACCUMULATION_STEPS = 5 * 8     # used to simulate larger batch sizes
BATCH_SIZE = 4                      # if GRAD_ACCUMULATION_STEPS > 1, this is micro-batch size
BLOCK_SIZE = 1024

# model
NUM_LAYERS = 12   # number of transformer layers
NUM_HEADS = 12    # number of attention heads
NUM_EMBD = 768    # embedding dimension
DROPOUT = 0.0     # for pretraining 0 is good, for finetuning try 0.1+
BIAS = False      # do we use bias inside LayerNorm and Linear layers?

# adamw optimizer
LEARNING_RATE = 6e-4   # max learning rate
MAX_ITERS = 600000     # total number of training iterations
WEIGHT_DECAY = 1e-1    # strength of weight decay
BETA1 = 0.9            # beta1 for adam
BETA2 = 0.95           # beta2 for adam
GRAD_CLIP = 1.0        # clip gradients at this value, or disable if == 0.0

# learning rate & decay settings
LR_DECAY = True          # whether to decay the learning rate
WARMUP_ITERS = 2000      # how many steps to warm up for
LR_DECAY_ITERS = 600000  # should be ~= max_iters per Chinchilla
LR_MIN = 6e-5            # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
DTYPE = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'

# -----------------------------------------------------------------------------
CONFIG_KEYS = [k for k,v in globals().items() if not k.startswith('_')
               and isinstance(v, (int, float, bool, str))]
# exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in CONFIG_KEYS} # will be useful for logging
# -----------------------------------------------------------------------------


TOKENS_PER_ITER = GRAD_ACCUMULATION_STEPS * BATCH_SIZE * BLOCK_SIZE
print(f"tokens per iteration will be: {TOKENS_PER_ITER:,}")

os.makedirs(OUT_DIR, exist_ok=True)
torch.manual_seed(SEED)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[DTYPE]
ctx = nullcontext() if DEVICE == 'cpu' else torch.amp.autocast(device_type=DEVICE, dtype=ptdtype)

def get_batch(split):
    """ Get a batch of data from the dataset """
    # Select dataset by need
    # We recreate np.memmap every batch to avoid a memory leak
    if split == 'train':
        data = np.memmap(os.path.join(DATA_DIR, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(DATA_DIR, 'val.bin'),   dtype=np.uint16, mode='r')

    # Select a random batch of BLOCK_SIZE tokens
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([torch.from_numpy((data[i:i+BLOCK_SIZE]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+BLOCK_SIZE]).astype(np.int64)) for i in ix])

    # Move to DEVICE
    if DEVICE == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(DEVICE, non_blocking=True)
        y = y.pin_memory().to(DEVICE, non_blocking=True)
    else:
        x, y = x.to(DEVICE), y.to(DEVICE)

    return x, y

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--init", default='scratch', help="From 'scratch'/'resume' training?")
    parser.add_argument("-c", "--compile", action='store_true', help="Compile the model?")
    args = parser.parse_args()

    # Initial valies may be overwritten by the checkpoint
    iter_num = 0
    best_val_loss = 1e9

    # load the metadata if it exists (for efficiency)
    meta_path = os.path.join(DATA_DIR, 'meta.pkl')
    META_VOCAB_SIZE = None
    if os.path.exists(meta_path):
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        META_VOCAB_SIZE = meta['vocab_size']
        print(f"found vocab_size = {META_VOCAB_SIZE} (inside {meta_path})")

    # model init
    model_args = dict(num_layers=NUM_LAYERS, num_heads=NUM_HEADS, num_embd=NUM_EMBD,
                      block_size=BLOCK_SIZE, bias=BIAS, vocab_size=None,
                      dropout=DROPOUT)

    if args.init == 'scratch':
        # Create a new model from scratch
        print("Initializing a new model from scratch")
        model_args['vocab_size'] = META_VOCAB_SIZE if META_VOCAB_SIZE is not None else 50304
        print (f"vocab_size = {model_args['vocab_size']}")
        gptconf = GPTConfig(**model_args)
        model = NanoGPT(gptconf)
    elif args.init == 'resume':
        # Resume training from a checkpoint
        print(f"Resuming training from {OUT_DIR}")
        ckpt_path = os.path.join(OUT_DIR, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        checkpoint_model_args = checkpoint['model_args']

        # Overwrite the model args with the ones from the checkpoint
        for k in ['num_layers', 'num_heads', 'num_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]

        # create the model
        gptconf = GPTConfig(**model_args)
        model = NanoGPT(gptconf)
        state_dict = checkpoint['model']

        # remove the original prefix from the keys
        for k,v in list(state_dict.items()):
            if k.startswith('_orig_mod.'):
                state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)

        # Load the state dict
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']

    # crop down the model block size if desired, using model surgery
    if BLOCK_SIZE < model.config.block_size:
        model.crop_block_size(BLOCK_SIZE)
        model_args['block_size'] = BLOCK_SIZE
    model.to(DEVICE)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=DTYPE == 'float16')

    # optimizer
    optimizer = model.configure_optimizers(WEIGHT_DECAY, LEARNING_RATE, (BETA1, BETA2), DEVICE)
    if args.init == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    # Clean up memory
    checkpoint = None

    # compile the model
    if args.compile:
        print("compiling the model...")
        model = torch.compile(model)

    # helps estimate an arbitrarily accurate loss over either split using many batches
    @torch.no_grad()
    def estimate_loss():
        """ Estimate the loss over the multiple iterations """
        out = {}
        model.eval()
        for split in ['train', 'val']:
            est_losses = torch.zeros(EVAL_ITERS)
            for i in range(EVAL_ITERS):
                context, targets = get_batch(split)
                with ctx:
                    _, est_loss = model(context, targets)
                est_losses[i] = est_loss.item()
            out[split] = est_losses.mean()
        model.train()
        return out

    def get_lr(it):
        """ Get the learning rate for the iteration """
        # Linear warmup for warmup_iters steps
        if it < WARMUP_ITERS:
            return LEARNING_RATE * it / WARMUP_ITERS

        #  Decay interval done, return min
        if it > LR_DECAY_ITERS:
            return LR_MIN

        # Scale LR by cosine annealing from LR to LR_MIN
        decay_ratio = (it - WARMUP_ITERS) / (LR_DECAY_ITERS - WARMUP_ITERS)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return LR_MIN + coeff * (LEARNING_RATE - LR_MIN)

    # training loop
    X, Y = get_batch('train') # fetch the very first batch

    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if LR_DECAY else LEARNING_RATE
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num % EVAL_INTERVAL == 0:
            loss = estimate_loss()
            print(f"step {iter_num}: train loss {loss['train']:.4f}, val loss {loss['val']:.4f}")

            if loss['val'] < best_val_loss or ALWAYS_SAVE:
                best_val_loss = loss['val']
                if iter_num > 0:
                    checkpoint = {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {OUT_DIR}")
                    #Save model with iteration number
                    torch.save(checkpoint, os.path.join(OUT_DIR, 'ckpt_' + str(iter_num) + '.pt'))

        if iter_num == 0 and EVAL_ONLY:
            break

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for micro_step in range(GRAD_ACCUMULATION_STEPS):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / GRAD_ACCUMULATION_STEPS
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train')
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
        # clip the gradient
        if GRAD_CLIP != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)

        # timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % LOG_INTERVAL == 0:
            # get loss as float. note: this is a CPU-GPU sync point
            # scale to undo the division above, approx the true total loss (exact would be a sum)
            lossf = loss.item() * GRAD_ACCUMULATION_STEPS
            if local_iter_num >= 5: # let the training loop settle a bit
                mfu = model.estimate_mfu(BATCH_SIZE * GRAD_ACCUMULATION_STEPS, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        iter_num += 1
        local_iter_num += 1

        # termination conditions
        if iter_num > MAX_ITERS:
            break
