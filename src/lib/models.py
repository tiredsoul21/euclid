""" Agent models """
import copy
import math

import numpy as np
import torch
from torch import nn

from .model_parts import Block, LayerNorm

class DQNConv1D(nn.Module):
    """ DQN model for 1D convolutional input """
    def __init__(self, shape, actionCount):
        """
        Create a DQN model for 1D convolutional input
        :param shape: shape of input (channels, in)
        """
        super(DQNConv1D, self).__init__()

        # Create convolutional layers - transform time series into features
        self.price_conv = nn.Sequential(
            nn.Conv1d(shape[0], 128, 5), nn.ReLU(),
            nn.Conv1d(128, 128, 5),      nn.ReLU(),
        )

        conv_output_size = self._get_conv_out(shape)

        # Create fully connected layers - transform features into value
        self.fc_value = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Create fully connected layers - transform features into advantage for each action
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, actionCount)
        )

    def _get_conv_out(self, shape):
        """
        Calculate the output size of the convolutional layers
        :param shape: shape of the input
        :return: size of the output
        """
        o = self.price_conv(torch.zeros(1, *shape))
        return o.view(1, -1).size(1) + 2

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input
        :return: value and advantage
        """
        # get priceData from dictionary
        conv_out = self.price_conv(x['priceData']).view(x['priceData'].size()[0], -1)

        # Get/Append hasPosition and position
        conv_out = torch.cat([conv_out, x['hasPosition'], x['position']], dim=1)

        val = self.fc_value(conv_out)
        adv = self.fc_advantage(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))

class DQNConv2D(nn.Module):
    """ DQN model for 2D convolutional input """
    def __init__(self, shapes, actionCount):
        """
        Create a DQN model for 1D convolutional input
        :param shape: shape of input (channels, in)
        """
        super(DQNConv2D, self).__init__()

        # Create convolutional layers - transform time series into features
        self.price_conv = nn.Sequential(
            nn.Conv2d(1, 128, 3), nn.ReLU(),
            nn.Conv2d(128, 128, (1,3)), nn.ReLU(),
        )

        conv_output_size = self._get_conv_out(shapes)

        # Create fully connected layers - transform features into value
        self.fc_value = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, 1)
        )

        # Create fully connected layers - transform features into advantage for each action
        self.fc_advantage = nn.Sequential(
            nn.Linear(conv_output_size, 512), nn.ReLU(),
            nn.Linear(512, actionCount)
        )

    def _get_conv_out(self, shapes):
        """
        Calculate the output size of the convolutional layers
        :param shape: shape of the input
        :return: size of the output
        """
        print(shapes)
        o = self.price_conv(torch.zeros(1, *(shapes['priceData'])))
        # Flatten and return the output
        return int(np.prod(o.size())) + 2

    def forward(self, x):
        """
        Forward pass of the model
        :param x: input
        :return: value and advantage
        """
        # get priceData from dictionary and add channel dimension
        # [batch, channels, *shape] == [batch, channels, x, y] required by Conv2D
        conv_out = self.price_conv(x['priceData'].unsqueeze(1))

        # Flatten the output of convolutional layers
        conv_out = conv_out.view(x['priceData'].size()[0], -1)

        # Get/Append hasPosition and position
        conv_out = torch.cat([conv_out, x['hasPosition'], x['position']], dim=1)

        val = self.fc_value(conv_out)
        adv = self.fc_advantage(conv_out)
        return val + (adv - adv.mean(dim=1, keepdim=True))

class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__(self, model):
        """
        Create a target net from a model (deep copy)
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        """
        Copy weights from model to target model
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """

        # Check input parameters
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0

        state = self.model.state_dict()
        target_state = self.target_model.state_dict()

        # Blend the parameters with a weighted average
        for k, v in state.items():
            target_state[k] = target_state[k] * alpha + (1 - alpha) * v

        # Load the blended parameters into the target model
        self.target_model.load_state_dict(target_state)

class BigramLanguageModel(nn.Module):
    """ A simple bigram language model """
    def __init__(self, vocab_size, n_embd):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, context: torch.tensor, target: torch.tensor = None):
        """ 'Forward Pass' -- A table lookup """
        # Get the embeddings for the tokens in the context (batch, time, n_embd)
        token_emb = self.token_embedding_table(context)
        # Get the logits for the next token (batch, time, vocab_size)
        logits = self.lm_head(token_emb)

        if target is None:
            loss = None
        else:
            batch, time, channels = logits.shape

            # Flatten the logits and target for cross_entropy & return
            logits = logits.view(batch*time, channels)
            targets = target.view(batch*time)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, in_context, max_new_tokens):
        """ Generate new tokens given a context in the Time dimension """
        for _ in range(max_new_tokens):
            # Get the predictions & remove the time dimension (B, C)
            logits, _ = self(in_context)
            logits = logits[:, -1, :]

            # Sample the next token from the distribution (B, 1)
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the context
            in_context = torch.cat((in_context, next_token), dim=1) # (B, T+1)
        return in_context

class CharacterGPT(nn.Module):
    """ A simple bigram language model """
    def __init__(self, gptConfig):
        super().__init__()
        # Assert required parameters
        assert gptConfig.vocab_size is not None
        assert gptConfig.num_embd is not None
        assert gptConfig.block_size is not None
        assert gptConfig.num_layers is not None
        self.config = gptConfig

        # Create the token and position embedding tables
        self.tkn_embed_tbl = nn.Embedding(self.config.vocab_size, self.config.num_embd)
        self.pos_embed_tbl = nn.Embedding(self.config.block_size, self.config.num_embd)

        self.blocks = nn.Sequential(*[Block(self.config) for _ in range(self.config.num_layers)])
        self.ln_f = nn.LayerNorm(self.config.num_embd)
        self.lm_head = nn.Linear(self.config.num_embd, self.config.vocab_size)

    def forward(self, context: torch.tensor, target: torch.tensor = None):
        """ 'Forward Pass' -- A table lookup """
        _, time = context.shape
        # Get the embeddings for the tokens in the context (batch, time, n_embd)
        tkn_emb = self.tkn_embed_tbl(context)
        # Get the embeddings for the positions in the context (batch, time, n_embd)
        pos_emb = self.pos_embed_tbl(torch.arange(time, device=context.device))

        # Get the logits for the next token (batch, time, vocab_size)
        x = tkn_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            batch, time, channels = logits.shape

            # Flatten the logits and target for cross_entropy & return
            logits = logits.view(batch*time, channels)
            targets = target.view(batch*time)
            loss = nn.functional.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, in_context, max_new_tokens):
        """ Generate new tokens given a context in the Time dimension """
        for _ in range(max_new_tokens):
            # Get the predictions & remove the time dimension (B, C)
            context_crop = in_context[:, -self.config.block_size:]
            logits, _ = self(context_crop)
            logits = logits[:, -1, :]

            # Sample the next token from the distribution (B, 1)
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the context
            in_context = torch.cat((in_context, next_token), dim=1) # (B, T+1)
        return in_context

class NanoGPT(nn.Module):
    """ NanoGPT model """

    def __init__(self, config):
        super().__init__()

        # Required parameters
        assert config.vocab_size is not None
        assert config.block_size is not None
        assert config.num_layers is not None
        assert config.num_heads is not None
        assert config.num_embd is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            tkn_embed_tbl = nn.Embedding(config.vocab_size, config.num_embd),
            pos_embed_tbl = nn.Embedding(config.block_size, config.num_embd),
            dropout = nn.Dropout(config.dropout),
            blocks = nn.ModuleList([Block(config) for _ in range(config.num_layers)]),
            ln_f = LayerNorm(config.num_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.num_embd, config.vocab_size, bias=False)

        # with weight tying https://paperswithcode.com/method/weight-tying
        self.transformer.tkn_embed_tbl.weight = self.lm_head.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.num_layers))

        # report number of parameters
        print(f"number of parameters: {self.get_num_params() / 1e6:.2f}M")

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.pos_embed_tbl.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, context: torch.tensor, targets: torch.tensor = None):
        """ Forward pass of the NanoGPT model """
        device = context.device
        _, time = context.size()

        # Make sure the context isn't too long
        assert time <= self.config.block_size, (f"Cannot forward sequence of length {time}, "
                                                f"block size is only {self.config.block_size}")

        # Get the embeddings for the tokens in the context (batch, time, n_embd)
        tok_emb = self.transformer.tkn_embed_tbl(context)
        # Get the embeddings for the positions in the context (batch, time, n_embd)
        pos = torch.arange(0, time, dtype=torch.long, device=device)
        pos_emb = self.transformer.pos_embed_tbl(pos)

        # Dropout and pass through the transformer blocks
        x = self.transformer.dropout(tok_emb + pos_emb)
        for block in self.transformer.blocks:
            x = block(x)
        # Layer normalization
        x = self.transformer.ln_f(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)),
                                               targets.view(-1), ignore_index=-1)
        else:
            # Inference: predict the next token based on the last token in the input sequence
            # note: using list [-1] to preserve the time dim
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(self, context, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take an input context and generate new tokens. Can use temperature to control
        the diversity of the generated text and top_k to crop the distribution to the
        top k options at each step. Both of these will affect the output token distribution.
        """
        for _ in range(max_new_tokens):
            # Get the predictions & remove the time dimension (B, C) - crop to needed size
            context_crop = context if context.size(1) <= self.config.block_size \
                                   else context[:, -self.config.block_size:]
            logits, _ = self(context_crop)
            # Scale by desired temperature (controls diversity of generated text)
            logits = logits[:, -1, :] / temperature

            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Sample the next token from the distribution (B, 1)
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the context
            context = torch.cat((context, next_token), dim=1)

        return context