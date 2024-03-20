""" Agent models """
import copy
import numpy as np

import torch
import torch.nn as nn

from .model_parts import Block

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
    def __init__(self, vocab_size, num_heads, n_embd, n_layers, block_size, dropout=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(num_heads, n_embd, block_size, dropout) 
                                      for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, context: torch.tensor, target: torch.tensor = None):
        """ 'Forward Pass' -- A table lookup """
        _, time = context.shape
        # Get the embeddings for the tokens in the context (batch, time, n_embd)
        token_emb = self.token_embedding_table(context)
        # Get the embeddings for the positions in the context (batch, time, n_embd)
        position_emb = self.position_embedding_table(torch.arange(time, device=context.device))

        # Get the logits for the next token (batch, time, vocab_size)
        x = token_emb + position_emb
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
            context_crop = in_context[:, -self.block_size:]
            logits, _ = self(context_crop)
            logits = logits[:, -1, :]

            # Sample the next token from the distribution (B, 1)
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the context
            in_context = torch.cat((in_context, next_token), dim=1) # (B, T+1)
        return in_context
