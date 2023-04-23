import random

import torch
import torch.nn as nn
import torch.optim as optim


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        # initialize model parameters
        self.input_size = input_size
        self.emb_size = emb_size
        self.linear_size = linear_size
        self.output_size = output_size
        self.max_length = max_length

        # initialize model layers
        self.self_attention1 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm1 = nn.LayerNorm(self.linear_size)
        self.feedforward1 = nn.Linear(self.emb_size, self.linear_size)
        self.feedforward2 = nn.Linear(self.emb_size, self.linear_size)
        self.norm2 = nn.LayerNorm(self.linear_size)

    # TODO Add drop out
    # - p4. Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack.
    # - We apply dropout [33] to the output of each sub-layer, before it is added to thesub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
    # positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
    # Pdrop = 0.1.(Vaswani, 2017)

    def forward(self, inputs):
        # multihead self attention
        self_attention1 = self.self_attention1(inputs, inputs, inputs)
        # add and normalize #TODO p4. After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent’s input to its output.
        add_residual1 = inputs + self_attention1[0]
        norm1 = self.norm1(add_residual1)
        # feed forward (x2)
        # Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        feedforward1 = self.feedforward1(norm1)
        # feedforward2 = self.feedforward2(norm1) #TODO uncomment
        # add and normalize
        add_residual2 = norm1 + feedforward1
        norm2 = self.norm2(add_residual2)
        output = norm2
        return output
