import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        # init decoder 1
        self.self_attention3 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm5 = nn.LayerNorm(self.linear_size)
        self.encoder_decoder_attention1 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm6 = nn.LayerNorm(self.linear_size)
        self.feedforward5 = nn.Linear(self.emb_size, self.linear_size)
        self.feedforward6 = nn.Linear(self.emb_size, self.linear_size)
        self.norm7 = nn.LayerNorm(self.linear_size)

    # TODO Add drop out
    # - p4. Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack.
    # - We apply dropout [33] to the output of each sub-layer, before it is added to thesub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
    # positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
    # Pdrop = 0.1.(Vaswani, 2017)

    def forward(self, inputs, encoder_output):
        # multihead self attention
        self_attention3 = self.self_attention3(inputs, inputs, inputs)
        # add and normalize
        add_residual5 = inputs + self_attention3[0]
        norm5 = self.norm5(add_residual5)
        # encoder-decoder multihead attention
        # In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. (Vaswani, 2017)
        encoder_decoder_attention1 = self.encoder_decoder_attention1(norm5, encoder_outputs, encoder_outputs)
        norm6 = self.norm6(add_residual5)
        # feedforward (x2)
        # Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        feedforward5 = self.feedforward5(encoder_decoder_attention1)
        # feedforard6 = self.feedforward6(encoder_decoder_attention1)
        # add and normalize
        # add_residual6
        # norm7
        return output