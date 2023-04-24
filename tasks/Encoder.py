import torch
import torch.nn as nn

class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.2):
        super(Encoder, self).__init__()

        # initialize model parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # initialize model layers
        self.self_attention = torch.nn.MultiheadAttention(self.hidden_size, self.num_heads) #TODO add dropout parameter here?
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.feedforward1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.feedforward2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.norm2 = nn.LayerNorm(self.hidden_size)

    # TODO Add drop out layers
    # - page 4 of T5 paper: Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack.
    # - Vaswani, 2017: We apply dropout [33] to the output of each sub-layer, before it is added to thesub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
    # positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
    # Pdrop = 0.1.

    def forward(self, inputs):
        # multihead self attention
        self_attention = self.self_attention(inputs, inputs, inputs)

        # add and normalize #TODO check implementation per page 4 of T5 paper: After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent’s input to its output.
        add_residual1 = inputs + self_attention[0]
        norm1 = self.norm1(add_residual1)
        # feed forward: Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        feedforward1 = self.feedforward1(norm1)
        relu = self.relu(feedforward1)
        feedforward2 = self.feedforward2(relu)

        # add and normalize
        add_residual2 = norm1 + feedforward2
        norm2 = self.norm2(add_residual2)
        output = norm2
        return output
