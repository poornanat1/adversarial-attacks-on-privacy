import torch
import torch.nn as nn

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.2):
        super(Decoder, self).__init__()

        # initialize model parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # initialize model layers
        self.masked_self_attention = torch.nn.MultiheadAttention(self.hidden_size, self.num_heads) #TODO add dropout parameter here?
        self.norm1 = nn.LayerNorm(self.hidden_size)
        self.encoder_decoder_attention = torch.nn.MultiheadAttention(self.hidden_size, self.num_heads)
        self.norm2 = nn.LayerNorm(self.hidden_size)
        self.feedforward1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.feedforward2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.norm3 = nn.LayerNorm(self.hidden_size)

    # TODO Add drop out layers
    # - page 4 of T5 paper: Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack.
    # - Vaswani, 2017: We apply dropout [33] to the output of each sub-layer, before it is added to thesub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
    # positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
    # Pdrop = 0.1.

    def forward(self, inputs, enc_output, attn_mask):
        # multihead self attention
        masked_self_attention = self.masked_self_attention(inputs, inputs, inputs, attn_mask= attn_mask)

        # add and normalize #TODO check implementation per page 4 of T5 paper: After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent’s input to its output.
        add_residual1 = inputs + masked_self_attention[0]
        norm1 = self.norm1(add_residual1)

        # encoder-decoder multihead attention
        # In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. (Vaswani, 2017)
        encoder_decoder_attention = self.encoder_decoder_attention(norm1, enc_output, enc_output)

        # add and normalize
        add_residual2 = norm1 + encoder_decoder_attention[0]
        norm2 = self.norm2(add_residual2)

        # feed forward: Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        feedforward1 = self.feedforward1(norm2)
        relu = self.relu(feedforward1)
        feedforward2 = self.feedforward2(relu)

        add_residual3 = norm2 + feedforward2
        norm3 = self.norm3(add_residual3)

        output = norm3
        return output