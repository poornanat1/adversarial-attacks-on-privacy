# Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

import torch
import torch.nn as nn

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(Decoder, self).__init__()

        # initialize model parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # initialize model layers
        self.dropout_input = nn.Dropout(p=dropout)
        self.masked_self_attention = nn.Sequential(nn.LayerNorm(self.hidden_size),
                                                   torch.nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=dropout),
                                                   nn.Dropout(p=dropout)
                                                   )
        
        self.encoder_decoder_attention = nn.Sequential(nn.LayerNorm(self.hidden_size),
                                                       torch.nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=dropout),
                                                       nn.Dropout(p=dropout)
                                                       )

        # feedforward: Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        self.feedforward = nn.Sequential(nn.LayerNorm(self.hidden_size),
                                         nn.Linear(self.hidden_size, self.hidden_size),
                                         nn.ReLU(),
                                         nn.Dropout(p=dropout),
                                         nn.Linear(self.hidden_size, self.hidden_size)
                                         )
        self.norm_output = nn.LayerNorm(self.hidden_size)
        self.dropout_output = nn.Dropout(p=dropout)

    def forward(self, inputs, enc_output, attn_mask):
        # dropout at the input of the entire stack
        hidden_states = self.dropout_input(inputs)

        # multihead self attention
        masked_self_attention, _ = self.masked_self_attention(hidden_states, hidden_states, hidden_states, attn_mask=attn_mask)

        # residual skip connection adds each subcomponent’s input to its output
        skip_connection1 = hidden_states + masked_self_attention
        
        # encoder-decoder multihead attention
        # In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. (Vaswani, 2017)
        encoder_decoder_attention, _ = self.encoder_decoder_attention(skip_connection1, enc_output, enc_output)

        # residual skip connection adds each subcomponent’s input to its output
        skip_connection2 = skip_connection1 + encoder_decoder_attention
        
        # feed forward network
        feedforward = self.feedforward(skip_connection2)

        # residual skip connection adds each subcomponent’s input to its output
        skip_connection3 = skip_connection2 + feedforward

        # normalization and dropout at the output of the entire stack
        output = self.dropout_output(self.norm_output(skip_connection3))

        return output