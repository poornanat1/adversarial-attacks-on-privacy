# Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py
# Ref: https://opacus.ai/api/dp_multihead_attention.html

import torch
import torch.nn as nn
from Encoder import T5LayerNorm

from opacus.layers.dp_multihead_attention import DPMultiheadAttention

class Decoder(nn.Module):
    """ 
        The decoder is similar in structure to the encoder except that it includes a standard attention mechanism 
        after each self-attention layer that attends to the output of the encoder. The self-attention mechanism in 
        the decoder also uses a form of autoregressive or causal self-attention, which only allows the model to 
        attend to past outputs. 
    """

    def __init__(self, hidden_size, num_heads, feedforward_size, device, dropout=0.1, dpsgd=False):
        super(Decoder, self).__init__()

        # initialize model parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout
        self.feedforward_size = feedforward_size
        self.device = device

        # Self-attention subcomponent
        self.norm_self_attn = T5LayerNorm(self.hidden_size)
        if dpsgd:
            self.masked_self_attn = DPMultiheadAttention(self.hidden_size, self.num_heads, dropout=self.dropout)
        else:
            self.masked_self_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=self.dropout)
        self.dropout_self_attn = nn.Dropout(p=self.dropout)
        
        # Encoder-decoder attention subcomponent
        self.norm_enc_dec_attn = T5LayerNorm(self.hidden_size)
        self.norm_enc_dec_attn2 = T5LayerNorm(self.hidden_size)
        if dpsgd:
            self.encoder_decoder_attn = DPMultiheadAttention(self.hidden_size, self.num_heads, dropout=self.dropout)
        else:
            self.encoder_decoder_attn = nn.MultiheadAttention(self.hidden_size, self.num_heads, dropout=self.dropout)
        self.dropout_enc_dec_attn = nn.Dropout(p=self.dropout)

        # Feedforward network: Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        self.feedforward = nn.Sequential(
            T5LayerNorm(self.hidden_size), 
            nn.Linear(self.hidden_size, self.feedforward_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.feedforward_size, self.hidden_size),
            nn.Dropout(p=self.dropout)
            )

    def forward(self, inputs, enc_output, attn_mask, padding_mask):
        torch.cuda.empty_cache()
        padding_mask=padding_mask.transpose(0, 1)
        # multihead self attention
        inputs = self.norm_self_attn(inputs).to(self.device)
        masked_self_attn, _ = self.masked_self_attn(inputs, inputs, inputs, attn_mask=attn_mask.to(self.device))
        masked_self_attn = self.dropout_self_attn(masked_self_attn)

        # residual skip connection adds each subcomponent’s input to its output
        skip_connection1 = inputs + masked_self_attn
        
        # encoder-decoder multihead attention
        # In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. (Vaswani, 2017)
        skip_connection1 = self.norm_enc_dec_attn(skip_connection1)
        enc_output = self.norm_enc_dec_attn2(enc_output)
        encoder_decoder_attn, _ = self.encoder_decoder_attn(skip_connection1, enc_output, enc_output, key_padding_mask=padding_mask)
        encoder_decoder_attn = self.dropout_enc_dec_attn(encoder_decoder_attn)

        # residual skip connection adds each subcomponent’s input to its output
        skip_connection2 = skip_connection1 + encoder_decoder_attn
        
        # feed forward network
        feedforward = self.feedforward(skip_connection2)

        # residual skip connection adds each subcomponent’s input to its output
        output = skip_connection2 + feedforward

        return output