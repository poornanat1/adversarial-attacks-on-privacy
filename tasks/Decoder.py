# Ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py

import torch
import torch.nn as nn

# from https://github.com/huggingface/transformers/blob/d95045717e1a5bd8ce71223b5b8920e27687dee4/src/transformers/models/t5/modeling_t5.py#L238
class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        # layer norm should always be calculated in float32
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states

class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model
        The encoder consists of a stack of “blocks”, each of which comprises two subcomponents: a self-attention layer
        followed by a small feed-forward network. Layer normalization (Ba et al., 2016) is applied to the input of each 
        subcomponent. We use a simplified version of layer normalization where the activations are only rescaled and no 
        additive bias is applied. After layer normalization, a residual skip connection (He et al., 2016) adds each 
        subcomponents input to its output. Dropout (Srivastava et al., 2014) is applied within the feed-forward network, 
        on the skip connection, on the attention weights, and at the input and output of the entire stack.

        The decoder is similar in structure to the encoder except that it includes a standard attention mechanism 
        after each self-attention layer that attends to the output of the encoder. The self-attention mechanism in 
        the decoder also uses a form of autoregressive or causal self-attention, which only allows the model to 
        attend to past outputs. 
    """

    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super(Decoder, self).__init__()

        # initialize model parameters
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.dropout = dropout

        # initialize model layers
        self.masked_self_attention = nn.Sequential(
            T5LayerNorm(self.hidden_size), 
            torch.nn.MultiheadAttention(self.hidden_size, self.num_heads, add_bias_kv=False, dropout=dropout),
            nn.Dropout(p=dropout)
        )
        
        self.encoder_decoder_attention = nn.Sequential(
            T5LayerNorm(self.hidden_size), 
            torch.nn.MultiheadAttention(self.hidden_size, self.num_heads, add_bias_kv=False, dropout=dropout),
            nn.Dropout(p=dropout)
            )

        # feedforward: Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        self.feedforward = nn.Sequential(
            T5LayerNorm(self.hidden_size), 
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hidden_size, bias=False),
            nn.Dropout(p=dropout)
            )

    def forward(self, inputs, enc_output, attn_mask):
        # multihead self attention
        masked_self_attention, _ = self.masked_self_attention(inputs, inputs, inputs, attn_mask=attn_mask)

        # residual skip connection adds each subcomponent’s input to its output
        skip_connection1 = inputs + masked_self_attention
        
        # encoder-decoder multihead attention
        # In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. (Vaswani, 2017)
        encoder_decoder_attention, _ = self.encoder_decoder_attention(skip_connection1, enc_output, enc_output)

        # residual skip connection adds each subcomponent’s input to its output
        skip_connection2 = skip_connection1 + encoder_decoder_attention
        
        # feed forward network
        feedforward = self.feedforward(skip_connection2)

        # residual skip connection adds each subcomponent’s input to its output
        output = skip_connection2 + feedforward

        return output