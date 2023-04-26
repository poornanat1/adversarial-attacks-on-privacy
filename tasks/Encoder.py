import torch
import torch.nn as nn

# T5LayerNorm function from: https://github.com/huggingface/transformers/blob/d95045717e1a5bd8ce71223b5b8920e27687dee4/src/transformers/models/t5/modeling_t5.py#L238
# Please leave citation in because this function is taken directly from T5 source code.
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


class Encoder(nn.Module):
    '''
        The encoder consists of a stack of “blocks”, each of which comprises two subcomponents: a self-attention layer
        followed by a small feed-forward network. Layer normalization (Ba et al., 2016) is applied to the input of each 
        subcomponent. We use a simplified version of layer normalization where the activations are only rescaled and no 
        additive bias is applied. After layer normalization, a residual skip connection (He et al., 2016) adds each 
        subcomponents input to its output. Dropout (Srivastava et al., 2014) is applied within the feed-forward network, 
        on the skip connection, on the attention weights, and at the input and output of the entire stack.
    '''
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout=0.1):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feedforward_size = feedforward_size
        self.dropout=dropout

        # Define self-attention layer
        self.norm_attn = T5LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads, dropout=self.dropout)
        self.dropout_attn = nn.Dropout(p=self.dropout)

        # Define feedforward network
        self.norm_ff = T5LayerNorm(hidden_size)
        self.feedforward = nn.Sequential(
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),
            nn.Linear(feedforward_size, hidden_size),
            nn.Dropout(p=self.dropout)
        )

    def forward(self, embedded_inputs):
        # self-attention layer
        embedded_inputs = self.norm_attn(embedded_inputs)
        attn_output, _ = self.self_attn(embedded_inputs, embedded_inputs, embedded_inputs)
        attn_output = self.dropout_attn(attn_output)

        # add residual skip connection
        skip_connection = embedded_inputs + attn_output
        
        # feedforward network
        skip_connection = self.norm_ff(skip_connection)
        feedforward = self.feedforward(skip_connection)

        # add residual skip connection
        output = skip_connection + feedforward

        return output
