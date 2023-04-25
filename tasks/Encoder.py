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


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, feedforward_size, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Define multi-head attention layer with input_size, num_heads attention heads
        self.attn = nn.MultiheadAttention(hidden_size, num_heads)
        # Define layer normalization layer with input dimension input_size
        self.norm = T5LayerNorm(hidden_size)
        # Define linear layer for output projection
        self.out_fc = nn.Linear(hidden_size, hidden_size)
        # Define first linear layer with input dimension input_size and output dimension hidden_size
        self.linear1 = nn.Linear(hidden_size, feedforward_size)
        # Define activation function (ReLU)
        self.activation = nn.ReLU()
        # Define second linear layer with input dimension feedforward_size and output dimension input_size
        self.linear2 = nn.Linear(feedforward_size, hidden_size)
        # Define dropout layer with dropout probability
        self.dropout = nn.Dropout(dropout)

    def forward(self, embedded_inputs):
        # For each layer of the encoder
        for i in range(self.num_layers):
            # Compute the self-attention of the input
            self_attn_output, _ = self.attn(embedded_inputs, embedded_inputs, embedded_inputs)
            # Add a residual connection to the input and normalize
            normalized_output = self.norm(embedded_inputs + self.dropout(self_attn_output))
            # Compute the feedforward transformation of the normalized output
            feedforward_output = self.linear2(self.activation(self.linear1(normalized_output)))
            # Add another residual connection to the feedforward output and normalize
            embedded_inputs = self.norm(normalized_output + self.dropout(feedforward_output))
        # Apply the output projection layer and return the final output of the encoder
        return self.out_fc(embedded_inputs)
