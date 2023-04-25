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
    def __init__(self, hidden_size, num_heads, feedforward_size, dropout):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.feedforward_size = feedforward_size

        # Define self-attention layer
        self.self_attn = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm1 = T5LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        # Define feedforward network
        self.feedforward = nn.Sequential(
            T5LayerNorm(hidden_size),
            nn.Linear(hidden_size, feedforward_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

    def forward(self, embedded_inputs):
        # Apply layer normalization and dropout
        norm_input = self.norm1(self.dropout1(embedded_inputs))

        # Apply self-attention layer
        attn_output, _ = self.self_attn(norm_input, norm_input, norm_input)
        attn_output = self.dropout2(attn_output)

        # add residual skip connection
        embedded_inputs = embedded_inputs + attn_output
        embedded_inputs = self.dropout3(embedded_inputs)

        # Apply feedforward network
        feedforward_output = self.feedforward(embedded_inputs)

        # add residual skip connection
        encoded_output = embedded_inputs + feedforward_output
        encoded_output = self.dropout4(encoded_output)

        return encoded_output
