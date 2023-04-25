import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, num_heads):
        super().__init__()

        # Define multi-head attention layer with input_size, num_heads attention heads
        self.attn = nn.MultiheadAttention(input_size, num_heads)

        # Define linear layer for output projection
        self.out_fc = nn.Linear(input_size, input_size)

    def forward(self, inputs, mask=None):
        # Permute input tensor to shape (seq_len, batch_size, input_size)
        inputs = inputs.permute(1, 0, 2)

        # Apply multi-head attention to input tensor, with key and value same as input tensor
        # Pass mask to key_padding_mask argument to ignore padding tokens
        attn_output, _ = self.attn(inputs, inputs, inputs, key_padding_mask=mask)

        # Permute output tensor back to shape (batch_size, seq_len, input_size)
        attn_output = attn_output.permute(1, 0, 2)

        # Apply linear projection to output tensor
        outputs = self.out_fc(attn_output)

        return outputs


class FeedForward(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()

        # Define first linear layer with input dimension input_size and output dimension hidden_size
        self.linear1 = nn.Linear(input_size, hidden_size)

        # Define activation function (GELU)
        self.activation = nn.GELU()

        # Define second linear layer with input dimension hidden_size and output dimension input_size
        self.linear2 = nn.Linear(hidden_size, input_size)

    def forward(self, inputs):
        # Apply first linear layer to input tensor
        outputs = self.linear1(inputs)

        # Apply activation function to output of first linear layer
        outputs = self.activation(outputs)

        # Apply second linear layer to output of activation function
        outputs = self.linear2(outputs)

        return outputs


class LayerNorm(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        # Define layer normalization layer with input dimension input_size
        self.norm = nn.LayerNorm(input_size)

    def forward(self, inputs):
        # Apply layer normalization to input tensor
        return self.norm(inputs)


class ResidualConnection(nn.Module):
    def __init__(self, sublayer, input_size, dropout):
        super().__init__()

        # Define residual connection layer with sublayer, layer normalization, and dropout
        self.sublayer = sublayer
        self.norm = LayerNorm(input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs, *args):
        # Save input tensor as residual
        residual = inputs

        # Apply sublayer to input tensor
        outputs = self.sublayer(inputs, *args)

        # Apply dropout to output tensor of sublayer
        outputs = self.dropout(outputs)

        # Add residual and output tensor of sublayer
        outputs = residual + outputs

        # Apply layer normalization to output tensor
        outputs = self.norm(outputs)

        return outputs


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, num_heads, feedforward_size, dropout):
        super().__init__()

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # Define the self-attention and feedforward layers as ModuleList
        self.attention_layers = nn.ModuleList([
            ResidualConnection(MultiHeadAttention(hidden_size, num_heads), hidden_size, dropout)
            for _ in range(num_layers)
        ])

        self.feedforward_layers = nn.ModuleList([
            ResidualConnection(FeedForward(hidden_size, feedforward_size), hidden_size, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, embedded_inputs):
        # Apply the self-attention and feedforward layers to the input
        for i in range(self.num_layers):
            attention_output = self.attention_layers[i](embedded_inputs)  # apply the ith self-attention layer to the input
            feedforward_output = self.feedforward_layers[i](attention_output)  # apply the ith feedforward layer to the output of the self-attention layer
            embedded_inputs = feedforward_output  # update the input with the output of the feedforward layer

        # Return the final output of the encoder
        return embedded_inputs