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
