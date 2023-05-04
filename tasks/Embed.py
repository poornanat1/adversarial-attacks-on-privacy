import torch
import torch.nn as nn

class Embed(nn.Module):
    """ The Encoder module of the Seq2Seq model
    """

    def __init__(self, input_size, hidden_size, max_length, device):
        super(Embed, self).__init__()

        # initialize model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.device = device

        # initialize model layers
        self.embedding = nn.Embedding(self.input_size, self.hidden_size)
        self.posembedding = nn.Embedding(self.max_length, self.hidden_size)

    def forward(self, inputs):
        token_embedding = self.embedding(inputs.clone().to(self.device))
        token_embedding = torch.transpose(token_embedding, 0, 1)
        position = torch.arange(self.max_length).to(self.device)
        pos_embedding = self.posembedding(position)
        embeddings = torch.add(token_embedding, pos_embedding)
        output = torch.transpose(embeddings, 0, 1)
        return output