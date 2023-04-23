import random

import torch
import torch.nn as nn
import torch.optim as optim


class Embed(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Embed, self).__init__()

        # initialize model parameters
        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type

        # initialize model layers
        self.embedding = nn.Embedding(self.input_size, self.emb_size)
        self.posembedding = nn.Embedding(self.max_length, self.emb_size)



    def forward(self, inputs):
        token_embedding = self.embedding(inputs)
        token_embedding = torch.transpose(token_embedding, 0, 1)
        position = torch.arange(self.max_length)  # .unsqueeze(1)
        pos_embedding = self.posembedding(position)
        embeddings = torch.add(token_embedding, pos_embedding)
        embeddings = torch.transpose(embeddings, 0, 1)
        return embeddings
        return output