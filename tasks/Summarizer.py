import torch
from torch import nn

class Summarizer(nn.Module):
    def __init__(self, input_size, emb_size, linear_size, device):
        super(Summarizer, self).__init__()
        self.device = device

        # initialize model paramters
        self.input_size = input_size
        self.emb_size = emb_size
        self.linear_size = linear_size

        # seed_torch(0) #TODO uncomment?
 
        #define layers
        self.embedding = nn.Embedding(self.input_size+1, self.emb_size)
        self.linear1 = nn.Linear(self.emb_size, self.linear_size)
        self.relu1 = nn.ReLU()
       
    def forward(self, inputs):
        embedding = self.embedding(inputs)
        linear1 = self.linear1(embedding)
        relu1 = self.relu1(linear1)
        output = relu1
        return output
