import torch
from torch import nn

class Summarizer(nn.Module):
    def __init__(self, input_size, emb_size, linear_size, output_size, device):
        super(Summarizer, self).__init__()
        self.device = device

        # initialize model parameters
        self.input_size = input_size
        self.emb_size = emb_size
        self.linear_size = linear_size
        self.output_size = output_size

        # seed_torch(0) #TODO uncomment?
 
        #define layers
        self.embedding = nn.Embedding(self.input_size, self.emb_size)
        self.linear1 = nn.Linear(in_features = self.emb_size, out_features = self.linear_size)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(in_features = self.linear_size, out_features = self.output_size)
        self.softmax = nn.Softmax(dim=0)
       
    def forward(self, inputs):
        embedding = self.embedding(inputs)
        linear1 = self.linear1(embedding)
        relu1 = self.relu1(linear1)
        linear2 = self.linear2(relu1)
        output = self.softmax(linear2)
        return output
