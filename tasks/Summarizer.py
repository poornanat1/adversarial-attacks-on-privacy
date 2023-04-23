import torch
from torch import nn

class Summarizer(nn.Module):
    def __init__(self, input_size, emb_size, linear_size, output_size, device, max_length = 1000):
        super(Summarizer, self).__init__()
        self.device = device

        # initialize model parameters
        self.input_size = input_size
        self.emb_size = emb_size
        self.linear_size = linear_size
        self.output_size = output_size
        self.max_length = max_length

        # seed_torch(0) #TODO uncomment?

        # init embedding layers
        self.embedding = nn.Embedding(self.input_size, self.emb_size)
        self.posembedding = nn.Embedding(self.max_length, self.emb_size)

        # init encoder 1
        self.self_attention1 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm1 = nn.LayerNorm(self.linear_size)
        self.feedforward1 = nn.Linear(self.emb_size, self.linear_size)
        self.feedforward2 = nn.Linear(self.emb_size, self.linear_size)
        self.norm2 = nn.LayerNorm(self.linear_size)

        # init encoder 2
        self.self_attention2 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm3 = nn.LayerNorm(self.linear_size)
        self.feedforward3 = nn.Linear(self.emb_size, self.linear_size)
        self.feedforward4 = nn.Linear(self.emb_size, self.linear_size)
        self.norm4 = nn.LayerNorm(self.linear_size)

        # init decoder 1
        self.self_attention3 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm5 = nn.LayerNorm(self.linear_size)
        self.encoder_decoder_attention1 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm6 = nn.LayerNorm(self.linear_size)
        self.feedforward5 = nn.Linear(self.emb_size, self.linear_size)
        self.feedforward6 = nn.Linear(self.emb_size, self.linear_size)
        self.norm7 = nn.LayerNorm(self.linear_size)

        # init decoder 2
        self.self_attention4 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm8 = nn.LayerNorm(self.linear_size)
        self.encoder_decoder_attention2 = torch.nn.MultiheadAttention(self.emb_size, num_heads=2)
        self.norm9 = nn.LayerNorm(self.linear_size)
        self.feedforward7 = nn.Linear(self.emb_size, self.linear_size)
        self.feedforward8 = nn.Linear(self.emb_size, self.linear_size)
        self.norm10 = nn.LayerNorm(self.linear_size)

        # final linear
        self.final_linear = nn.Linear(self.linear_size,output_size)
        self.softmax = nn.Softmax()

    #TODO Add drop out
    # - p4. Dropout (Srivastava et al., 2014) is applied within the feed-forward network, on the skip connection, on the attention weights, and at the input and output of the entire stack.
    # - We apply dropout [33] to the output of each sub-layer, before it is added to thesub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the
    # positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of
    # Pdrop = 0.1.(Vaswani, 2017)

    def forward(self, inputs):
        embedding = self.embed(inputs)
        encoder1 = self.encoder1(embedding)
        # encoder2 = self.encoder2(encoder1)
        # decoder1 = self.decoder1(inputs, encoder2)

        # decoder_input = 0 #TODO update
        # decoder1 = self.decoder1(decoder_input, encoder1) #TODO Treat like Seq2Seq and use output sequence as input to decoder?
        # # decoder2 = self.decoder2(decoder1, encoder2)

        # final_linear = self.final_linear(encoder1)
        output = encoder1
        return output

    def embed(self, inputs):
        token_embedding = self.embedding(inputs)
        token_embedding = torch.transpose(token_embedding,0,1)
        position = torch.arange(self.max_length)  # .unsqueeze(1)
        pos_embedding = self.posembedding(position)
        embeddings = torch.add(token_embedding, pos_embedding)
        embeddings = torch.transpose(embeddings,0,1)
        return embeddings

    def feedforward(self, inputs):
        # Two linear transformations with a ReLU activation in between (Vaswani, 2017)
        pass

    def encoder1(self, inputs):
        # multihead self attention
        self_attention1 = self.self_attention1(inputs, inputs, inputs)
        # add and normalize #TODO p4. After layer normalization, a residual skip connection (He et al., 2016) adds each subcomponent’s input to its output.
        add_residual1 = inputs + self_attention1[0]
        norm1 = self.norm1(add_residual1)
        #feed forward (x2)
        feedforward1 = self.feedforward1(norm1)
        # feedforward2 = self.feedforward2(norm1) #TODO uncomment
        # add and normalize
        add_residual2 = norm1 + feedforward1
        norm2 = self.norm2(add_residual2)
        output = norm2
        return output

    def encoder2(self, inputs):
        # same as encoder 1
        return output

    def decoder1(self, inputs, encoder_outputs):
        # multihead self attention
        self_attention3 = self.self_attention3(inputs, inputs, inputs)
        # add and normalize
        add_residual5 = inputs + self_attention3[0]
        norm5 = self.norm5(add_residual5)
        # encoder-decoder multihead attention
        # In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. (Vaswani, 2017)
        encoder_decoder_attention1 = self.encoder_decoder_attention1(norm5, encoder_outputs, encoder_outputs)
        norm6 = self.norm6(add_residual5)
        # feedforward (x2)
        feedforward5 = self.feedforward5(encoder_decoder_attention1)
        # feedforard6 = self.feedforward6(encoder_decoder_attention1)
        # add and normalize
        # add_residual6
        # norm7
        return output

    def decoder2(self, inputs):
        # same as decoder 1
        return output

    def final_linear(self, inputs):
        # linear
        final_linear = self.final_linear(inputs)
        #softmax
        softmax = self.softmax(final_linear)
        output = softmax
        return output
