import torch
from torch import nn
from Embed import Embed
from Encoder import Encoder
from Decoder import Decoder

class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,out_seq_len, device, max_length, num_heads, dropout):
        super(Summarizer, self).__init__()

        # initialize model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.out_seq_len = out_seq_len
        self.device = device
        self.max_length = max_length
        self.num_heads = num_heads
        self.dropout = dropout

        # initialize embed, decoder, and encoder layers
        self.embed_encoder = Embed(self.input_size, self.hidden_size, self.max_length)
        self.embed_decoder = Embed(self.hidden_size, self.hidden_size, self.out_seq_len)
        self.encoder1 = Encoder(self.hidden_size, self.num_heads, self.dropout)
        self.encoder2 = Encoder(self.hidden_size, self.num_heads, self.dropout)
        self.decoder1 = Decoder(self.hidden_size, self.num_heads, self.dropout)
        self.decoder2 = Decoder(self.hidden_size, self.num_heads, self.dropout)

        # initialize model final linear layer
        self.final_linear = nn.Linear(self.hidden_size, self.output_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_inputs):
        batch_size = enc_inputs.shape[1] # should be 128

        enc_embedding = self.embed_encoder(enc_inputs)
        enc_output = self.encoder_layers(enc_embedding)

        sos_token = enc_inputs[0].unsqueeze(0)  # TODO update this to make this a real SoS token?

        # initial decoder inputs
        dec_initial_input = sos_token
        output_len = 0
        out_seq_len = self.out_seq_len
        output_size = self.output_size
        model_outputs = torch.zeros(out_seq_len, batch_size, output_size)
        dec_inputs = torch.zeros((out_seq_len, batch_size), dtype = torch.int64)
        dec_inputs[0] = dec_initial_input
        attn_mask = torch.ones((out_seq_len, out_seq_len), dtype = torch.bool)
        attn_mask[0] = False

        while output_len < out_seq_len-1:  # TODO Also stop when I predict an <EOS> token?
            dec_embedding  = self.embed_decoder(dec_inputs)
            d_out = self.decoder_layers(dec_embedding, enc_output, attn_mask = attn_mask)
            d_out_attention = d_out[0:output_len+1]
            model_out = self.final_layers(d_out_attention)

            #increment output length
            output_len += 1

            # add predictions for this position to model_outputs
            model_outputs[1:output_len+1] = model_out

            # update dec_inputs
            predictions = torch.argmax(model_out, dim = 2)
            dec_inputs[1:output_len+1] = predictions

            # update attention mask
            attn_mask[output_len] = False

        predicted_words = torch.argmax(model_outputs, dim = 2)
        return predicted_words

    def encoder_layers(self, inputs):
        encoder1 = self.encoder1(inputs)
        # encoder2 = self.encoder2(encoder1)
        output = encoder1
        return output

    def decoder_layers(self, inputs, encoder_outputs, attn_mask):
        decoder1 = self.decoder1(inputs, encoder_outputs, attn_mask)
        # decoder2 = self.decoder2(decoder1, encoder_outputs, attn_mask)
        output = decoder1
        return output

    def final_layers(self, inputs):
        # linear
        final_linear = self.final_linear(inputs)
        #softmax
        softmax = self.softmax(final_linear)
        output = softmax
        return output
