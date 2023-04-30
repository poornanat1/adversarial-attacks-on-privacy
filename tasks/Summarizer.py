import torch
from torch import nn
from Embed import Embed
from Encoder import Encoder
from Decoder import Decoder


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, max_length, num_heads, dropout, model_type="base"):
        super(Summarizer, self).__init__()

        # initialize model parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.device = device
        self.max_length = max_length
        self.num_heads = num_heads
        self.dropout = dropout
        self.dpsgd = True if model_type=="dp-sgd" else False

        # initialize embedding layers
        self.embed_encoder = Embed(self.input_size, self.hidden_size, self.max_length)
        self.embed_decoder = Embed(self.output_size, self.hidden_size, self.max_length)

        # initialize encoder layers
        self.enc_dropout_input = nn.Dropout(p=self.dropout)
        self.encoder1 = Encoder(hidden_size=self.hidden_size, 
                                num_heads=self.num_heads,
                                feedforward_size=self.hidden_size * 4, 
                                dropout=self.dropout,
                                dpsgd=self.dpsgd)
        self.encoder2 = Encoder(hidden_size=self.hidden_size, 
                                num_heads=self.num_heads,
                                feedforward_size=self.hidden_size * 4, 
                                dropout=self.dropout,
                                dpsgd=self.dpsgd)
        self.enc_dropout_output = nn.Dropout(p=self.dropout)
        
        # initialize stack of decoder layers
        self.dec_dropout_input = nn.Dropout(p=self.dropout)
        self.decoder1 = Decoder(hidden_size=self.hidden_size, 
                                num_heads=self.num_heads, 
                                feedforward_size=self.hidden_size * 4,
                                dropout=self.dropout,
                                dpsgd=self.dpsgd)
        self.decoder2 = Decoder(hidden_size=self.hidden_size, 
                                num_heads=self.num_heads, 
                                feedforward_size=self.hidden_size * 4,
                                dropout=self.dropout,
                                dpsgd=self.dpsgd)
        self.dec_dropout_output = nn.Dropout(p=self.dropout)

        # initialize model final linear layer
        self.final_linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, enc_inputs):
        batch_size = enc_inputs.shape[1]
        enc_embedding = self.embed_encoder(enc_inputs)
        enc_output = self.encoder_layers(enc_embedding)

        sos_token = 101
        eos_token = 102

        # initial decoder inputs
        model_outputs = torch.zeros(self.max_length, batch_size, self.output_size)
        dec_inputs = torch.zeros((self.max_length, batch_size), dtype=torch.int64)
        dec_inputs[0] = sos_token
        
        attn_mask = torch.logical_not(torch.tril(torch.ones((self.max_length, self.max_length), dtype=torch.bool)))

        # ended_sequences = set()
        
        for i in range(self.max_length):  
            # update attention mask
            attn_mask[i] = False

            dec_embedding = self.embed_decoder(dec_inputs)
            d_out = self.decoder_layers(dec_embedding, enc_output, attn_mask=attn_mask)
            d_out_attention = d_out[1:i+1]
            model_out = self.final_layers(d_out_attention)

            # add predictions for this position to model_outputs
            model_outputs[1:i+1] = model_out

            # update dec_inputs
            predictions = torch.argmax(model_out, dim=2)
            dec_inputs[1:i+1] = predictions

            # # If any predicted token is the EOS token, stop generating output for that sequence
            # if eos_token in predictions:
            #     break_indices = torch.where(predictions == eos_token)[0]
            #     for j in break_indices:
            #         # If the sequence has already ended, skip it
            #         if j in ended_sequences:
            #             continue
            #         # Mark the sequence as ended
            #         ended_sequences.add(j)
            #         # Trim the output sequence for the ended sequence
            #         dec_inputs[j,] = dec_inputs[j,:i+1]

        return model_outputs

    def encoder_layers(self, inputs):
        # dropout at the input of the entire stack
        inputs = self.enc_dropout_input(inputs)

        encoder1 = self.encoder1(inputs)
        encoder2 = self.encoder2(encoder1)

        # dropout at the output of the entire stack
        output = self.enc_dropout_output(encoder2)
        return output

    def decoder_layers(self, inputs, encoder_outputs, attn_mask):
        # dropout at the input of the entire stack
        inputs = self.dec_dropout_input(inputs)

        # stack of decoder blocks
        decoder1 = self.decoder1(inputs, encoder_outputs, attn_mask)
        decoder2 = self.decoder2(decoder1, encoder_outputs, attn_mask)

        # dropout at the output of the entire stack
        output = self.dec_dropout_output(decoder2)
        return output

    def final_layers(self, inputs):
        # linear
        final_linear = self.final_linear(inputs)
        return final_linear
