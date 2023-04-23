import torch
from torch import nn

class Summarizer(nn.Module):
    def __init__(self, encoder, decoder,embed, out_seq_len, input_size, emb_size, linear_size, output_size, device, max_length = 1000):
        super(Summarizer, self).__init__()

        # initialize model parameters
        self.device = device
        self.encoder = encoder.to(device)
        self.decoder = decoder.to(device)
        self.embed = embed.to(device)
        self.out_seq_len = out_seq_len

        # initialize model final layers
        self.final_linear = nn.Linear(self.linear_size,output_size)
        self.softmax = nn.Softmax()

    def forward(self, enc_inputs):
        batch_size = enc_inputs.shape[1] #TODO double check this

        enc_embedding = self.embed(enc_inputs)
        enc_output = self.encoder_layers(enc_embedding)

        sos_token = enc_inputs[:, 0].unsqueeze(1)  # TODO update this

        # initial decoder inputs
        dec_input = sos_token
        output_len = 0
        output_size = self.output_size
        model_outputs = torch.zeros(batch_size, seq_len, dec_output_size)

        while output_len < output_size:  # TODO Also stop when I predict an <EOS> token?
            dec_embedding  = self.embed(dec_input)
            dec_output = self.decoder_layers(dec_embedding, enc_output)

            model_outputs = torch.transpose(model_outputs, 0, 1)
            model_outputs[output_len] = dec_output
            model_outputs = torch.transpose(model_outputs, 0, 1)
            output_len += 1
            # take dec_output and get next input
            dec_next_input = torch.ones((batch_size, 1), dtype=torch.int64)
            argmax1 = torch.argmax(dec_output[0]) #TODO delete?
            dec_next_input[0] = torch.argmax(dec_output[0])
            dec_next_input[1] = torch.argmax(dec_output[1])
            dec_input = dec_next_input

        final_linear = self.final_linear(dec_outputs)
        model_output = final_linear
        return model_output

   def decoder_layers(self, inputs):
       encoder1 = self.encoder(inputs)
       encoder2 = self.encoder(encoder1)
       output = encoder2
       return output

   def decoder_layers(self, inputs, encoder_outputs):
       decoder1 = self.decoder(inputs, encoder_outputs)
       decoder2 = self.decoder(decoder1, encoder_outputs)
       output = encoder2
       return output

    def final_linear(self, inputs):
        # linear
        final_linear = self.final_linear(inputs)
        #softmax
        softmax = self.softmax(final_linear)
        output = softmax
        return output
