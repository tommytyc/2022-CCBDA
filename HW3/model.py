import torch
import torch.nn as nn
import torch.nn.functional as F

class RecurrentAutoEncoder(nn.Module):
    class Encoder(nn.Module):
        def __init__(self, input_size, hidden_size=128, dropout=0.1):
            super(RecurrentAutoEncoder.Encoder, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.dropout = dropout
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_size, hidden_size)
        
        def forward(self, x):
            x, _ = self.rnn(x)
            x = self.fc(x)
            return F.leaky_relu(x)

    class Decoder(nn.Module):
        def __init__(self, input_size, hidden_size=128, dropout=0):
            super(RecurrentAutoEncoder.Decoder, self).__init__()
            self.input_size = input_size
            self.hidden_size = hidden_size
            self.dropout = dropout
            self.rnn = nn.LSTM(input_size=hidden_size, hidden_size=input_size, dropout=dropout, batch_first=True)
            self.fc = nn.Linear(hidden_size, hidden_size)

        def forward(self, x):
            x = F.leaky_relu(self.fc(x))
            x, _ = self.rnn(x)
            return x.squeeze().view(-1, self.input_size)
    
    def __init__(self, input_size, hidden_size=256, dropout=0):
        super(RecurrentAutoEncoder, self).__init__()
        self.encoder = self.Encoder(input_size, hidden_size, dropout)
        self.decoder = self.Decoder(input_size, hidden_size, dropout)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x