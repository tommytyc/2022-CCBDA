import torch
import torch.nn as nn
from torch.autograd import Variable

dil = 2

class CNNNet(nn.Module):

    def __init__(self):
        super().__init__()
        # 4 conv blocks / flatten / linear / sigmoid
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                stride=1,
                padding=2,
                # dilation=dil
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
                padding=2,
                # dilation=dil
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=2,
                # dilation=dil
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                stride=1,
                padding=2,
                # dilation=dil
            ),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(17920, 1024)
        self.linear2 = nn.Linear(1024, 128)
        self.linear3 = nn.Linear(128, 1)
        self.droupout = nn.Dropout(p=0.4)
        self.sigmoid = nn.Sigmoid()

    def forward(self, audio):
        x = self.conv1(audio)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = torch.tanh(x)
        # x = self.droupout(x)
        x = self.linear2(x)
        x = torch.tanh(x)
        logits = self.linear3(x)
        predictions = self.sigmoid(logits)
        return predictions

class RNNNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNNNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True, dropout=0.2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(hidden_size, int(hidden_size/2))
        self.fc2 = nn.Linear(int(hidden_size/2), int(hidden_size/2))
        self.fc3 = nn.Linear(int(hidden_size/2), 1)
        self.dropout = nn.Dropout(p=0.2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        hidden = None
        x = self.lstm(x, hidden)
        x = torch.tanh(self.fc1(x[0]))
        x = self.dropout(x)
        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x