import argparse
import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
from dataset import MusicDataset
from model import CNNNet

EPOCHS = 200
LR = 2e-5
BATCH_SIZE = 32
SEED = 17

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_path', type=str, default='train.csv', help='train path')
    parser.add_argument('--audio_dir', type=str, default='audios/', help='audio dir')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='epochs')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def prepare_data(train_path, audio_dir, batch_size):
    train_data = MusicDataset(train_path, audio_dir, mode='train')
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    return train_data, train_dataloader

def train(args):
    set_seed(args.seed)
    train_data, train_dataloader = prepare_data(args.train_path, args.audio_dir, args.batch_size)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CNNNet().to(device)
    optimizer = torch.optim.RAdam(model.parameters(), lr=args.lr)
    loss_func = nn.MSELoss().to(device)
    for epoch in range(args.epochs):
        for step, (batch_x, batch_y) in enumerate(train_dataloader):
            batch_x = batch_x.to(device).float()
            batch_y = batch_y.to(device).float()
            output = model(batch_x)
            loss = loss_func(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 10 == 0:
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.item())
    torch.save(model.state_dict(), f'model/CNNNet.pkl')

if __name__ == '__main__':
    args = set_arg()
    train(args)