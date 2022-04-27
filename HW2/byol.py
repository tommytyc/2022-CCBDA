import copy
import os
import random
import torchvision.models
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from dataset import MRI_Dataset
from tqdm import tqdm
import argparse
from utils import loss_fn, KNN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 500
LR = 2e-5
BATCH_SIZE = 32
SEED = 17

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size=4096, output_size=512):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.LeakyReLU = nn.LeakyReLU()

    def forward(self, x):
        out = self.fc1(x)
        out = self.batch_norm1(out)
        out = self.LeakyReLU(out)
        out = self.fc2(out)
        return out

class EMA():
    def __init__(self, beta):
        super(EMA, self).__init__()
        self.beta = beta
    
    def update(self, target, online):
        for target_param, online_param in zip(target.parameters(), online.parameters()):
            target_weight, online_weight = target_param.data, online_param.data
            if target_weight is not None:
                target_param.data = target_weight * self.beta + online_weight * (1.0 - self.beta)
            else:
                target_param.data = online_weight

class BYOL(nn.Module):
    def __init__(self, moving_avg_decay=0.99):
        super().__init__()
        self.moving_avg_decay = moving_avg_decay
        self.online_encoder = torchvision.models.resnet50(pretrained=False)
        # self.online_encoder.fc = nn.Linear(self.online_encoder.fc.in_features, 512)
        self.online_projector = MLP(1000, 1024, 512)
        self.ema = EMA(moving_avg_decay)
        self.online_predictor = MLP(512, 4096, 512)
        self.target_encoder = None
        self.target_projector = None

    def update_ema(self):
        self.ema.update(self.target_encoder, self.online_encoder)

    def inference(self, x):
        x = self.online_encoder(x)
        x = self.online_projector(x)
        return x

    def forward(self, img1, img2):
        online_proj_1 = self.online_encoder(img1)
        online_proj_1 = self.online_projector(online_proj_1)
        online_proj_2 = self.online_encoder(img2)
        online_proj_2 = self.online_projector(online_proj_2)
        online_pred_1 = self.online_predictor(online_proj_1)
        online_pred_2 = self.online_predictor(online_proj_2)
        with torch.no_grad():
            self.target_encoder = copy.deepcopy(self.online_encoder)
            self.target_projector = copy.deepcopy(self.online_projector)
            for p in self.target_encoder.parameters():
                p.requires_grad = False
            for p in self.target_projector.parameters():
                p.requires_grad = False
            target_proj_1 = self.target_encoder(img1)
            target_proj_1 = self.target_projector(target_proj_1)
            target_proj_2 = self.target_encoder(img2)
            target_proj_2 = self.target_projector(target_proj_2)
            target_proj_1.detach_()
            target_proj_2.detach_()
        
        loss_one = loss_fn(online_pred_1, target_proj_2.detach())
        loss_two = loss_fn(online_pred_2, target_proj_1.detach())
        loss = (loss_one + loss_two).mean()
        
        return loss
        
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_imgdir', type=str, default='data/unlabeled/', help='train image dir path')
    parser.add_argument('--test_imgdir', type=str, default='data/test/', help='test image dir path')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='epochs')
    parser.add_argument('--lr', type=float, default=LR, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='batch size')
    parser.add_argument('--seed', type=int, default=SEED, help='seed')
    parser.add_argument('--pretrained_path', type=str, help='pretrained model path')
    return parser.parse_args()

def train(args):
    set_seed(args.seed)
    train_dataset = MRI_Dataset(args.train_imgdir)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    learner = BYOL().to(DEVICE)
    if args.pretrained_path:
        learner.online_encoder.load_state_dict(torch.load(args.pretrained_path))
    optimizer = torch.optim.AdamW(learner.parameters(), lr=args.lr)
    with tqdm(range(args.epochs)) as tepoch:
        for epoch in tepoch:
            learner.train()
            tepoch.set_description(f"Epoch {epoch}")
            for img1, img2 in train_loader:
                img1, img2 = img1.to(DEVICE), img2.to(DEVICE)
                loss = learner(img1, img2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                learner.update_ema()
            
            torch.save({
                'resnet': learner.online_encoder.state_dict(),
                'proj': learner.online_projector.state_dict(),
                'pred': learner.online_predictor.state_dict()
            }, 'BYOL_encoder.pt')
            learner.eval()
            acc = test(args)
            tepoch.set_postfix(acc=acc)

def test(args):
    # encoder = torchvision.models.resnet50(pretrained=False)
    # encoder.fc = nn.Linear(encoder.fc.in_features, 512)
    encoder = BYOL()
    encoder.online_encoder.load_state_dict(torch.load('BYOL_encoder.pt', map_location=DEVICE)['resnet'])
    encoder.online_projector.load_state_dict(torch.load('BYOL_encoder.pt', map_location=DEVICE)['proj'])
    encoder.online_predictor.load_state_dict(torch.load('BYOL_encoder.pt', map_location=DEVICE)['pred'])
    # encoder.load_state_dict(torch.load('resnet_encoder.pt'))
    encoder.to(DEVICE)
    encoder.eval()
    test_dataset_list = [MRI_Dataset(os.path.join(args.test_imgdir, f'{i}/')) for i in range(4)]
    test_loader_list = [DataLoader(dataset, batch_size=1, shuffle=False) for dataset in test_dataset_list]
    embedding, classes = [], []
    for i, test_loader in enumerate(test_loader_list):
        classes.append(torch.ones(len(test_loader)) * i)
        for img1, _ in test_loader:
            img1 = img1.to(DEVICE)
            embedding.append(encoder.inference(img1))
    embedding = torch.cat(embedding, dim=0)
    classes = torch.cat(classes, dim=0)
    acc = KNN(embedding, classes, batch_size=16)
    print(f'Accuracy: {acc}')
    # return acc

if __name__ == "__main__":
    args = set_arg()
    # train(args)
    test(args)