import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from model import RecurrentAutoEncoder
from tqdm import tqdm
import pickle
from sklearn.metrics import roc_auc_score

DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
EPOCHS = 1000
LR = 2e-5
SEED = 17
LAMBDA = 1.2

def set_seed(seed) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_dataset(path, mode='nolabel'):
    data = pd.read_csv(path)
    seq_len = len(data)
    if mode == 'nolabel':
        data = data[['telemetry']]
        data = data.astype(np.float32).to_numpy()
        data = data.reshape(seq_len, 1)
    else:
        data = data[['telemetry', 'label']]
        data = data.astype(np.float32).to_numpy()
        data = data.reshape(seq_len, 2)
    return torch.tensor(data)

def L1_loss_sum(y_pred, y_true):
    return torch.abs(y_pred - y_true).sum().requires_grad_(True)

def L1_loss(y_pred, y_true):
    return torch.abs(y_pred - y_true).requires_grad_(False)

if __name__ == '__main__':
    sensors = ['A', 'B', 'C', 'D', 'E']
    public_preds_list, private_preds_list = [], []
    # criterion = nn.L1Loss(reduction='sum')
    criterion = L1_loss_sum
    set_seed(SEED)
    for s in sensors:
        print('\033[93m' + f'Training {s}' + '\033[0m')
        normal_dataset = create_dataset(f'data/sensor_{s}_normal.csv')
        public_dataset = create_dataset(f'data/sensor_{s}_public.csv', mode='label')
        private_dataset = create_dataset(f'data/sensor_{s}_private.csv')

        model = RecurrentAutoEncoder(input_size=1).to(DEVICE)
        # model.load_state_dict(torch.load(f'model/{s}.pt'))

        optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
        best_auc = 0
        with tqdm(range(EPOCHS)) as tepoch:
            for epoch in tepoch:
                tepoch.set_description(f"Epoch {epoch}")
                model.train()
                # normal
                data = normal_dataset
                data = data.to(DEVICE)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, data)
                loss.backward()
                optimizer.step()

                model.eval()
                data, label = public_dataset[:, 0].view(-1, 1), public_dataset[:, 1].view(-1, 1)
                data = data.to(DEVICE)
                label = label.to(DEVICE)
                with torch.no_grad():
                    output = model(data)
                    output = L1_loss(output, data)
                    for i in range(len(output)):
                        if label[i] == torch.tensor(0.):
                            output[i] = torch.tensor(0.)
                        else:
                            output[i] *= LAMBDA
                auc = roc_auc_score(label.cpu().numpy(), output.cpu().numpy())
                if auc >= best_auc:
                    best_auc = auc
                    torch.save(model.state_dict(), f'model/RELU_{s}.pt')
                tepoch.set_postfix(loss=loss.item(), auc=auc)
        
        model.eval()
        data, label = public_dataset[:, 0].view(-1, 1), public_dataset[:, 1].view(-1, 1)
        data = data.to(DEVICE)
        with torch.no_grad():
            output = model(data)
            output = L1_loss(output, data)
            for i in range(len(output)):
                if label[i] == torch.tensor(0.):
                    output[i] = torch.tensor(0.)
                else:
                    output[i] *= LAMBDA
            public_err = output.cpu().numpy().squeeze()

        data = private_dataset
        data = data.to(DEVICE)
        with torch.no_grad():
            output = model(data)
            private_err = L1_loss(output, data).cpu().numpy().squeeze()

        public_preds_list.append(public_err)
        private_preds_list.append(private_err)

    pred_df = pd.DataFrame(columns=['id', 'pred'])
    pred = []
    for i in public_preds_list:
        for j in i:
            pred.append(j)
    for i in private_preds_list:
        for j in i:
            pred.append(j)
    pred_df['id'] = [i for i in range(len(pred))]
    pred_df['pred'] = pred
    pred_df.to_csv('pred.csv', index=False)
