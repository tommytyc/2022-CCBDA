import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import MusicDataset
from model import CNNNet

if __name__ == '__main__':
    test_data = MusicDataset('test.csv', 'audios/', mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CNNNet().to(device)
    model.load_state_dict(torch.load('model/CNNNet.pkl'))
    track = []
    pred = []
    for step, (track_name, signal) in enumerate(test_dataloader):
        signal = signal.to(device).float()
        output = model(signal)
        track.append(track_name[0])
        pred.append(output.item())

    df = pd.DataFrame({'track': track, 'score': pred})
    df.to_csv('submission.csv', index=False)