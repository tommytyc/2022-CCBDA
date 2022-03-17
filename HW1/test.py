import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import MusicDataset
from model import CNNNet, RNNNet

RESNET_SIZE = 101

if __name__ == '__main__':
    test_data = MusicDataset('data/test.csv', 'data/audios/', mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CNNNet().to(device)
    # model = RNNNet(input_size=110080, hidden_size=1024, num_layers=8).to(device)
    # model = ResNet(RESNET_SIZE).to(device)
    model.load_state_dict(torch.load(f'model/CNNNet.pkl'))
    track = []
    pred = []
    with torch.no_grad():
        for step, (track_name, audio) in enumerate(test_dataloader):
            audio = audio.to(device).float()
            # audio_feature = audio_feature.to(device).float()
            # output = model(audio, audio_feature.unsqueeze(1))
            output = model(audio)
            track.append(track_name[0])
            pred.append(output.item())

    df = pd.DataFrame({'track': track, 'score': pred})
    df.to_csv('result/submission.csv', index=False)