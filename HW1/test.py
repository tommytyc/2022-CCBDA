import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from dataset import MusicDataset
from model import CNNNet, RNNNet

def set_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_path', type=str, default='data/test.csv', help='test path')
    parser.add_argument('--audio_dir', type=str, default='data/audios/', help='audio dir')
    parser.add_argument('--filename', type=str, nargs='+', help='filename list')
    return parser.parse_args()

if __name__ == '__main__':
    args = set_arg()
    if args.filename:
        test_data = MusicDataset(None, None, args.filename, mode='TA_test')
    else:
        test_data = MusicDataset(args.test_path, args.audio_dir, mode='test')
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=False)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = CNNNet().to(device)
    # model = RNNNet(input_size=110080, hidden_size=1024, num_layers=8).to(device)
    model.load_state_dict(torch.load(f'model/CNNNet.pkl'))
    track = []
    pred = []
    with torch.no_grad():
        for step, (track_name, audio) in enumerate(test_dataloader):
            audio = audio.to(device).float()
            output = model(audio)
            track.append(track_name[0])
            pred.append(output.item())

    df = pd.DataFrame({'track': track, 'score': pred})
    df.to_csv('result/submission.csv', index=False)