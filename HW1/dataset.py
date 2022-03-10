import pandas as pd
import numpy as np
import torchaudio
import torch
from torch.utils.data import Dataset, DataLoader

class MusicDataset(Dataset):
    def __init__(self, csv_file, audio_dir, target_sample_rate=44100, mode='train', transformation=True):
        self.df = pd.read_csv(csv_file)
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate
        self.num_samples = self.target_sample_rate * 5
        self.mode = mode
        self.transformation = transformation
        if transformation:
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=target_sample_rate,
                n_fft=1024,
                hop_length=512,
                n_mels=64
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        audio_path = self.get_audio_path(idx)
        signal, sr = torchaudio.load(audio_path)
        signal = self.resample_if_necessary(signal, sr)
        signal = self.mix_down_if_necessary(signal)
        signal = self.cut_if_necessary(signal)
        signal = self.right_pad_if_necessary(signal)
        if self.transformation:
            signal = self.transform(signal)
        if self.mode != 'test':
            score = self.df.loc[idx]['score']
            return signal, score
        else:
            return self.df.loc[idx]['track'], signal

    def get_audio_path(self, idx):
        return self.audio_dir + self.df.loc[idx]['track']

    def cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, :self.num_samples]
        return signal

    def right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding)
        return signal

    def resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal