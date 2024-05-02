from ED_test_0429_2 import MultiEnvelopeDiscriminator, Generator, generator_loss, discriminator_loss, feature_loss
import torch
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
# 사용자 코드 파일에서 필요한 클래스 및 함수 임포트
from meldataset import mel_spectrogram, MelDataset, load_wav
import json

class HParams:
    def __init__(self, config):
        self.resblock = config['resblock']
        self.upsample_rates = config['upsample_rates']
        self.upsample_kernel_sizes = config['upsample_kernel_sizes']
        self.upsample_initial_channel = config['upsample_initial_channel']
        self.resblock_kernel_sizes = config['resblock_kernel_sizes']
        self.resblock_dilation_sizes = config['resblock_dilation_sizes']

# JSON 파일을 읽고 하이퍼파라미터 객체 생성
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return HParams(config)

class VoiceDataset(Dataset):
    def __init__(self, directory, transform=None, limit=100):
        self.directory = directory
        self.filenames = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.wav')][:limit]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        audio_path = self.filenames[idx]
        wav = load_wav(audio_path)
        mel = mel_spectrogram(y = wav.squeeze(1), n_fft=1024, num_mels=80, sampling_rate=22050, hop_size=256, win_size=1024, fmin=0, fmax=8000)  # 멜 스펙트로그램으로 변경
        if self.transform:
            mel = self.transform(mel)
        return mel, wav  # wav를 반환하도록 수정
    
discriminator = MultiEnvelopeDiscriminator()

def train_gan(generator, discriminator, dataloader, epochs=10):
    device = torch.device('cpu')
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for i, (mel, audio) in enumerate(dataloader):
            real_audio = Variable(audio.float()).to(device)
            mel = Variable(mel.float()).to(device)

            fake_audio = generator(mel)
            real_scores, fake_scores, _, _ = discriminator(real_audio, fake_audio)
            real_scores, fake_scores = torch.sigmoid(real_scores), torch.sigmoid(fake_scores)  # 확률 변환 추가

            d_optimizer.zero_grad()
            real_loss = criterion(real_scores, torch.ones_like(real_scores))
            fake_loss = criterion(fake_scores, torch.zeros_like(fake_scores))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            g_optimizer.zero_grad()
            g_loss = criterion(fake_scores, torch.ones_like(fake_scores))
            g_loss.backward()
            g_optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Step [{i+1}/{len(dataloader)}], D Loss: {d_loss.item()}, G Loss: {g_loss.item()}')

class HParams:
    def __init__(self, config):
        self.resblock = config['resblock']
        self.upsample_rates = config['upsample_rates']
        self.upsample_kernel_sizes = config['upsample_kernel_sizes']
        self.upsample_initial_channel = config['upsample_initial_channel']
        self.resblock_kernel_sizes = config['resblock_kernel_sizes']
        self.resblock_dilation_sizes = config['resblock_dilation_sizes']

def main():
    config_path = 'C:/aproject/MED/config_v1.json'  # 설정 파일 경로
    hparams = load_config(config_path)  # 하이퍼파라미터 로드

    # 데이터셋 및 데이터 로더 설정
    dataset_path = "C:/aproject/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/wavs"
    dataset = VoiceDataset(directory=dataset_path)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

    # 모델 인스턴스화
    generator = Generator(hparams)
    discriminator = MultiEnvelopeDiscriminator()

    # GAN 트레이닝 함수 호출
    train_gan(generator, discriminator, dataloader)

if __name__ == '__main__':
    main()