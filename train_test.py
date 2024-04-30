import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import itertools
import os
import time
import argparse
import json
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel
from env import AttrDict, build_env
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
from models import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator, feature_loss, generator_loss,\
    discriminator_loss
from DE_test_0429 import MultiEnvelopeDiscriminator
from envelope_pre import Envelope
from utils import plot_spectrogram, scan_checkpoint, load_checkpoint, save_checkpoint

# 환경 설정
device = torch.device('cpu')  # CPU 설정
audio_path = 'path_to_your_audio_file.wav'  # 오디오 파일 경로

# 데이터 로드 설정
training_files = get_dataset_filelist('LJSpeech-1.1/training.txt')[:100]  # LJ 데이터셋의 100개 항목 로드
train_set = MelDataset(training_files, segment_size=8192, n_fft=1024, hop_length=256, win_length=1024, sampling_rate=22050, mel_fmin=0.0, mel_fmax=8000.0)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=1, pin_memory=True)

# 모델 초기화
generator = Generator().to(device)
discriminator = MultiEnvelopeDiscriminator(audio_path).to(device)

# 체크포인트 로드
state_dict_g = load_checkpoint('generator_cp.pth', device)
generator.load_state_dict(state_dict_g)

# 모델 초기화
generator = Generator().to(device)
discriminator = MultiEnvelopeDiscriminator().to(device)

# 최적화 도구 설정
learning_rate = 0.0001
betas = (0.5, 0.999)  # GAN 훈련에 일반적으로 사용되는 베타 값

generator_optimizer = torch.optim.AdamW(generator.parameters(), lr=learning_rate, betas=betas)
discriminator_optimizer = torch.optim.AdamW(discriminator.parameters(), lr=learning_rate, betas=betas)

# 훈련 루프
def train():
    generator.train()
    discriminator.train()

    for epoch in range(10):  # 10 에폭 동안 훈련
        for i, (mel, audio) in enumerate(train_loader):
            mel, audio = mel.to(device), audio.to(device)

            # 생성기를 통해 오디오 생성
            fake_audio = generator(mel)

            # 판별기 업데이트
            discriminator_loss = F.mse_loss(discriminator(fake_audio), discriminator(audio))
            discriminator.zero_grad()
            discriminator_loss.backward()
            discriminator_optimizer.step()

            # 생성기 업데이트
            generator_loss = F.mse_loss(discriminator(fake_audio), torch.ones_like(discriminator(fake_audio)))
            generator.zero_grad()
            generator_loss.backward()
            generator_optimizer.step()

            if i % 10 == 0:
                print(f"Epoch: {epoch}, Step: {i}, D Loss: {discriminator_loss.item()}, G Loss: {generator_loss.item()}")

if __name__ == '__main__':
    train()
