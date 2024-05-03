from MED_final import MultiEnvelopeDiscriminator, Generator, generator_loss, discriminator_loss, feature_loss, MultiPeriodDiscriminator
from meldataset import MelDataset, mel_spectrogram, get_dataset_filelist
import os
import random
import json
import torch
from torch.utils.data import DataLoader


class HyperParams:
    def __init__(self, config):
        self.resblock = config['resblock']
        self.upsample_rates = config['upsample_rates']
        self.upsample_kernel_sizes = config['upsample_kernel_sizes']
        self.upsample_initial_channel = config['upsample_initial_channel']
        self.resblock_kernel_sizes = config['resblock_kernel_sizes']
        self.resblock_dilation_sizes = config['resblock_dilation_sizes']
        self.segment_size = config['segment_size']
        self.num_mels = config['num_mels']
        self.n_fft = config['n_fft']
        self.hop_size = config['hop_size']
        self.win_size = config['win_size']
        self.sampling_rate = config['sampling_rate']
        self.fmin = config['fmin']
        self.fmax = config['fmax']
        self.batch_size = config['batch_size']
 
def load_config(path):
    with open(path, 'r') as file:
        config = json.load(file)
    return HyperParams(config)

h = load_config('config_v1.json')
epochs = 50
learning_rate = 0.0001
device ='cpu'

 
data_path = 'C:/aproject/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/wavs'

file_list = [os.path.join(data_path, fname) for fname in os.listdir(data_path)]
random.shuffle(file_list)
file_list = file_list[:10]

# DataLoader 설정
dataset = MelDataset(file_list, h.segment_size, h.n_fft, h.num_mels, h.hop_size,
                     h.win_size, h.sampling_rate, h.fmin, h.fmax, device='cpu')

# DataLoader 설정
data_loader = DataLoader(dataset, batch_size=h.batch_size, shuffle=True, num_workers=0)

generator = Generator(h).to(device)
discriminator = MultiEnvelopeDiscriminator().to(device)

# 옵티마이저
optimizer_g = torch.optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
"""
for epoch in range(epochs):
    for i, data in enumerate(data_loader):
        mel, audio, _, mel_loss = data  # filename과 mel_loss는 사용하지 않음
        mel = mel.to(device)
        audio = audio.to(device)

        # 판별기 업데이트
        optimizer_d.zero_grad()
        real_scores, fake_scores, real_features, fake_features = discriminator(audio, generator(mel))
        d_loss = discriminator_loss(real_scores, fake_scores)
        d_loss.backward()
        optimizer_d.step()

        # 생성기 업데이트
        optimizer_g.zero_grad()
        fake_audio = generator(mel)
        fake_scores, fake_features, _, _ = discriminator(audio, fake_audio)
        g_loss = generator_loss(fake_scores)
        f_loss = feature_loss(real_features, fake_features)
        total_loss = g_loss + f_loss
        total_loss.backward()
        optimizer_g.step()

        # 진행 상황 출력
        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}")
"""
# 학습 루프
for epoch in range(epochs):
    for i, data in enumerate(data_loader):

        mel, audio, _, mel_loss = data
        
        mel = mel.to(device)
        audio = audio.to(device)

        # 생성된 가짜 음성
        fake_audio = generator(mel)

        # 판별기 업데이트
        optimizer_d.zero_grad()
        # 실제 음성과 가짜 음성을 discriminator에 입력
        real_scores, fake_scores, real_features, fake_features = discriminator(audio, fake_audio)
        
        # discriminator 손실 계산
        d_loss, r_losses, g_losses = discriminator_loss(real_scores, fake_scores)
        d_loss.backward()
        optimizer_d.step()

        # 생성기 업데이트
        optimizer_g.zero_grad()
        # 생성된 가짜 음성에 대한 새로운 평가를 받아 다시 판별
        _, fake_scores, _, fake_features = discriminator(audio, fake_audio)
        g_loss, gen_losses = generator_loss(fake_scores)  # 생성기 손실 계산
        f_loss = feature_loss(real_features, fake_features)  # 특징 손실
        total_loss = g_loss + f_loss  # 총 손실
        total_loss.backward()
        optimizer_g.step()

        if i % 10 == 0:
            print(f"Epoch {epoch+1}, Batch {i}, D Loss: {d_loss.item()}, G Loss: {g_loss.item()}, Real Losses: {r_losses}, Generated Losses: {g_losses}, Feature Loss: {f_loss.item()}")

# 모델 저장
torch.save(generator.state_dict(), 'generator.pth')
torch.save(discriminator.state_dict(), 'discriminator.pth')