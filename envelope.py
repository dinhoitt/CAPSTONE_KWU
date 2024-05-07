import librosa
import numpy as np
from scipy.signal import hilbert
import torch
from scipy.signal import butter, cheby1, cheby2, ellip, bessel
from scipy.signal import filtfilt
import matplotlib.pyplot as plt

"""
5월 7일 완성!!!!!!! 
"""

# audio_path = 'C:/aproject/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/wavs/LJ001-0025.wav'

class LowPassFilter:
    def __init__(self, filter, cutoff, fs, order=2, rp=None, rs=None):
        self.filter = filter
        self.cutoff = cutoff
        self.fs = fs
        self.order = order
        self.rp = rp
        self.rs = rs
        self.nyq = 0.5 * fs
        self.normal_cutoff = cutoff / self.nyq

    def _apply_filter(self, b, a, data):
        return filtfilt(b, a, data)

    def filter_data(self, data):
        if self.filter == 'butter':
            b, a = butter(self.order, self.normal_cutoff, btype='low', analog=False)
        elif self.filter == 'cheby1':
            b, a = cheby1(self.order, self.rp, self.normal_cutoff, btype='low', analog=False)
        elif self.filter == 'cheby2':
            b, a = cheby2(self.order, self.rs, self.normal_cutoff, btype='low', analog=False)
        elif self.filter == 'ellip':
            b, a = ellip(self.order, self.rp, self.rs, self.normal_cutoff, btype='low', analog=False)
        elif self.filter == 'bessel':
            b, a = bessel(self.order, self.normal_cutoff, btype='low', analog=False, norm='phase')
        else:
            raise ValueError("Unsupported filter type")
        return self._apply_filter(b, a, data)

class Envelope:
    def __init__(self, x, max_freq, filter='butter', order=5, rp=0.5, rs=40):    
        self.signal = x
        self.sr = 22050
        self.max_freq = max_freq
        if self.max_freq == 0:
            self.cutoff = 0  # 또는 적절한 기본값 설정
        else:
            self.cutoff = self.max_freq
        self.filter = filter
        self.order = order
        self.rp = rp
        self.rs = rs
        self.envelope = torch.from_numpy(self.apply_lowpass_and_compute_envelope()).float()
        if torch.cuda.is_available():
            self.envelope = self.envelope.cuda()  # GPU가 사용 가능하면, GPU로 이동

    def apply_lowpass_and_compute_envelope(self):
        self.signal = self.signal.cpu().detach().numpy()
        if self.max_freq == -1:
            envelope_lower = -np.abs(hilbert(-self.signal))
            return envelope_lower
        elif self.max_freq == 0:
            return self.signal
        elif self.max_freq == 1:
            envelope_upper = np.abs(hilbert(self.signal))
            return envelope_upper
        else:
            lpf = LowPassFilter(self.filter, self.cutoff, self.sr, self.order, self.rp, self.rs)
            filtered_signal = lpf.filter_data(self.signal)
            return np.abs(hilbert(filtered_signal))
        
# audio_path = 'C:/aproject/hifi-gan/LJSpeech-1.1/LJSpeech-1.1/wavs/LJ001-0025.wav'

# x, _ = librosa.load(audio_path)
# x = torch.from_numpy(x).float()

# envelope_2 = Envelope(x, max_freq=1000)

# envelope_data = envelope_2.envelope.numpy()

# time = np.linspace(0, len(envelope_data) / 22050, num=len(envelope_data))

# plt.figure(figsize=(10, 4))  # 그래프 크기 설정
# plt.plot(time, envelope_data, label='Envelope')  # 그래프에 데이터 플롯
# plt.title('Signal Envelope')  # 그래프 제목
# plt.xlabel('Time (seconds)')  # x축 라벨
# plt.ylabel('Amplitude')  # y축 라벨
# plt.legend()  # 범례 표시
# plt.show()  # 그래프 표시



