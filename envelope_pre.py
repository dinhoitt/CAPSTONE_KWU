import librosa
import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt


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
        from scipy.signal import filtfilt
        return filtfilt(b, a, data)

    def filter_data(self, data):
        from scipy.signal import butter, cheby1, cheby2, ellip, bessel
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
    def __init__(self, signal, sr, f_times, filter='butter', order=5, rp=0.5, rs=40):
        self.signal = signal
        self.sr = sr
        self.f_times = f_times
        self.filter = filter
        self.order = order
        self.rp = rp
        self.rs = rs
        self.envelope = self.apply_lowpass_and_compute_envelope()

    def find_max_freq(self):
        fft_result = np.fft.fft(self.signal)
        power_spectrum = np.abs(fft_result)
        freqs = np.fft.fftfreq(len(power_spectrum), 1 / self.sr)
        max_freq_index = np.argmax(power_spectrum[:len(power_spectrum)//2])
        max_freq = abs(freqs[max_freq_index])
        return max_freq

    def compute_envelope(self):
        return np.abs(hilbert(self.signal))

    def apply_lowpass_and_compute_envelope(self):
        max_freq = self.find_max_freq()
        if self.f_times == 0:
            cutoff = max_freq
        else:
            cutoff = max_freq / self.f_times
        lpf = LowPassFilter(self.filter, cutoff, self.sr, self.order, self.rp, self.rs)
        filtered_signal = lpf.filter_data(self.signal)
        return np.abs(hilbert(filtered_signal))



'''
# 사용 예
# /LJ001-0001.wav
audio_path = 'C:/LDMsample/MED/LJ001-0002.wav'
signal, sr = librosa.load(audio_path, sr=None)
envelope_2 = Envelope(audio_path, f_times=0)
envelope_3 = Envelope(audio_path, f_times=1)
envelope_4 = Envelope(audio_path, f_times=1, filter='cheby1')
envelope_5 = Envelope(audio_path, f_times=1, filter='ellip')
envelope_6 = Envelope(audio_path, f_times=1, filter='bessel')


fig, axs = plt.subplots(5, 1, figsize=(14, 10), sharex=True)

# 각 Subplot에 대한 데이터와 레이블 설정
envelopes = [envelope_2.envelope, envelope_3.envelope, envelope_4.envelope, envelope_5.envelope, envelope_6.envelope]
titles = ['Envelope 2', 'Envelope 3', 'Envelope 4', 'Envelope 5', 'Envelope 6']

# 각 Envelope를 서브플롯에 그리기
for ax, envelope, title in zip(axs, envelopes, titles):
    ax.plot(envelope)
    ax.set_title(title)
    ax.set_ylabel('Amplitude')

# X 축 레이블 설정
axs[-1].set_xlabel('Sample Index')

# Plot 표시
plt.tight_layout()
plt.show()
'''