import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm
from utils import init_weights, get_padding
from envelope_final import Envelope

LRELU_SLOPE = 0.1

class DiscriminatorE(nn.Module):
    def __init__(self, signal, sr, f_times, num_layers=5, channels=64, kernel_size=3, use_spectral_norm=False):
        super(DiscriminatorE, self).__init__()
        self.envelope_processor = Envelope(signal, sr, f_times=f_times)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            layer = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size//2)
            if use_spectral_norm:
                layer = weight_norm(layer)
            self.layers.append(layer)

    def forward(self, x):
        fmap = []
        x = self.envelope_processor.envelope

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiEnvelopeDiscriminator(nn.Module):
    def __init__(self, signal, sr):
        super(MultiEnvelopeDiscriminator, self).__init__()
        f_times_values = [-1, 0, 1, 2, 3]
        self.discriminators = nn.ModuleList(
            [DiscriminatorE(signal, sr, f_times=f) for f in f_times_values]
        )

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


"""
솔직히 잘 모르겠음...?
envelope부터 수정을 해야할거 같은데...
"""