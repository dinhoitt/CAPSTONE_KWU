class DiscriminatorE(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorE, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(Conv1d(128, 128, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(128, 256, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        # 최종 특징 맵을 하나의 값으로 요약하는 convolution layer
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1)) 

    def forward(self, x):
        # x는 original signal의 envelope
        fmap = [] # 각 레이어에서 생성된 특징 맵을 저장하는 리스트

        # 이 루프를 통해 컨볼루션 레이어에서 데이터 처리
        for l in self.convs: 
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE) #leaky_relu 활성화 함수를 적용
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)
        """
        최종적으로 torch.flatten을 사용하여
        모든 특징을 하나의 차원으로 평탄화하여 결과를 반환
        이를 통해 최종 디스크리미네이터 점수와 특징 맵을 출력
        """
        return x, fmap

class MultiEnvelopeDiscriminator(torch.nn.Module):
    def __init__(self):
        super(MultiEnvelopeDiscriminator, self).__init__()
        # 여기서 구체적인 구조 및 개발 수정
        self.discriminators = nn.ModuleList([
            DiscriminatorE(use_spectral_norm=True),
            DiscriminatorE(1),
            DiscriminatorE(2),
            DiscriminatorE(1/2),
            DiscriminatorE(1/3),
            DiscriminatorE(1/5),
        ])

    def forward(self, envelope, envelope_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for d in self.discriminators:
            y_d_r, fmap_r = d(envelope)
            y_d_g, fmap_g = d(envelope_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs