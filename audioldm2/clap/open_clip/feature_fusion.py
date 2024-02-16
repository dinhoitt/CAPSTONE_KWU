"""
Feature Fusion for Varible-Length Data Processing
AFF/iAFF is referred and modified from https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py
According to the paper: Yimian Dai et al, Attentional Feature Fusion, IEEE Winter Conference on Applications of Computer Vision, WACV 2021
"""

"""
feature_fusion.py 파일은 다양한 길이의 데이터 처리를 위한 특징 융합 기법, 특히 Attentional Feature Fusion (AFF) 및 그 변형인 iAFF(Improved AFF)를 구현. 
이 기법들은 Yimian Dai et al.에 의해 제안되었으며, 지역적(local) 및 전역적(global) 어텐션 메커니즘을 사용하여 두 특징 맵(feature maps)을 융합. 
이 방법은 다양한 길이의 입력 데이터를 처리할 때 유용하며, 특히 멀티모달 데이터 처리에서 각 모드의 정보를 효과적으로 통합할 수 있음.
"""

import torch
import torch.nn as nn

"""
DAF (Direct Add Fuse) 클래스
DAF 클래스는 간단한 직접 덧셈 융합 방식을 구현. 
입력 x와 residual을 직접 더해 결과를 반환. 
이 방식은 가장 기본적인 특징 융합 방법 중 하나.
"""

class DAF(nn.Module):
    """
    DAF 클래스는 간단한 직접 덧셈 융합 방식을 구현. 입력 x와 잔차 residual을 직접 더해 결과를 반환. 이 방식은 가장 기본적인 특징 융합 방법 중 하나.
    """

    def __init__(self):
        super(DAF, self).__init__()

    def forward(self, x, residual):
        return x + residual


class iAFF(nn.Module):
    """
    iAFF 클래스는 향상된 어텐션 기반 특징 융합 기법을 구현. 여기서는 입력 채널 수(channels), 축소 비율(r), 그리고 융합 유형(type: "1D" 또는 "2D")을 설정할 수 있음.
    지역(local) 및 전역(global) 어텐션을 위한 컨볼루션 레이어와 배치 정규화, ReLU 활성화 함수를 포함하는 시퀀셜 모듈을 정의.
    """

    def __init__(self, channels=64, r=4, type="2D"):
        super(iAFF, self).__init__()
        inter_channels = int(channels // r)

        if type == "1D":
            # 本地注意力
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            # 全局注意力
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )

            # 第二次本地注意力
            self.local_att2 = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            # 第二次全局注意力
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == "2D":
            # 本地注意力
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # 全局注意力
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )

            # 第二次本地注意力
            self.local_att2 = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            # 第二次全局注意力
            self.global_att2 = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        else:
            raise f"the type is not supported"

        self.sigmoid = nn.Sigmoid()

    """
    forward 메소드에서는 입력 x와 잔차 residual에 대해 첫 번째 지역 및 전역 어텐션을 적용한 후, 이를 기반으로 가중치를 계산하여 최종적으로 입력과 잔차를 어텐션 가중치에 따라 융합. 
    두 번째 지역 및 전역 어텐션 과정도 유사하게 수행됩니다
    """

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xi = x * wei + residual * (1 - wei)

        xl2 = self.local_att2(xi)
        xg2 = self.global_att(xi)
        xlg2 = xl2 + xg2
        wei2 = self.sigmoid(xlg2)
        xo = x * wei2 + residual * (1 - wei2)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo


class AFF(nn.Module):
    """
    AFF 클래스는 원래의 어텐션 기반 특징 융합 기법을 구현. 구성 및 작동 방식은 iAFF와 유사하지만, 두 번째 지역 및 전역 어텐션 과정이 없음.
    """

    def __init__(self, channels=64, r=4, type="2D"):
        super(AFF, self).__init__()
        inter_channels = int(channels // r)

        if type == "1D":
            self.local_att = nn.Sequential(
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Conv1d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv1d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm1d(channels),
            )
        elif type == "2D":
            self.local_att = nn.Sequential(
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
            self.global_att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, inter_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(inter_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(inter_channels, channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(channels),
            )
        else:
            raise f"the type is not supported."

        self.sigmoid = nn.Sigmoid()

    """
    forward 메소드에서 입력 x와 잔차 residual에 대해 지역 및 전역 어텐션을 적용하고, 어텐션 가중치를 사용하여 최종 출력을 생성
    """

    def forward(self, x, residual):
        flag = False
        xa = x + residual
        if xa.size(0) == 1:
            xa = torch.cat([xa, xa], dim=0)
            flag = True
        xl = self.local_att(xa)
        xg = self.global_att(xa)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        xo = 2 * x * wei + 2 * residual * (1 - wei)
        if flag:
            xo = xo[0].unsqueeze(0)
        return xo

"""
일반적인 특징
이러한 클래스들은 모두 nn.Module을 상속받아 PyTorch의 모듈 시스템과 호환.
sigmoid 함수를 사용하여 어텐션 가중치를 [0, 1] 범위로 제한. 이를 통해 입력과 잔차 사이의 융합 정도를 조절.
type 매개변수는 "1D" 또는 "2D"를 지원하여, 각각 1차원 또는 2차원 데이터에 대한 융합을 수행할 수 있도록 함.
이 코드는 복잡한 데이터 구조를 다루는 심층 학습 모델, 특히 멀티모달 시나리오에서 입력 특징들 사이의 상호작용을 모델링하기 위한 효과적인 방법을 제공. 
어텐션 메커니즘을 통해 중요한 정보를 강조하고, 여러 입력 소스에서 얻은 정보를 통합하여 성능을 향상시킬 수 있음.
"""