# Ke Chen
# knutchen@ucsd.edu
# HTS-AT: A HIERARCHICAL TOKEN-SEMANTIC AUDIO TRANSFORMER FOR SOUND CLASSIFICATION AND DETECTION
# Some layers designed on the model
# below codes are based and referred from https://github.com/microsoft/Swin-Transformer
# Swin Transformer for Computer Vision: https://arxiv.org/pdf/2103.14030.pdf

import torch
import torch.nn as nn
from itertools import repeat
import collections.abc
import math
import warnings

from torch.nn.init import _calculate_fan_in_and_fan_out
import torch.utils.checkpoint as checkpoint

import random

from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from torchlibrosa.augmentation import SpecAugmentation

from itertools import repeat
from .utils import do_mixup, interpolate

from .feature_fusion import iAFF, AFF, DAF


# from PyTorch internals
"""
이 코드는 다양한 형태의 입력을 받아 특정 길이의 튜플로 변환하는 유틸리티 함수들을 정의하는 부분. 
PyTorch 내부에서 자주 사용되는 패턴으로, 모듈의 구성 요소에 대한 인자가 단일 값으로 주어지거나 여러 값을 포함하는 시퀀스(예: 리스트나 튜플)로 주어질 때, 이를 일관되게 처리하기 위해 사용됨.
"""
"""
_ntuple(n)
n: 반환되어야 하는 튜플의 길이.
parse(x): 함수 내부에 정의된 내부 함수로, 입력 x를 처리하여 길이 n의 튜플로 변환.
입력 x가 이미 반복 가능한 객체(collections.abc.Iterable의 인스턴스, 예를 들어 리스트나 튜플)인 경우, 그대로 반환.
그렇지 않은 경우(예: 단일 숫자나 문자열 등), x를 n번 반복하여 길이 n의 튜플을 생성.
"""
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse

"""
이들은 _ntuple 함수를 사용하여 미리 정의된 길이의 튜플 생성 함수.
예를 들어, to_2tuple 함수는 주어진 입력을 길이가 2인 튜플로 변환. 
이는 모델 구성 요소에 대한 파라미터가 2차원 형태로 필요할 때 유용하게 사용됨(예: 이미지의 너비와 높이)
"""
"""
이러한 함수들은 모델의 파라미터를 더 유연하게 처리할 수 있게 해줌. 
예를 들어, 컨볼루션 레이어에서 커널 크기나 스트라이드 값을 설정할 때, 
모든 차원에 동일한 값을 적용하고자 할 때 단일 숫자를 제공할 수 있고, 각 차원에 다른 값을 적용하고자 할 때는 튜플을 제공할 수 있음. 
이를 통해 코드의 가독성과 유연성이 향상됨.
"""

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

"""
drop_path 함수
x: 입력 텐서.
drop_prob: 드롭될 확률입니다. 예를 들어 0.1이면 텐서의 각 샘플에 대해 10% 확률로 경로가 드롭됨.
training: 모델이 훈련 모드에 있는지 여부를 나타내는 불리언 값. 테스트 시에는 드롭 패스를 적용하지 않기 위해 사용됨.
이 함수는 주어진 확률에 따라 입력 텐서의 일부 경로를 무작위로 드롭함. 
실제로는 입력 텐서의 각 샘플에 대해 독립적으로 드롭 패스 마스크를 생성하고, 
이를 통해 일부 출력을 0으로 만들어 버리는 대신, 드롭되지 않은 출력을 보정하기 위해 남은 출력을 keep_prob로 나누어 크기를 조정.
"""

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

"""
DropPath 클래스
drop_prob: 드롭될 확률. __init__ 메소드에서 초기화.
DropPath 클래스는 nn.Module을 상속받아 PyTorch 모듈로 구현됨. 
이 클래스의 forward 메소드에서는 drop_path 함수를 호출하여 입력 텐서에 대해 드롭 패스를 적용. 
이 클래스는 모델 정의 시에 드롭 패스를 쉽게 적용할 수 있도록 도와줌.

드롭 패스 기법은 특히 레지듀얼 블록(Residual Blocks)이나 트랜스포머(Transformer) 모델 같이 깊은 네트워크 구조에서 유용하게 사용됨. 
이를 통해 모델의 일부 경로를 무작위로 생략함으로써 과적합을 방지하고, 모델의 일반화 성능을 향상시킬 수 있음.
"""

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

"""
*PatchEmbed
2D 이미지(또는 오디오 처리 컨텍스트에서의 스펙트로그램)를 일련의 평평한 패치로 변환하고, 각 패치를 더 높은 차원의 공간으로 임베딩하는 모듈. 
이는 오디오 데이터를 트랜스포머 계층에서 처리할 수 있는 형태로 변환하는 첫 단계.
"""

class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    """
    이 클래스는 변환된 패치 시퀀스를 다음 단계의 모델(예: Transformer)에 입력으로 제공하는 데 사용. 
    패치 기반의 접근 방식은 이미지 또는 다른 2차원 데이터를 처리할 때 공간적 구조를 유지하면서도 효율적인 처리를 가능하게 함.
    """

    """
    파라미터:
    img_size: 입력 이미지의 크기. 튜플 형태로 (높이, 너비)를 지정할 수 있음.
    patch_size: 각 패치의 크기. 튜플 형태로 (높이, 너비)를 지정할 수 있음.
    in_chans: 입력 이미지의 채널 수.
    embed_dim: 임베딩 차원의 크기. 각 패치가 변환될 벡터의 차원.
    norm_layer: 임베딩에 적용할 정규화 계층. 기본값은 None으로, 정규화를 적용하지 않음.
    flatten: 임베딩된 패치들을 일렬로 펼칠지 여부.
    patch_stride: 패치 추출 시 스트라이드. 튜플 형태로 지정하며, 패치 간의 간격을 결정.
    enable_fusion, fusion_type: 특정 융합 기법을 사용할지 여부와 그 유형을 지정.
    """
    
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
        patch_stride=16,
        enable_fusion=False,
        fusion_type="None",
    ):
        """
        임베딩 과정:

        입력 이미지로부터 정의된 patch_size와 patch_stride에 따라 패치들을 추출.
        추출된 각 패치를 embed_dim 차원의 벡터로 변환하기 위해 nn.Conv2d 컨볼루션 연산을 사용. 이 과정에서 패치의 위치 정보는 잃어버리게 됨.
        선택적으로, norm_layer가 지정되어 있다면, 임베딩된 패치들에 정규화를 적용.
        enable_fusion과 fusion_type에 따라 추가적인 처리 가능. 예를 들어, fusion_type이 "channel_map"이라면, 입력 채널 수에 따라 다른 처리를 적용할 수 있음.       
        """
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patch_stride = to_2tuple(patch_stride)
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.grid_size = (
            img_size[0] // patch_stride[0],
            img_size[1] // patch_stride[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        padding = (
            (patch_size[0] - patch_stride[0]) // 2,
            (patch_size[1] - patch_stride[1]) // 2,
        )

        if (self.enable_fusion) and (self.fusion_type == "channel_map"):
            self.proj = nn.Conv2d(
                in_chans * 4,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_stride,
                padding=padding,
            )
        else:
            self.proj = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=patch_size,
                stride=patch_stride,
                padding=padding,
            )
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            self.mel_conv2d = nn.Conv2d(
                in_chans,
                embed_dim,
                kernel_size=(patch_size[0], patch_size[1] * 3),
                stride=(patch_stride[0], patch_stride[1] * 3),
                padding=padding,
            )
            if self.fusion_type == "daf_2d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_2d":
                self.fusion_model = AFF(channels=embed_dim, type="2D")
            elif self.fusion_type == "iaff_2d":
                self.fusion_model = iAFF(channels=embed_dim, type="2D")

    """
    forward 메서드:

    입력 이미지 x를 받아, 위에서 설명한 패치 임베딩 과정을 수행.
    enable_fusion이 활성화되어 있고, 지정된 fusion_type에 따라 추가적인 융합 처리를 할 수 있음. 
    예를 들어, "daf_2d", "aff_2d", "iaff_2d"와 같은 융합 기법을 적용할 수 있음.
    최종적으로, 변환된 패치 시퀀스를 반환.
    """

    def forward(self, x, longer_idx=None):
        if (self.enable_fusion) and (
            self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d"]
        ):
            global_x = x[:, 0:1, :, :]

            # global processing
            B, C, H, W = global_x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            global_x = self.proj(global_x)
            TW = global_x.size(-1)
            if len(longer_idx) > 0:
                # local processing
                local_x = x[longer_idx, 1:, :, :].contiguous()
                B, C, H, W = local_x.shape
                local_x = local_x.view(B * C, 1, H, W)
                local_x = self.mel_conv2d(local_x)
                local_x = local_x.view(
                    B, C, local_x.size(1), local_x.size(2), local_x.size(3)
                )
                local_x = local_x.permute((0, 2, 3, 1, 4)).contiguous().flatten(3)
                TB, TC, TH, _ = local_x.size()
                if local_x.size(-1) < TW:
                    local_x = torch.cat(
                        [
                            local_x,
                            torch.zeros(
                                (TB, TC, TH, TW - local_x.size(-1)),
                                device=global_x.device,
                            ),
                        ],
                        dim=-1,
                    )
                else:
                    local_x = local_x[:, :, :, :TW]

                global_x[longer_idx] = self.fusion_model(global_x[longer_idx], local_x)
            x = global_x
        else:
            B, C, H, W = x.shape
            assert (
                H == self.img_size[0] and W == self.img_size[1]
            ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
            x = self.proj(x)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x



class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    """
    __init__ 메서드는 MLP 모듈을 초기화. 
    
    in_features: 입력층의 노드(뉴런) 수.
    hidden_features: 은닉층의 노드 수. 기본값은 in_features와 동일하게 설정.
    out_features: 출력층의 노드 수. 기본값은 in_features와 동일하게 설정. 이는 기본적으로 입력 크기와 출력 크기를 같게 설정하지만, 필요에 따라 다르게 설정할 수 있음.
    act_layer: 활성화 함수. 기본값은 GELU(Gaussian Error Linear Unit). GELU는 ReLU(Rectified Linear Unit)의 일반화된 형태로, 신경망에서 널리 사용되는 활성화 함수 중 하나.
    drop: 드롭아웃 비율로, 과적합을 방지하기 위해 사용.
    """

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    """
    forward 메서드는 네트워크를 통해 입력 데이터 x가 순전파되는 과정을 정의.

    self.fc1(x): 첫 번째 선형(전결합) 층을 통과.
    self.act(x): 활성화 함수를 적용.
    self.drop(x): 드롭아웃을 적용.
    self.fc2(x): 두 번째 선형(전결합) 층을 통과.
    마지막으로 다시 드롭아웃을 적용한 후 최종 출력을 반환.
    """

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

"""
_no_grad_trunc_normal_ 함수는 주어진 범위 [a, b] 내에서 잘린(truncated) 정규 분포를 따르는 값을 갖는 텐서를 생성. 
이 방식은 특정 범위 내에서만 값을 가질 때 유용하며, 초기 가중치 설정 등에서 사용. 
PyTorch의 공식 구현에서 발췌한 이 함수는 특정 평균(mean)과 표준편차(std)를 가지는 정규 분포에서 값을 샘플링하되, [a, b] 범위 밖의 값은 제외하고 샘플링.
"""

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """
    norm_cdf는 표준 정규 분포의 CDF를 계산하는 내부 함수. 
    여기서 math.erf는 오차 함수(Error Function)로, 가우시안(Gaussian) 적분을 계산.
    """
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    #만약 평균이 주어진 범위 [a, b]로부터 너무 멀리 떨어져 있다면(평균 ± 2표준편차가 [a, b] 범위를 벗어난 경우), 값의 분포가 올바르지 않을 수 있다는 경고를 발생
    if (mean < a - 2 * std) or (mean > b + 2 * std): 
        warnings.warn(
            "mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
            "The distribution of values may be incorrect.",
            stacklevel=2,
        )

    """
    잘린(truncated) 정규 분포 샘플링:

    먼저, 범위 [a, b] 내에서 잘린 정규 분포의 상한(u)과 하한(l)을 CDF를 사용하여 계산.
    텐서를 [l, u] 범위의 균일 분포에서 샘플링하여 채운 후, 이를 [2l-1, 2u-1] 범위로 변환.
    그 다음, 표준 정규 분포의 역 CDF 변환을 사용하여 값들을 잘린 정규 분포로 변환. 이는 tensor.erfinv_()를 통해 수행됨.
    마지막으로, 샘플링된 값들을 지정된 평균과 표준편차로 변환하고, [a, b] 범위로 제한(clamp)하여 최종적으로 조정.
    """

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor

"""
trunc_normal_ 함수는 입력된 텐서를 지정된 평균(mean)과 표준편차(std)를 가진 잘린(truncated) 정규 분포로부터의 값으로 채움. 
이 함수는 값이 [a, b] 범위 밖에 있을 경우 다시 그리기를 반복하여, 모든 값이 이 범위 내에 있도록 함. 
이 함수는 특히 모델 가중치의 초기화에 유용하게 사용됨
"""

def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

"""
variance_scaling_:
주어진 텐서에 분산 스케일링 초기화를 적용하는 함수. 
이 초기화 방식은 신경망의 가중치를 초기화할 때 널리 사용되며, 가중치 값들이 특정 분산을 가지도록 설정함으로써 학습 초기 단계에서 신경망의 안정성을 높이는 데 도움을 줌.

tensor: 초기화를 적용할 텐서.
scale: 분산 스케일링에 사용될 스케일 값. 기본값은 1.0.
mode: 초기화에서 사용할 모드로, 'fan_in', 'fan_out', 또는 'fan_avg'가 될 수 있음.
'fan_in'은 입력 단위의 수를 기준으로 분산을 조정.
'fan_out'은 출력 단위의 수를 기준으로 분산을 조정.
'fan_avg'는 입력과 출력 단위의 평균을 기준으로 분산을 조정.
distribution: 가중치 분포 유형으로, 'truncated_normal', 'normal', 또는 'uniform'이 될 수 있음.
"""

def variance_scaling_(tensor, scale=1.0, mode="fan_in", distribution="normal"):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    #팬 값 계산: _calculate_fan_in_and_fan_out 함수를 사용하여 입력(fan_in)과 출력(fan_out) 단위의 수를 계산
    #분산 계산: 선택된 모드(fan_in, fan_out, fan_avg)에 따라 분모를 결정하고, 이를 사용해 분산을 계산합니다. 분산은 scale / denom으로 정의
    if mode == "fan_in":
        denom = fan_in
    elif mode == "fan_out":
        denom = fan_out
    elif mode == "fan_avg":
        denom = (fan_in + fan_out) / 2

    variance = scale / denom

    """
    분포에 따른 초기화:
    'truncated_normal': 잘린 정규 분포를 사용하여 텐서를 초기화. 이 때, 표준편차는 계산된 분산의 제곱근으로 조정되며, -2와 2 사이의 값으로 제한.
    'normal': 일반 정규 분포를 사용하여 텐서를 초기화. 표준편차는 계산된 분산의 제곱근으로 설정.
    'uniform': 균등 분포를 사용하여 텐서를 초기화. 경계는 계산된 분산의 제곱근에 3을 곱한 값으로 설정.
    """

    if distribution == "truncated_normal":
        # constant is stddev of standard normal truncated to (-2, 2)
        trunc_normal_(tensor, std=math.sqrt(variance) / 0.87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")

"""
lecun_normal_:
이 함수는 텐서(tensor)에 LeCun 정규 분포 초기화를 적용하는 함수. 이 초기화 방법은 Yann LeCun이 제안한 것으로, 신경망의 가중치를 초기화할 때 사용. 
특히, 이 함수는 신경망이 학습을 시작할 때 가중치를 적절한 범위 내에서 초기화하여 학습 과정의 효율성을 높이는 데 도움을 줌
"""

def lecun_normal_(tensor):
    """
    mode="fan_in": 가중치 텐서의 fan_in 값을 사용. fan_in은 가중치 텐서가 연결된 이전 계층의 뉴런(입력 단위) 수를 의미. 
    LeCun 초기화에서는 이전 계층의 뉴런 수에 기반하여 가중치의 분산을 조정하므로, fan_in 모드가 사용됨.

    distribution="truncated_normal": 초기화에 사용되는 값들이 잘린(truncated) 정규 분포에서 추출. 
    이는 가중치 값이 너무 크거나 작은 극단적인 값들을 제외하고, 일정 범위 내의 값들만을 사용하여 초기화한다는 의미. 
    이렇게 함으로써 초기 가중치 값들이 너무 극단적으로 치우치는 것을 방지하고, 신경망의 학습 안정성과 수렴 속도를 향상시킬 수 있음.
    """
    variance_scaling_(tensor, mode="fan_in", distribution="truncated_normal")

"""
window_partition 함수와 window_reverse 함수는 이미지나 텐서를 작은 윈도우(창)로 분할하고, 이러한 분할된 윈도우들을 다시 원래의 형태로 복구하는 과정을 담당. 
이러한 기능은 특히 비전 변환기(Vision Transformers)나 스윈 변환기(Swin Transformers)와 같은 모델에서 지역적 특성을 처리할 때 사용됨.
"""

def window_partition(x, window_size):
    """
    window_partition 함수
    목적: 입력된 텐서 x를 window_size에 따라 작은 윈도우로 분할.
    입력:
    x: (B, H, W, C) 형태의 텐서, 여기서 B는 배치 크기, H와 W는 각각 높이와 너비, C는 채널 수를 의미.
    window_size: 윈도우의 크기(높이와 너비가 같다고 가정).
    출력: 분할된 윈도우들이 담긴 텐서 (num_windows*B, window_size, window_size, C).
    """

    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    #입력 텐서를 view를 사용해 (B, H // window_size, window_size, W // window_size, window_size, C)로 재구성. 이는 각 차원을 윈도우 크기에 맞게 분할
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = (
        #permute를 사용해 텐서의 차원을 재배치하여 윈도우를 연속적으로 배열
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    window_reverse 함수
    목적: window_partition에 의해 분할된 윈도우들을 다시 원래의 크기 (B, H, W, C)로 복구.
    입력:
    windows: 분할된 윈도우들이 담긴 텐서.
    window_size: 원래 윈도우의 크기.
    H, W: 복구할 이미지의 높이와 너비.
    출력: 복구된 텐서 (B, H, W, C).
    """

    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    # 윈도우들이 포함된 텐서 windows의 크기를 계산하여 배치 크기 B를 결정
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    # view를 사용해 텐서를 원래 이미지의 구조에 맞게 재구성. 이 과정에서 윈도우들은 원래의 위치로 배치
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    # permute와 view를 사용해 최종적으로 (B, H, W, C) 형태의 텐서로 복구
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


"""
*WindowAttention: 
로컬 윈도우 내에서 자기 주의를 계산하는 모듈로, 모델이 입력의 다른 부분에 선택적으로 집중할 수 있게 함. 
상대 위치 편향의 사용은 모델이 각 윈도우 내의 패치 배열을 이해하는 데 도움을 줌.
"""

class WindowAttention(nn.Module):
    r"""Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    """
    초기화 매개변수:
    dim: 입력 채널의 수.
    window_size: 윈도우의 높이와 너비를 나타내는 튜플.
    num_heads: 어텐션 헤드의 수.
    qkv_bias: 쿼리(Query), 키(Key), 값(Value) 계산에 학습 가능한 바이어스를 추가할지 여부를 결정. 기본값은 True.
    qk_scale: 헤드 차원의 역수 제곱근을 기본값으로 사용하는 쿼리와 키의 스케일 팩터. 명시적으로 설정할 수도 있음.
    attn_drop: 어텐션 가중치에 적용되는 드롭아웃 비율.
    proj_drop: 출력에 적용되는 드롭아웃 비율.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # define a parameter table of relative position bias
        # relative_position_bias_table: 상대 위치에 대한 바이어스를 저장하는 테이블. 이 바이어스는 윈도우 내의 각 토큰 쌍 사이의 상대적 위치에 따라 다름.
        self.relative_position_bias_table = nn.Parameter( 
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)# softmax: 어텐션 가중치를 계산하기 위한 소프트맥스 함수

    """
    forward 메소드:
    입력 특성 x와 선택적 마스크 mask를 받아들임.
    x를 사용하여 쿼리, 키, 값 벡터를 생성하고, 스케일링된 닷-프로덕트 어텐션을 계산.
    상대 위치 바이어스를 어텐션 스코어에 추가하고, 선택적으로 마스크를 적용.
    어텐션 가중치를 적용한 후, 결과를 선형 변환하고 최종 출력을 반환.
    이 모듈은 특히 이미지 또는 다른 2차원 데이터를 처리할 때 유용하며, 윈도우 내의 요소들 사이의 상호 작용을 모델링하여 성능을 향상시키는 데 도움을 줌.
    """

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = (
            self.qkv(x) # qkv: 입력 특성으로부터 쿼리, 키, 값 벡터를 생성하는 선형 변환.
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1
        ).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn) 
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"


# We use the model based on Swintransformer Block, therefore we can use the swin-transformer pretrained model
"""
*SwinTransformerBlock
Swin Transformer의 기본 빌딩 블록으로, 상대 위치 편향이 포함된 자기 주의 메커니즘과 MLP(다층 퍼셉트론)를 포함. 
이동된 윈도우를 지원하여 모델이 전역 컨텍스트를 더 잘 포착할 수 있음.
"""
class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_before_mlp="ln",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.norm_before_mlp = norm_before_mlp
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert (
            0 <= self.shift_size < self.window_size
        ), "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.norm_before_mlp == "ln":
            self.norm2 = nn.LayerNorm(dim)
        elif self.norm_before_mlp == "bn":
            self.norm2 = lambda x: nn.BatchNorm1d(dim)(x.transpose(1, 2)).transpose(
                1, 2
            )
        else:
            raise NotImplementedError
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(
                img_mask, self.window_size
            )  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        # pdb.set_trace()
        H, W = self.input_resolution
        # print("H: ", H)
        # print("W: ", W)
        # pdb.set_trace()
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn = self.attn(
            x_windows, mask=self.attn_mask
        )  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn

    def extra_repr(self):
        return (
            f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, "
            f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"
        )

"""
*PatchMerging: 
인접한 패치를 병합하여 특징 맵의 공간 해상도를 줄이는 계층으로, 채널 차원을 실질적으로 두 배로 늘림. 
이 연산은 CNN에서의 다운샘플링과 유사하며, 계층적 표현을 구축하는 데 도움을 줌.
"""

class PatchMerging(nn.Module):
    r"""Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self):
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

"""
*BasicLayer: 
모델의 계층적 구조에서 특정 해상도에서 입력을 처리하는 한 단계를 나타냄. 
이는 여러 Swin Transformer 블록을 포함하며, 마지막에는 다운샘플링 계층이 있을 수 있음.
"""

class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        norm_before_mlp="ln",
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                    norm_before_mlp=norm_before_mlp,
                )
                for i in range(depth)
            ]
        )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x):
        attns = []
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x, attn = blk(x)
                if not self.training:
                    attns.append(attn.unsqueeze(0))
        if self.downsample is not None:
            x = self.downsample(x)
        if not self.training:
            attn = torch.cat(attns, dim=0)
            attn = torch.mean(attn, dim=0)
        return x, attn

    def extra_repr(self):
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


# The Core of HTSAT
class HTSAT_Swin_Transformer(nn.Module):
    r"""HTSAT based on the Swin Transformer
    Args:
        spec_size (int | tuple(int)): Input Spectrogram size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        path_stride (iot | tuple(int)): Patch Stride for Frequency and Time Axis. Default: 4
        in_chans (int): Number of input image channels. Default: 1 (mono)
        num_classes (int): Number of classes for classification head. Default: 527
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each HTSAT-Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 8
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        config (module): The configuration Module from config.py
    """

    def __init__(
        self,
        spec_size=256,
        patch_size=4,
        patch_stride=(4, 4),
        in_chans=1,
        num_classes=527,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[4, 8, 16, 32],
        window_size=8,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        norm_before_mlp="ln",
        config=None,
        enable_fusion=False,
        fusion_type="None",
        **kwargs,
    ):
        super(HTSAT_Swin_Transformer, self).__init__()

        self.config = config
        self.spec_size = spec_size
        self.patch_stride = patch_stride
        self.patch_size = patch_size
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.depths = depths
        self.ape = ape
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.num_layers = len(self.depths)
        self.num_features = int(self.embed_dim * 2 ** (self.num_layers - 1))

        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate

        self.qkv_bias = qkv_bias
        self.qk_scale = None

        self.patch_norm = patch_norm
        self.norm_layer = norm_layer if self.patch_norm else None
        self.norm_before_mlp = norm_before_mlp
        self.mlp_ratio = mlp_ratio

        self.use_checkpoint = use_checkpoint

        self.enable_fusion = enable_fusion
        self.fusion_type = fusion_type

        #  process mel-spec ; used only once
        self.freq_ratio = self.spec_size // self.config.mel_bins
        window = "hann"
        center = True
        pad_mode = "reflect"
        ref = 1.0
        amin = 1e-10
        top_db = None
        self.interpolate_ratio = 32  # Downsampled ratio
        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(
            n_fft=config.window_size,
            hop_length=config.hop_size,
            win_length=config.window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )
        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(
            sr=config.sample_rate,
            n_fft=config.window_size,
            n_mels=config.mel_bins,
            fmin=config.fmin,
            fmax=config.fmax,
            ref=ref,
            amin=amin,
            top_db=top_db,
            freeze_parameters=True,
        )
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(
            time_drop_width=64,
            time_stripes_num=2,
            freq_drop_width=8,
            freq_stripes_num=2,
        )  # 2 2
        self.bn0 = nn.BatchNorm2d(self.config.mel_bins)

        # split spctrogram into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=self.spec_size,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer,
            patch_stride=patch_stride,
            enable_fusion=self.enable_fusion,
            fusion_type=self.fusion_type,
        )

        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.grid_size
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, self.embed_dim)
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=self.drop_rate)

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(self.embed_dim * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_scale=self.qk_scale,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[
                    sum(self.depths[:i_layer]) : sum(self.depths[: i_layer + 1])
                ],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint,
                norm_before_mlp=self.norm_before_mlp,
            )
            self.layers.append(layer)

        self.norm = self.norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.maxpool = nn.AdaptiveMaxPool1d(1)

        SF = (
            self.spec_size
            // (2 ** (len(self.depths) - 1))
            // self.patch_stride[0]
            // self.freq_ratio
        )
        self.tscam_conv = nn.Conv2d(
            in_channels=self.num_features,
            out_channels=self.num_classes,
            kernel_size=(SF, 3),
            padding=(0, 1),
        )
        self.head = nn.Linear(num_classes, num_classes)

        if (self.enable_fusion) and (
            self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]
        ):
            self.mel_conv1d = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, stride=3, padding=2),
                nn.BatchNorm1d(64),
            )
            if self.fusion_type == "daf_1d":
                self.fusion_model = DAF()
            elif self.fusion_type == "aff_1d":
                self.fusion_model = AFF(channels=64, type="1D")
            elif self.fusion_type == "iaff_1d":
                self.fusion_model = iAFF(channels=64, type="1D")

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"absolute_pos_embed"}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {"relative_position_bias_table"}

    def forward_features(self, x, longer_idx=None):
        # A deprecated optimization for using a hierarchical output from different blocks

        frames_num = x.shape[2]
        x = self.patch_embed(x, longer_idx=longer_idx)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        for i, layer in enumerate(self.layers):
            x, attn = layer(x)
        # for x
        x = self.norm(x)
        B, N, C = x.shape
        SF = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[0]
        ST = frames_num // (2 ** (len(self.depths) - 1)) // self.patch_stride[1]
        x = x.permute(0, 2, 1).contiguous().reshape(B, C, SF, ST)
        B, C, F, T = x.shape
        # group 2D CNN
        c_freq_bin = F // self.freq_ratio
        x = x.reshape(B, C, F // c_freq_bin, c_freq_bin, T)
        x = x.permute(0, 1, 3, 2, 4).contiguous().reshape(B, C, c_freq_bin, -1)
        # get latent_output
        fine_grained_latent_output = torch.mean(x, dim=2)
        fine_grained_latent_output = interpolate(
            fine_grained_latent_output.permute(0, 2, 1).contiguous(),
            8 * self.patch_stride[1],
        )

        latent_output = self.avgpool(torch.flatten(x, 2))
        latent_output = torch.flatten(latent_output, 1)

        # display the attention map, if needed

        x = self.tscam_conv(x)
        x = torch.flatten(x, 2)  # B, C, T

        fpx = interpolate(
            torch.sigmoid(x).permute(0, 2, 1).contiguous(), 8 * self.patch_stride[1]
        )

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        output_dict = {
            "framewise_output": fpx,  # already sigmoided
            "clipwise_output": torch.sigmoid(x),
            "fine_grained_embedding": fine_grained_latent_output,
            "embedding": latent_output,
        }

        return output_dict

    def crop_wav(self, x, crop_size, spe_pos=None):
        time_steps = x.shape[2]
        tx = torch.zeros(x.shape[0], x.shape[1], crop_size, x.shape[3]).to(x.device)
        for i in range(len(x)):
            if spe_pos is None:
                crop_pos = random.randint(0, time_steps - crop_size - 1)
            else:
                crop_pos = spe_pos
            tx[i][0] = x[i, 0, crop_pos : crop_pos + crop_size, :]
        return tx

    # Reshape the wavform to a img size, if you want to use the pretrained swin transformer model
    def reshape_wav2img(self, x):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert (
            T <= target_T and F <= target_F
        ), "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(
                x, (target_T, x.shape[3]), mode="bicubic", align_corners=True
            )
        if F < target_F:
            x = nn.functional.interpolate(
                x, (x.shape[2], target_F), mode="bicubic", align_corners=True
            )
        x = x.permute(0, 1, 3, 2).contiguous()
        x = x.reshape(
            x.shape[0],
            x.shape[1],
            x.shape[2],
            self.freq_ratio,
            x.shape[3] // self.freq_ratio,
        )
        # print(x.shape)
        x = x.permute(0, 1, 3, 2, 4).contiguous()
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3], x.shape[4])
        return x

    # Repeat the wavform to a img size, if you want to use the pretrained swin transformer model
    def repeat_wat2img(self, x, cur_pos):
        B, C, T, F = x.shape
        target_T = int(self.spec_size * self.freq_ratio)
        target_F = self.spec_size // self.freq_ratio
        assert (
            T <= target_T and F <= target_F
        ), "the wav size should less than or equal to the swin input size"
        # to avoid bicubic zero error
        if T < target_T:
            x = nn.functional.interpolate(
                x, (target_T, x.shape[3]), mode="bicubic", align_corners=True
            )
        if F < target_F:
            x = nn.functional.interpolate(
                x, (x.shape[2], target_F), mode="bicubic", align_corners=True
            )
        x = x.permute(0, 1, 3, 2).contiguous()  # B C F T
        x = x[:, :, :, cur_pos : cur_pos + self.spec_size]
        x = x.repeat(repeats=(1, 1, 4, 1))
        return x

    def forward(
        self, x: torch.Tensor, mixup_lambda=None, infer_mode=False, device=None
    ):  # out_feat_keys: List[str] = None):
        if self.enable_fusion and x["longer"].sum() == 0:
            # if no audio is longer than 10s, then randomly select one audio to be longer
            x["longer"][torch.randint(0, x["longer"].shape[0], (1,))] = True

        if not self.enable_fusion:
            x = x["waveform"].to(device=device, non_blocking=True)
            x = self.spectrogram_extractor(x)  # (batch_size, 1, time_steps, freq_bins)
            x = self.logmel_extractor(x)  # (batch_size, 1, time_steps, mel_bins)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            if self.training:
                x = self.spec_augmenter(x)

            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)

            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x)
        else:
            longer_list = x["longer"].to(device=device, non_blocking=True)
            x = x["mel_fusion"].to(device=device, non_blocking=True)
            x = x.transpose(1, 3)
            x = self.bn0(x)
            x = x.transpose(1, 3)
            longer_list_idx = torch.where(longer_list)[0]
            if self.fusion_type in ["daf_1d", "aff_1d", "iaff_1d"]:
                new_x = x[:, 0:1, :, :].clone().contiguous()
                if len(longer_list_idx) > 0:
                    # local processing
                    fusion_x_local = x[longer_list_idx, 1:, :, :].clone().contiguous()
                    FB, FC, FT, FF = fusion_x_local.size()
                    fusion_x_local = fusion_x_local.view(FB * FC, FT, FF)
                    fusion_x_local = torch.permute(
                        fusion_x_local, (0, 2, 1)
                    ).contiguous()
                    fusion_x_local = self.mel_conv1d(fusion_x_local)
                    fusion_x_local = fusion_x_local.view(
                        FB, FC, FF, fusion_x_local.size(-1)
                    )
                    fusion_x_local = (
                        torch.permute(fusion_x_local, (0, 2, 1, 3))
                        .contiguous()
                        .flatten(2)
                    )
                    if fusion_x_local.size(-1) < FT:
                        fusion_x_local = torch.cat(
                            [
                                fusion_x_local,
                                torch.zeros(
                                    (FB, FF, FT - fusion_x_local.size(-1)),
                                    device=device,
                                ),
                            ],
                            dim=-1,
                        )
                    else:
                        fusion_x_local = fusion_x_local[:, :, :FT]
                    # 1D fusion
                    new_x = new_x.squeeze(1).permute((0, 2, 1)).contiguous()
                    new_x[longer_list_idx] = self.fusion_model(
                        new_x[longer_list_idx], fusion_x_local
                    )
                    x = new_x.permute((0, 2, 1)).contiguous()[:, None, :, :]
                else:
                    x = new_x

            elif self.fusion_type in ["daf_2d", "aff_2d", "iaff_2d", "channel_map"]:
                x = x  # no change

            if self.training:
                x = self.spec_augmenter(x)
            if self.training and mixup_lambda is not None:
                x = do_mixup(x, mixup_lambda)

            x = self.reshape_wav2img(x)
            output_dict = self.forward_features(x, longer_idx=longer_list_idx)

        # if infer_mode:
        #     # in infer mode. we need to handle different length audio input
        #     frame_num = x.shape[2]
        #     target_T = int(self.spec_size * self.freq_ratio)
        #     repeat_ratio = math.floor(target_T / frame_num)
        #     x = x.repeat(repeats=(1,1,repeat_ratio,1))
        #     x = self.reshape_wav2img(x)
        #     output_dict = self.forward_features(x)
        # else:
        #     if x.shape[2] > self.freq_ratio * self.spec_size:
        #         if self.training:
        #             x = self.crop_wav(x, crop_size=self.freq_ratio * self.spec_size)
        #             x = self.reshape_wav2img(x)
        #             output_dict = self.forward_features(x)
        #         else:
        #             # Change: Hard code here
        #             overlap_size = (x.shape[2] - 1) // 4
        #             output_dicts = []
        #             crop_size = (x.shape[2] - 1) // 2
        #             for cur_pos in range(0, x.shape[2] - crop_size - 1, overlap_size):
        #                 tx = self.crop_wav(x, crop_size = crop_size, spe_pos = cur_pos)
        #                 tx = self.reshape_wav2img(tx)
        #                 output_dicts.append(self.forward_features(tx))
        #             clipwise_output = torch.zeros_like(output_dicts[0]["clipwise_output"]).float().to(x.device)
        #             framewise_output = torch.zeros_like(output_dicts[0]["framewise_output"]).float().to(x.device)
        #             for d in output_dicts:
        #                 clipwise_output += d["clipwise_output"]
        #                 framewise_output += d["framewise_output"]
        #             clipwise_output  = clipwise_output / len(output_dicts)
        #             framewise_output = framewise_output / len(output_dicts)
        #             output_dict = {
        #                 'framewise_output': framewise_output,
        #                 'clipwise_output': clipwise_output
        #             }
        #     else: # this part is typically used, and most easy one
        #         x = self.reshape_wav2img(x)
        #         output_dict = self.forward_features(x)
        # x = self.head(x)

        # We process the data in the dataloader part, in that here we only consider the input_T < fixed_T

        return output_dict


def create_htsat_model(audio_cfg, enable_fusion=False, fusion_type="None"):
    try:
        assert audio_cfg.model_name in [
            "tiny",
            "base",
            "large",
        ], "model name for HTS-AT is wrong!"
        if audio_cfg.model_name == "tiny":
            model = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=audio_cfg,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type,
            )
        elif audio_cfg.model_name == "base":
            model = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                embed_dim=128,
                depths=[2, 2, 12, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=audio_cfg,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type,
            )
        elif audio_cfg.model_name == "large":
            model = HTSAT_Swin_Transformer(
                spec_size=256,
                patch_size=4,
                patch_stride=(4, 4),
                num_classes=audio_cfg.class_num,
                embed_dim=256,
                depths=[2, 2, 12, 2],
                num_heads=[4, 8, 16, 32],
                window_size=8,
                config=audio_cfg,
                enable_fusion=enable_fusion,
                fusion_type=fusion_type,
            )

        return model
    except:
        raise RuntimeError(
            f"Import Model for {audio_cfg.model_name} not found, or the audio cfg parameters are not enough."
        )
