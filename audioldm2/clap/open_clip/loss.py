import torch
import torch.distributed.nn
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, accuracy_score

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

"""
gater_features
이 함수는 분산 학습 환경에서 오디오 특징(audio features)과 텍스트 특징(text features)을 수집하고, 필요에 따라 멀티 레이어 퍼셉트론(MLP)을 통과한 특징을 처리.
Horovod와 PyTorch Distributed를 사용하여 다중 GPU 또는 노드에서 학습 데이터를 수집. 이는 모델이 전체 데이터셋에서 학습할 수 있게 하여, 모델의 일반화 성능을 향상시키는 데 도움이 됨.
local_loss 매개변수는 로컬 노드에서만 손실을 계산할지, 전체 데이터셋에 대해 손실을 계산할지를 결정.
gather_with_grad 매개변수는 기울기 계산을 위해 데이터를 수집할지 여부를 결정.
"""

def gather_features(
    audio_features,
    text_features,
    audio_features_mlp=None,
    text_features_mlp=None,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False,
    mlp_loss=False,
):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        if gather_with_grad:
            all_audio_features = hvd.allgather(audio_features)
            all_text_features = hvd.allgather(text_features)
            if mlp_loss:
                all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                all_text_features_mlp = hvd.allgather(text_features_mlp)
        else:
            with torch.no_grad():
                all_audio_features = hvd.allgather(audio_features)
                all_text_features = hvd.allgather(text_features)
                if mlp_loss:
                    all_audio_features_mlp = hvd.allgather(audio_features_mlp)
                    all_text_features_mlp = hvd.allgather(text_features_mlp)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_audio_features = list(
                    all_audio_features.chunk(world_size, dim=0)
                )
                gathered_text_features = list(
                    all_text_features.chunk(world_size, dim=0)
                )
                gathered_audio_features[rank] = audio_features
                gathered_text_features[rank] = text_features
                all_audio_features = torch.cat(gathered_audio_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
                if mlp_loss:
                    gathered_audio_features_mlp = list(
                        all_audio_features_mlp.chunk(world_size, dim=0)
                    )
                    gathered_text_features_mlp = list(
                        all_text_features_mlp.chunk(world_size, dim=0)
                    )
                    gathered_audio_features_mlp[rank] = audio_features_mlp
                    gathered_text_features_mlp[rank] = text_features_mlp
                    all_audio_features_mlp = torch.cat(
                        gathered_audio_features_mlp, dim=0
                    )
                    all_text_features_mlp = torch.cat(gathered_text_features_mlp, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_audio_features = torch.cat(
                torch.distributed.nn.all_gather(audio_features), dim=0
            )
            all_text_features = torch.cat(
                torch.distributed.nn.all_gather(text_features), dim=0
            )
            if mlp_loss:
                all_audio_features_mlp = torch.cat(
                    torch.distributed.nn.all_gather(audio_features_mlp), dim=0
                )
                all_text_features_mlp = torch.cat(
                    torch.distributed.nn.all_gather(text_features_mlp), dim=0
                )
        else:
            gathered_audio_features = [
                torch.zeros_like(audio_features) for _ in range(world_size)
            ]
            gathered_text_features = [
                torch.zeros_like(text_features) for _ in range(world_size)
            ]
            dist.all_gather(gathered_audio_features, audio_features)
            dist.all_gather(gathered_text_features, text_features)
            if mlp_loss:
                gathered_audio_features_mlp = [
                    torch.zeros_like(audio_features_mlp) for _ in range(world_size)
                ]
                gathered_text_features_mlp = [
                    torch.zeros_like(text_features_mlp) for _ in range(world_size)
                ]
                dist.all_gather(gathered_audio_features_mlp, audio_features_mlp)
                dist.all_gather(gathered_text_features_mlp, text_features_mlp)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_audio_features[rank] = audio_features
                gathered_text_features[rank] = text_features
                if mlp_loss:
                    gathered_audio_features_mlp[rank] = audio_features_mlp
                    gathered_text_features_mlp[rank] = text_features_mlp

            all_audio_features = torch.cat(gathered_audio_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)
            if mlp_loss:
                all_audio_features_mlp = torch.cat(gathered_audio_features_mlp, dim=0)
                all_text_features_mlp = torch.cat(gathered_text_features_mlp, dim=0)
    if mlp_loss:
        return (
            all_audio_features,
            all_text_features,
            all_audio_features_mlp,
            all_text_features_mlp,
        )
    else:
        return all_audio_features, all_text_features

"""
ClipLoss
이 클래스는 오디오와 텍스트(또는 다른 모달) 간의 관계를 학습하기 위한 손실 함수를 정의.
클립 손실(Clip Loss)은 오디오 특징과 텍스트 특징 사이의 상호 정보를 최대화하기 위해 설계됨. 이를 통해 모델이 두 모달 사이의 의미적인 매칭을 학습할 수 있음.
MLP 손실 옵션을 통해 추가적인 MLP 네트워크를 통과한 특징을 사용하여 손실을 계산할 수 있습니다. 이는 특징의 표현력을 더욱 향상시킬 수 있음.
손실 계산 시, 로컬 또는 전체 데이터셋에 대한 선택적 계산이 가능.
가중치 손실 옵션을 통해 손실 함수에 가중치를 적용하여 특정 샘플에 더 많은 중요도를 부여할 수 있음.
"""

class ClipLoss(nn.Module):
    def __init__(
        self,
        local_loss=False,
        gather_with_grad=False,
        cache_labels=False,
        rank=0,
        world_size=1,
        use_horovod=False,
        mlp_loss=False,
        weight_loss_kappa=0,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.mlp_loss = mlp_loss
        self.weighted_loss = bool(weight_loss_kappa != 0)
        self.weight_loss_kappa = weight_loss_kappa
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(
        self,
        audio_features,
        text_features,
        logit_scale_a,
        logit_scale_t=None,
        audio_features_mlp=None,
        text_features_mlp=None,
    ):
        device = audio_features.device
        if self.mlp_loss:
            if self.world_size > 1:
                (
                    all_audio_features,
                    all_text_features,
                    all_audio_features_mlp,
                    all_text_features_mlp,
                ) = gather_features(
                    audio_features=audio_features,
                    text_features=text_features,
                    audio_features_mlp=audio_features_mlp,
                    text_features_mlp=text_features_mlp,
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss,
                )
                if self.local_loss:
                    a_logits_per_audio = (
                        logit_scale_a * audio_features @ all_text_features_mlp.T
                    )
                    a_logits_per_text = (
                        logit_scale_a * text_features_mlp @ all_audio_features.T
                    )
                    t_logits_per_audio = (
                        logit_scale_t * audio_features_mlp @ all_text_features.T
                    )
                    t_logits_per_text = (
                        logit_scale_t * text_features @ all_audio_features_mlp.T
                    )
                else:
                    a_logits_per_audio = (
                        logit_scale_a * all_audio_features @ all_text_features_mlp.T
                    )
                    a_logits_per_text = a_logits_per_audio.T
                    t_logits_per_audio = (
                        logit_scale_t * all_audio_features_mlp @ all_text_features.T
                    )
                    t_logits_per_text = t_logits_per_audio.T
            else:
                a_logits_per_audio = (
                    logit_scale_a * audio_features @ text_features_mlp.T
                )
                a_logits_per_text = logit_scale_a * text_features_mlp @ audio_features.T
                t_logits_per_audio = (
                    logit_scale_t * audio_features_mlp @ text_features.T
                )
                t_logits_per_text = logit_scale_t * text_features @ audio_features_mlp.T

            # calculated ground-truth and cache if enabled
            num_logits = a_logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]

            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(a_logits_per_audio, labels)
                    + F.cross_entropy(a_logits_per_text, labels)
                    + F.cross_entropy(t_logits_per_audio, labels)
                    + F.cross_entropy(t_logits_per_text, labels)
                ) / 4
            else:
                audio_weight = (audio_features @ audio_features.T).detach()
                audio_weight = (
                    torch.exp(
                        torch.sum(audio_weight, axis=1)
                        / (self.weight_loss_kappa * len(audio_weight))
                    )
                ).detach()
                text_weight = (text_features @ text_features.T).detach()
                text_weight = (
                    torch.exp(
                        torch.sum(text_weight, axis=1)
                        / (self.weight_loss_kappa * len(text_features))
                    )
                ).detach()
                total_loss = (
                    F.cross_entropy(a_logits_per_audio, labels, weight=audio_weight)
                    + F.cross_entropy(a_logits_per_text, labels, weight=audio_weight)
                    + F.cross_entropy(t_logits_per_audio, labels, weight=text_weight)
                    + F.cross_entropy(t_logits_per_text, labels, weight=text_weight)
                ) / 4
        else:
            if self.world_size > 1:
                all_audio_features, all_text_features = gather_features(
                    audio_features=audio_features,
                    text_features=text_features,
                    local_loss=self.local_loss,
                    gather_with_grad=self.gather_with_grad,
                    rank=self.rank,
                    world_size=self.world_size,
                    use_horovod=self.use_horovod,
                    mlp_loss=self.mlp_loss,
                )

                if self.local_loss:
                    logits_per_audio = (
                        logit_scale_a * audio_features @ all_text_features.T
                    )
                    logits_per_text = (
                        logit_scale_a * text_features @ all_audio_features.T
                    )
                else:
                    logits_per_audio = (
                        logit_scale_a * all_audio_features @ all_text_features.T
                    )
                    logits_per_text = logits_per_audio.T
            else:
                logits_per_audio = logit_scale_a * audio_features @ text_features.T
                logits_per_text = logit_scale_a * text_features @ audio_features.T

            # calculated ground-truth and cache if enabled
            num_logits = logits_per_audio.shape[0]
            if self.prev_num_logits != num_logits or device not in self.labels:
                labels = torch.arange(num_logits, device=device, dtype=torch.long)
                if self.world_size > 1 and self.local_loss:
                    labels = labels + num_logits * self.rank
                if self.cache_labels:
                    self.labels[device] = labels
                    self.prev_num_logits = num_logits
            else:
                labels = self.labels[device]
            if not self.weighted_loss:
                total_loss = (
                    F.cross_entropy(logits_per_audio, labels)
                    + F.cross_entropy(logits_per_text, labels)
                ) / 2
            else:
                audio_weight = (all_audio_features @ all_audio_features.T).detach()
                audio_weight = (
                    torch.exp(
                        torch.sum(audio_weight, axis=1)
                        / (self.weight_loss_kappa * len(all_audio_features))
                    )
                ).detach()
                text_weight = (all_text_features @ all_text_features.T).detach()
                text_weight = (
                    torch.exp(
                        torch.sum(text_weight, axis=1)
                        / (self.weight_loss_kappa * len(all_text_features))
                    )
                ).detach()
                total_loss = (
                    F.cross_entropy(logits_per_audio, labels, weight=text_weight)
                    + F.cross_entropy(logits_per_text, labels, weight=audio_weight)
                ) / 2
        return total_loss

"""
lp_gather_features
라벨 분포 학습(Label Distribution Learning, LDL)에 사용되는 특성을 수집하는 함수
"""

def lp_gather_features(pred, target, world_size=1, use_horovod=False):
    if use_horovod:
        assert hvd is not None, "Please install horovod"
        with torch.no_grad():
            all_preds = hvd.allgather(pred)
            all_targets = hvd.allgath(target)
    else:
        gathered_preds = [torch.zeros_like(pred) for _ in range(world_size)]
        gathered_targets = [torch.zeros_like(target) for _ in range(world_size)]

        dist.all_gather(gathered_preds, pred)
        dist.all_gather(gathered_targets, target)
        all_preds = torch.cat(gathered_preds, dim=0)
        all_targets = torch.cat(gathered_targets, dim=0)

    return all_preds, all_targets

"""
get_map, get_acc, get_mauc
모델 예측에 대한 평균 정밀도(Mean Average Precision, MAP), 정확도(Accuracy, ACC), 평균 AUC(Mean AUC, MAUC)를 계산.
이 메트릭들은 모델의 성능을 다각도로 평가하기 위해 사용됨
"""

def get_map(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(average_precision_score(target, pred, average=None))


def get_acc(pred, target):
    pred = torch.argmax(pred, 1).numpy()
    target = torch.argmax(target, 1).numpy()
    return accuracy_score(target, pred)


def get_mauc(pred, target):
    pred = torch.sigmoid(pred).numpy()
    target = target.numpy()
    return np.mean(roc_auc_score(target, pred, average=None))

"""
LPMetrics
여러 성능 평가 메트릭을 관리하고 계산하는 데 사용됨.
사용자가 지정한 메트릭 목록에 따라 예측과 타겟 간의 성능을 평가하고 결과를 사전 형태로 반환.
"""

class LPMetrics(object):
    def __init__(self, metric_names=["map", "acc", "mauc"]):
        self.metrics = []
        for name in metric_names:
            self.metrics.append(self.get_metric(name))
        self.metric_names = metric_names

    def get_metric(self, name):
        if name == "map":
            return get_map
        elif name == "acc":
            return get_acc
        elif name == "mauc":
            return get_mauc
        else:
            raise ValueError(f"the metric should be at least one of [map, acc, mauc]")

    def evaluate_mertics(self, pred, target):
        metric_dict = {}
        for i in range(len(self.metric_names)):
            metric_dict[self.metric_names[i]] = self.metrics[i](pred, target)
        return metric_dict


def calc_celoss(pred, target):
    target = torch.argmax(target, 1).long()
    return nn.CrossEntropyLoss()(pred, target)

"""
LPLoss
학습 시 사용할 손실 함수를 정의.
이진 크로스 엔트로피(BCE), 크로스 엔트로피(CE), 평균 제곱 오차(MSE) 등 다양한 손실 함수를 지원.
"""

class LPLoss(nn.Module):
    def __init__(self, loss_name):
        super().__init__()
        if loss_name == "bce":
            self.loss_func = nn.BCEWithLogitsLoss()
        elif loss_name == "ce":
            self.loss_func = calc_celoss
        elif loss_name == "mse":
            self.loss_func = nn.MSELoss()
        else:
            raise ValueError(f"the loss func should be at least one of [bce, ce, mse]")

    def forward(self, pred, target):
        loss = self.loss_func(pred, target)
        return loss
