import numpy as np
import torch
from torch import nn as nn
from torchvision.ops.misc import FrozenBatchNorm2d
import logging
from tqdm import tqdm
import random
import json
import os
import pathlib

# TODO: (yusong) this not a good place to store those information and does not scale. Need to be fixed later.
dataset_split = {
    "audiocaps": ["train", "valid", "test"],
    "audioset": ["balanced_train", "unbalanced_train", "eval"],
    "BBCSoundEffects": ["train", "test"],
    "Clotho": ["train", "test", "valid"],
    "free_to_use_sounds": ["train", "test"],
    "paramount_motion": ["train", "test"],
    "sonniss_game_effects": ["train", "test"],
    "wesoundeffects": ["train", "test"],
    "MACS": ["train", "test"],
    "freesound": ["train", "test"],
    "FSD50K": ["train", "test", "valid"],
    "fsd50k_class_label": ["train", "test", "valid"],
    "esc50": ["train", "test"],
    "audiostock": ["train", "test"],
    "freesound_no_overlap_noesc50": ["train", "test"],
    "epidemic_sound_effects": ["train", "test"],
    "VGGSound": ["train", "test"],
    "urbansound8k_class_label": ["train", "test"],
    "audioset_t5": ["balanced_train", "unbalanced_train", "eval"],
    "epidemic_sound_effects_t5": ["train", "test"],
    "WavText5K": ["train", "test"],
    "esc50_no_overlap": ["train", "test"],
    "usd8k_no_overlap": ["train", "test"],
    "fsd50k_200_class_label": ["train", "test", "valid"],
}

"""
freeze_batch_norm_2d() 함수:
모듈 내의 모든 BatchNorm2d 및 SyncBatchNorm 레이어를 FrozenBatchNorm2d로 변환. 
이는 배치 정규화 레이어의 매개변수를 고정하여 모델 학습 중에 업데이트되지 않게 함. 
이는 특정 시나리오에서 모델의 일반화 성능을 향상시키는 데 도움이 될 수 있음.
"""

def freeze_batch_norm_2d(module, module_match={}, name=""):
    """
    Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`. If `module` is
    itself an instance of either `BatchNorm2d` or `SyncBatchNorm`, it is converted into `FrozenBatchNorm2d` and
    returned. Otherwise, the module is walked recursively and submodules are converted in place.

    Args:
        module (torch.nn.Module): Any PyTorch module.
        module_match (dict): Dictionary of full module names to freeze (all if empty)
        name (str): Full module name (prefix)

    Returns:
        torch.nn.Module: Resulting module

    Inspired by https://github.com/pytorch/pytorch/blob/a5895f85be0f10212791145bfedc0261d364f103/torch/nn/modules/batchnorm.py#L762
    """
    res = module
    is_match = True
    if module_match:
        is_match = name in module_match
    if is_match and isinstance(
        module, (nn.modules.batchnorm.BatchNorm2d, nn.modules.batchnorm.SyncBatchNorm)
    ):
        res = FrozenBatchNorm2d(module.num_features)
        res.num_features = module.num_features
        res.affine = module.affine
        if module.affine:
            res.weight.data = module.weight.data.clone().detach()
            res.bias.data = module.bias.data.clone().detach()
        res.running_mean.data = module.running_mean.data
        res.running_var.data = module.running_var.data
        res.eps = module.eps
    else:
        for child_name, child in module.named_children():
            full_child_name = ".".join([name, child_name]) if name else child_name
            new_child = freeze_batch_norm_2d(child, module_match, full_child_name)
            if new_child is not child:
                res.add_module(child_name, new_child)
    return res

"""
exist() 함수:
지정된 데이터셋 이름과 유형이 존재하는지 확인. 이는 데이터 처리 과정에서 유효한 데이터셋과 분할을 확인하는 데 사용.
"""

def exist(dataset_name, dataset_type):
    """
    Check if dataset exists
    """
    if dataset_type in dataset_split[dataset_name]:
        return True
    else:
        return False

"""
get_tar_path_from_dataset_name() 및 get_tar_path_from_txts() 함수:
데이터셋 이름, 유형, 위치 등을 기반으로 데이터 파일의 경로를 생성. 이는 웹 데이터셋이나 로컬 파일 시스템에서 학습 데이터를 효율적으로 로드하는 데 사용
"""

def get_tar_path_from_dataset_name(
    dataset_names, dataset_types, islocal, dataset_path, proportion=1, full_dataset=None
):
    """
    Get tar path from dataset name and type
    """
    output = []
    for n in dataset_names:
        if full_dataset is not None and n in full_dataset:
            current_dataset_types = dataset_split[n]
        else:
            current_dataset_types = dataset_types
        for s in current_dataset_types:
            tmp = []
            if islocal:
                sizefilepath_ = f"{dataset_path}/{n}/{s}/sizes.json"
                if not os.path.exists(sizefilepath_):
                    sizefilepath_ = f"./json_files/{n}/{s}/sizes.json"
            else:
                sizefilepath_ = f"./json_files/{n}/{s}/sizes.json"
            if not os.path.exists(sizefilepath_):
                continue
            sizes = json.load(open(sizefilepath_, "r"))
            for k in sizes.keys():
                if islocal:
                    tmp.append(f"{dataset_path}/{n}/{s}/{k}")
                else:
                    tmp.append(
                        f"pipe:aws s3 --cli-connect-timeout 0 cp s3://s-laion-audio/webdataset_tar/{n}/{s}/{k} -"
                    )
            if proportion != 1:
                tmp = random.sample(tmp, int(proportion * len(tmp)))
            output.append(tmp)
    return sum(output, [])


def get_tar_path_from_txts(txt_path, islocal, proportion=1):
    """
    Get tar path from txt path
    """
    if isinstance(txt_path, (list, tuple)):
        return sum(
            [
                get_tar_path_from_txts(
                    txt_path[i], islocal=islocal, proportion=proportion
                )
                for i in range(len(txt_path))
            ],
            [],
        )
    if isinstance(txt_path, str):
        with open(txt_path) as f:
            lines = f.readlines()
        if islocal:
            lines = [
                lines[i]
                .split("\n")[0]
                .replace("pipe:aws s3 cp s3://s-laion-audio/", "/mnt/audio_clip/")
                for i in range(len(lines))
            ]
        else:
            lines = [
                lines[i].split("\n")[0].replace(".tar", ".tar -")
                for i in range(len(lines))
            ]
        if proportion != 1:
            print("Sampling tars with proportion of {}".format(proportion))
            lines = random.sample(lines, int(proportion * len(lines)))
        return lines

"""
get_mix_lambda() 및 do_mixup() 함수:
믹스업(mixup) 데이터 증강 기법을 구현. 이 기법은 두 샘플을 선형적으로 혼합하여 모델의 일반화 성능을 향상시키는 데 도움이 됨
"""

def get_mix_lambda(mixup_alpha, batch_size):
    mixup_lambdas = [
        np.random.beta(mixup_alpha, mixup_alpha, 1)[0] for _ in range(batch_size)
    ]
    return np.array(mixup_lambdas).astype(np.float32)


def do_mixup(x, mixup_lambda):
    """
    Args:
      x: (batch_size , ...)
      mixup_lambda: (batch_size,)
    Returns:
      out: (batch_size, ...)
    """
    out = (
        x.transpose(0, -1) * mixup_lambda
        + torch.flip(x, dims=[0]).transpose(0, -1) * (1 - mixup_lambda)
    ).transpose(0, -1)
    return out

"""
interpolate() 및 pad_framewise_output() 함수:
시계열 데이터나 프레임 단위 출력을 다루기 위한 함수. 예를 들어, CNN에서 다운샘플링으로 인한 해상도 감소를 보정하거나 출력 길이를 입력 데이터와 일치시키는 데 사용
"""

def interpolate(x, ratio):
    """Interpolate data in time domain. This is used to compensate the
    resolution reduction in downsampling of a CNN.

    Args:
      x: (batch_size, time_steps, classes_num)
      ratio: int, ratio to interpolate
    Returns:
      upsampled: (batch_size, time_steps * ratio, classes_num)
    """
    (batch_size, time_steps, classes_num) = x.shape
    upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
    upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
    return upsampled


def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.
    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad
    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(
        1, frames_num - framewise_output.shape[1], 1
    )
    """tensor for padding"""

    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""


def save_to_dict(s, o_={}):
    sp = s.split(": ")
    o_.update({sp[0]: float(sp[1])})
    return o_


def get_data_from_log(txt_path):
    """
    Output dictionary from out.txt log file
    """
    with open(txt_path) as f:
        lines = f.readlines()
    val_data = {}
    train_data = {}
    train_losses = []
    train_losses_epoch = []
    for i in range(len(lines)):
        if "| INFO |" in lines[i]:
            if "Eval Epoch" in lines[i]:
                if "val_loss" in lines[i]:
                    # float(regex.sub("", lines[310].split("	")[-1]).replace(" ", ""))
                    line = lines[i].split("Eval Epoch: ")[-1]
                    num_epoch = int(line.split("	")[0].split(" ")[0])
                    d = {
                        line.split("	")[0]
                        .split(" ")[1]
                        .replace(":", ""): float(line.split("	")[0].split(" ")[-1])
                    }
                    for i in range(1, len(line.split("	"))):
                        d = save_to_dict(line.split("	")[i], d)
                    val_data[num_epoch] = d
            elif "Train Epoch" in lines[i]:
                num_epoch = int(lines[i].split("Train Epoch: ")[1][0])
                loss = float(lines[i].split("Loss: ")[-1].split(" (")[0])
                train_losses.append(loss)
                train_losses_epoch.append(num_epoch)
    for i in range(len(train_losses)):
        train_data[i] = {
            "num_epoch": train_losses_epoch[i],
            "train_loss": train_losses[i],
        }
    return train_data, val_data

"""
save_p(), load_p(), save_json(), load_json() 함수:
객체, 설정, 결과 등을 저장하고 로드하는 데 사용되는 직렬화 및 역직렬화 함수. 이를 통해 모델의 설정이나 실험 결과를 파일로 쉽게 저장하고 재사용할 수 있음
"""

def save_p(obj, filename):
    import pickle

    try:
        from deepdiff import DeepDiff
    except:
        os.system("pip install deepdiff")
        from deepdiff import DeepDiff
    with open(filename, "wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)  # highest protocol
    with open(filename, "rb") as file:
        z = pickle.load(file)
    assert (
        DeepDiff(obj, z, ignore_string_case=True) == {}
    ), "there is something wrong with the saving process"
    return


def load_p(filename):
    import pickle

    with open(filename, "rb") as file:
        z = pickle.load(file)
    return z


def save_json(data, name="data.json"):
    import json

    with open(name, "w") as fp:
        json.dump(data, fp)
    return


def load_json(name):
    import json

    with open(name, "r") as fp:
        data = json.load(fp)
    return data

"""
load_class_label() 함수:
클래스 레이블 정보를 로드하는 함수. 다양한 파일 형식(pickle, json, numpy, csv 등)을 지원하여 다양한 데이터 소스에서 클래스 레이블을 효율적으로 사용할 수 있게 함
"""

def load_class_label(path):
    # https://stackoverflow.com/questions/48004243/how-to-share-large-read-only-dictionary-list-across-processes-in-multiprocessing
    # https://stackoverflow.com/questions/45693949/storing-strings-in-a-multiprocessing-sharedctypes-array
    out = None
    if path is not None:
        if pathlib.Path(path).suffix in [".pkl", ".pickle"]:
            out = load_p(path)
        elif pathlib.Path(path).suffix in [".json", ".txt"]:
            out = load_json(path)
        elif pathlib.Path(path).suffix in [".npy", ".npz"]:
            out = np.load(path)
        elif pathlib.Path(path).suffix in [".csv"]:
            import pandas as pd

            out = pd.read_csv(path)
    return out
    # if out is None:
    #     return None
    # else:
    #     key = Array(c_wchar, '\n'.join(list(out.keys())), lock=False)
    #     val = Array('i', out.values(), lock=False)
    #     return (key, val)


from torch import optim

"""
get_optimizer() 함수:
모델 학습을 위한 옵티마이저를 생성. AdamW, SGD, Adam 등 다양한 옵티마이저를 지원하여 학습 과정을 사용자 정의
"""

def get_optimizer(params, lr, betas, eps, momentum, optimizer_name):
    if optimizer_name.lower() == "adamw":
        optimizer = optim.AdamW(params, lr=lr, betas=betas, eps=eps)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(params, lr=lr, momentum=momentum)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(params, lr=lr, betas=betas, eps=eps)
    else:
        raise ValueError("optimizer name is not correct")
    return optimizer



# def process_ipc(index_path, classes_num, filename):
#     # load data
#     logging.info("Load Data...............")
#     ipc = [[] for _ in range(classes_num)]
#     with h5py.File(index_path, "r") as f:
#         for i in tqdm(range(len(f["target"]))):
#             t_class = np.where(f["target"][i])[0]
#             for t in t_class:
#                 ipc[t].append(i)
#     print(ipc)
#     np.save(filename, ipc)
#     logging.info("Load Data Succeed...............")