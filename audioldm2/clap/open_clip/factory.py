"""
주로 모델 구성 및 생성, 사전 훈련된 가중치의 로딩, 모델 구성 파일의 관리 등을 담당하는 기능들을 포함

이 코드는 모델 구성 및 관리, 사전 훈련된 가중치의 로딩, 입력 데이터 전처리 설정 등 CLAP 모델을 사용하기 위한 다양한 기능을 제공. 
모델 구성 파일을 통해 모델 아키텍처를 유연하게 정의하고 관리할 수 있으며, 다양한 사전 훈련된 가중치를 적용하여 모델을 초기화할 수 있음.
"""

import json
import logging
import os
import re
from copy import deepcopy
from pathlib import Path

import torch

from .model import CLAP, convert_weights_to_fp16
from .openai import load_openai_model
from .pretrained import get_pretrained_url, download_pretrained
from .transform import image_transform

# 모델 구성 파일이 위치한 경로를 저장하는 리스트. Path(__file__).parent / "model_configs/"를 통해 현재 스크립트 파일과 동일한 디렉터리에 있는 model_configs 폴더를 참조
_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"] 
# 모델 이름과 해당 모델의 구성(config)을 매핑하는 딕셔너리. 모델 구성은 주로 json파일로 저장
_MODEL_CONFIGS = {}

"""
_natural_key 함수
문자열을 자연 정렬하기 위한 키를 생성. 숫자가 포함된 문자열을 정렬할 때 숫자 부분을 인식하여 자연스러운 순서로 정렬할 수 있게 도와줌
"""
def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]

"""
_rescan_model_configs 함수
_MODEL_CONFIG_PATHS에 지정된 경로들을 스캔하여 .json 확장자를 가진 모델 구성 파일들을 찾고, 이를 _MODEL_CONFIGS 딕셔너리에 로드. 
이 함수는 모델 구성을 동적으로 관리할 수 있게 해줌.
"""

def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        if os.path.basename(cf)[0] == ".":
            continue  # Ignore hidden files

        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "audio_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


_rescan_model_configs()  # initial populate of model config registry


"""
load_state_dict 함수
주어진 체크포인트 경로에서 모델의 상태 딕셔너리(state dict)를 로드하는 함수. skip_params 옵션을 통해 모듈 접두사를 제거할 수 있음.
"""

def load_state_dict(checkpoint_path: str, map_location="cpu", skip_params=True):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if skip_params:
        if next(iter(state_dict.items()))[0].startswith("module"):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
    # for k in state_dict:
    #     if k.startswith('transformer'):
    #         v = state_dict.pop(k)
    #         state_dict['text_branch.' + k[12:]] = v
    return state_dict

"""
create_model 함수
모델 이름, 사전 훈련된 가중치, 정밀도, 장치 등의 매개변수를 기반으로 CLAP 모델 인스턴스를 생성하고 초기화. 
사전 훈련된 모델을 로드하거나, 특정 구성에 맞게 모델을 조정할 수 있음.
enable_fusion과 fusion_type 매개변수를 통해 모델의 특징 융합 방식을 설정할 수 있음.
"""

def create_model(
    amodel_name: str,
    tmodel_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    openai_model_cache_dir: str = os.path.expanduser("~/.cache/clip"),
    skip_params=True,
    pretrained_audio: str = "",
    pretrained_text: str = "",
    enable_fusion: bool = False,
    fusion_type: str = "None"
    # pretrained_image: bool = False,
):
    amodel_name = amodel_name.replace(
        "/", "-"
    )  # for callers using old naming with / in ViT names
    pretrained_orig = pretrained
    pretrained = pretrained.lower()
    if pretrained == "openai":
        if amodel_name in _MODEL_CONFIGS:
            logging.info(f"Loading {amodel_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
        else:
            logging.error(
                f"Model config for {amodel_name} not found; available models {list_models()}."
            )
            raise RuntimeError(f"Model config for {amodel_name} not found.")

        logging.info(f"Loading pretrained ViT-B-16 text encoder from OpenAI.")
        # Hard Code in model name
        model_cfg["text_cfg"]["model_type"] = tmodel_name
        model = load_openai_model(
            "ViT-B-16",
            model_cfg,
            device=device,
            jit=jit,
            cache_dir=openai_model_cache_dir,
            enable_fusion=enable_fusion,
            fusion_type=fusion_type,
        )
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        if amodel_name in _MODEL_CONFIGS:
            logging.info(f"Loading {amodel_name} model config.")
            model_cfg = deepcopy(_MODEL_CONFIGS[amodel_name])
        else:
            logging.error(
                f"Model config for {amodel_name} not found; available models {list_models()}."
            )
            raise RuntimeError(f"Model config for {amodel_name} not found.")

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        # if pretrained_image:
        #     if 'timm_amodel_name' in model_cfg.get('vision_cfg', {}):
        #         # pretrained weight loading for timm models set via vision_cfg
        #         model_cfg['vision_cfg']['timm_model_pretrained'] = True
        #     else:
        #         assert False, 'pretrained image towers currently only supported for timm models'
        model_cfg["text_cfg"]["model_type"] = tmodel_name
        model_cfg["enable_fusion"] = enable_fusion
        model_cfg["fusion_type"] = fusion_type
        model = CLAP(**model_cfg)

        if pretrained:
            checkpoint_path = ""
            url = get_pretrained_url(amodel_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url, root=openai_model_cache_dir)
            elif os.path.exists(pretrained_orig):
                checkpoint_path = pretrained_orig
            if checkpoint_path:
                logging.info(
                    f"Loading pretrained {amodel_name}-{tmodel_name} weights ({pretrained})."
                )
                ckpt = load_state_dict(checkpoint_path, skip_params=True)
                model.load_state_dict(ckpt)
                param_names = [n for n, p in model.named_parameters()]
                # for n in param_names:
                #     print(n, "\t", "Loaded" if n in ckpt else "Unloaded")
            else:
                logging.warning(
                    f"Pretrained weights ({pretrained}) not found for model {amodel_name}."
                )
                raise RuntimeError(
                    f"Pretrained weights ({pretrained}) not found for model {amodel_name}."
                )

        if pretrained_audio:
            if amodel_name.startswith("PANN"):
                if "Cnn14_mAP" in pretrained_audio:  # official checkpoint
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")
                    audio_ckpt = audio_ckpt["model"]
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if (
                            "spectrogram_extractor" not in key
                            and "logmel_extractor" not in key
                        ):
                            v = audio_ckpt.pop(key)
                            audio_ckpt["audio_branch." + key] = v
                elif os.path.basename(pretrained_audio).startswith(
                    "PANN"
                ):  # checkpoint trained via HTSAT codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")
                    audio_ckpt = audio_ckpt["state_dict"]
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith("sed_model"):
                            v = audio_ckpt.pop(key)
                            audio_ckpt["audio_branch." + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith(
                    "finetuned"
                ):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")
                else:
                    raise ValueError("Unknown audio checkpoint")
            elif amodel_name.startswith("HTSAT"):
                if "HTSAT_AudioSet_Saved" in pretrained_audio:  # official checkpoint
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")
                    audio_ckpt = audio_ckpt["state_dict"]
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith("sed_model") and (
                            "spectrogram_extractor" not in key
                            and "logmel_extractor" not in key
                        ):
                            v = audio_ckpt.pop(key)
                            audio_ckpt["audio_branch." + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith(
                    "HTSAT"
                ):  # checkpoint trained via HTSAT codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")
                    audio_ckpt = audio_ckpt["state_dict"]
                    keys = list(audio_ckpt.keys())
                    for key in keys:
                        if key.startswith("sed_model"):
                            v = audio_ckpt.pop(key)
                            audio_ckpt["audio_branch." + key[10:]] = v
                elif os.path.basename(pretrained_audio).startswith(
                    "finetuned"
                ):  # checkpoint trained via linear probe codebase
                    audio_ckpt = torch.load(pretrained_audio, map_location="cpu")
                else:
                    raise ValueError("Unknown audio checkpoint")
            else:
                raise f"this audio encoder pretrained checkpoint is not support"

            model.load_state_dict(audio_ckpt, strict=False)
            logging.info(
                f"Loading pretrained {amodel_name} weights ({pretrained_audio})."
            )
            param_names = [n for n, p in model.named_parameters()]
            for n in param_names:
                print(n, "\t", "Loaded" if n in audio_ckpt else "Unloaded")

        model.to(device=device)
        if precision == "fp16":
            assert device.type != "cpu"
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model, model_cfg

"""
create_model_and_transforms 함수
모델과 데이터 전처리 변환(transform)을 생성. 
이 함수는 모델을 생성하고, 모델에 맞는 이미지 사이즈로 입력 이미지를 변환하는 전처리 파이프라인을 설정
"""

def create_model_and_transforms(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    # pretrained_image: bool = False,
):
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        jit,
        force_quick_gelu=force_quick_gelu,
        # pretrained_image=pretrained_image
    )
    preprocess_train = image_transform(model.visual.image_size, is_train=True)
    preprocess_val = image_transform(model.visual.image_size, is_train=False)
    return model, preprocess_train, preprocess_val

"""
list_models 함수
사용 가능한 모델 아키텍처를 나열. 이는 _MODEL_CONFIGS 딕셔너리에 로드된 모델 구성을 기반으로 함.
"""

def list_models():
    """enumerate available model architectures based on config files"""
    return list(_MODEL_CONFIGS.keys())

"""
add_model_config 함수
새로운 모델 구성 파일 또는 경로를 _MODEL_CONFIG_PATHS에 추가하고, _rescan_model_configs 함수를 호출하여 모델 구성을 갱신
"""

def add_model_config(path):
    """add model config path or file and update registry"""
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()
