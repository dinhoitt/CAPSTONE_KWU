import hashlib
import os
import urllib
import warnings

from tqdm import tqdm

_RN50 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
    cc12m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt",
)

_RN50_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt",
    cc12m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt",
)

_RN101 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt",
)

_RN101_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
    yfcc15m="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt",
)

_RN50x4 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
)

_RN50x16 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
)

_RN50x64 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt",
)

_VITB32 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    laion400m_e31="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
    laion400m_e32="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    laion400m_avg="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
)

_VITB32_quickgelu = dict(
    openai="https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
    laion400m_e31="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt",
    laion400m_e32="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt",
    laion400m_avg="https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_avg-8a00ab3c.pt",
)

_VITB16 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
)

_VITL14 = dict(
    openai="https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt",
)

_PRETRAINED = {
    "RN50": _RN50,
    "RN50-quickgelu": _RN50_quickgelu,
    "RN101": _RN101,
    "RN101-quickgelu": _RN101_quickgelu,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "ViT-B-32": _VITB32,
    "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "ViT-B-16": _VITB16,
    "ViT-L-14": _VITL14,
}

"""
list_pretrained
이 함수는 사용 가능한 사전 훈련된 모델의 목록을 반환. 모델 이름과 사전 훈련 태그(tag)를 조합한 문자열 또는 튜플 형태로 결과를 제공
"""

def list_pretrained(as_str: bool = False):
    """returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [
        ":".join([k, t]) if as_str else (k, t)
        for k in _PRETRAINED.keys()
        for t in _PRETRAINED[k].keys()
    ]

"""
list_pretrained_tag_models
특정 태그(tag)에 해당하는 모든 모델을 반환합. 예를 들어, openai 태그에 해당하는 모든 모델을 조회할 수 있음.
"""

def list_pretrained_tag_models(tag: str):
    """return all models having the specified pretrain tag"""
    models = []
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models

"""
list_pretrained_model_tags
지정된 모델 아키텍처에 대해 사용 가능한 모든 사전 훈련 태그를 반환.
"""

def list_pretrained_model_tags(model: str):
    """return all pretrain tags for the specified model architecture"""
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags

"""
get_pretrained_url
지정된 모델과 태그에 해당하는 사전 훈련된 모델의 URL 주소를 반환.
"""

def get_pretrained_url(model: str, tag: str):
    if model not in _PRETRAINED:
        return ""
    model_pretrained = _PRETRAINED[model]
    if tag not in model_pretrained:
        return ""
    return model_pretrained[tag]

"""
download_pretrained
지정된 URL에서 사전 훈련된 모델을 다운로드합니다. 다운로드가 완료되면 해당 파일의 경로를 반환. 
파일이 이미 존재하고 SHA256 체크섬이 일치하는 경우 다운로드를 건너뛰고 기존 파일의 경로를 반환. 
SHA256 체크섬이 제공되지 않은 경우 파일의 존재 여부만 확인.
"""

def download_pretrained(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    if "openaipublic" in url:
        expected_sha256 = url.split("/")[-2]
    else:
        expected_sha256 = ""

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if (
                hashlib.sha256(open(download_target, "rb").read()).hexdigest()
                == expected_sha256
            ):
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file"
                )
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(
            total=int(source.info().get("Content-Length")),
            ncols=80,
            unit="iB",
            unit_scale=True,
        ) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if (
        expected_sha256
        and hashlib.sha256(open(download_target, "rb").read()).hexdigest()
        != expected_sha256
    ):
        raise RuntimeError(
            f"Model has been downloaded but the SHA256 checksum does not not match"
        )

    return download_target
