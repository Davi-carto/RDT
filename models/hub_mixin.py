import os
from pathlib import Path
from typing import Dict, Optional, Union

from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.constants import (PYTORCH_WEIGHTS_NAME,
                                       SAFETENSORS_SINGLE_FILE)
from huggingface_hub.file_download import hf_hub_download
from huggingface_hub.utils import EntryNotFoundError, is_torch_available


if is_torch_available():
    import torch  # type: ignore


class CompatiblePyTorchModelHubMixin(PyTorchModelHubMixin):
    """Mixin class to load Pytorch models from the Hub."""
    def _save_pretrained(self, save_directory: Path) -> None:
        """Save weights from a Pytorch model to a local directory."""
        # To bypass saving into safetensor by default
        model_to_save = self.module if hasattr(self, "module") else self  # type: ignore
        torch.save(model_to_save.state_dict(), save_directory / PYTORCH_WEIGHTS_NAME)

    @classmethod
    def _from_pretrained(
        cls,
        *,
        model_id: str,          # 模型ID或本地路径
        revision: Optional[str], # 模型版本
        cache_dir: Optional[Union[str, Path]], # 缓存目录
        force_download: bool,    # 是否强制重新下载
        proxies: Optional[Dict], # 代理设置
        resume_download: Optional[bool], # 是否断点续传
        local_files_only: bool,  # 是否只使用本地文件
        token: Union[str, bool, None], # HuggingFace token
        map_location: str = "cpu", # 模型加载位置
        strict: bool = False,    # 是否严格加载权重
        **model_kwargs,          # 其他模型参数
    ):
        """从预训练权重加载PyTorch模型。"""
        # 使用传入的参数初始化模型
        model = cls(**model_kwargs)

        # 如果model_id是本地目录路径
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            try:
                # 首先尝试加载safetensors格式的权重文件
                model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except FileNotFoundError:
                # 如果找不到safetensors文件，则尝试加载pytorch格式的权重文件
                model_file = os.path.join(model_id, PYTORCH_WEIGHTS_NAME)
                return cls._load_as_pickle(model, model_file, map_location, strict)
        # 如果model_id是HuggingFace模型ID
        else:
            try:
                # 首先尝试从HuggingFace下载safetensors格式的权重文件
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_safetensor(model, model_file, map_location, strict)
            except EntryNotFoundError:
                # 如果找不到safetensors文件，则下载pytorch格式的权重文件
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=PYTORCH_WEIGHTS_NAME,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                return cls._load_as_pickle(model, model_file, map_location, strict)