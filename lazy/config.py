"""引擎配置。

从 HuggingFace 模型目录读 config，并补上引擎自己需要的字段。
"""

from dataclasses import dataclass
import os

from transformers import AutoConfig

from lazy.utils.logger import init_logger

logger = init_logger(__name__)


@dataclass(slots=True)
class Config:
    """一次引擎启动用到的配置。

    Attributes:
        model: 本地模型目录（或者 HF 名字）。
        max_model_len: 单条序列允许的最大长度。
        tensor_parallel_size: 张量并行度，目前只支持 1。
        enforce_eager: 是否强制 eager 执行，跳过图优化。
        hf_config: 从模型目录加载出来的 HF config。
    """

    model: str
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: AutoConfig | None = None

    def __post_init__(self) -> None:
        """加载 HF config 并做基本校验。"""
        assert os.path.isdir(self.model)
        self.hf_config = AutoConfig.from_pretrained(self.model)
        logger.debug("loaded config from %s: %s", self.model, self.hf_config)
