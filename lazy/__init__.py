"""LazyLazy 包入口。

对外只暴露 LLM（引擎）和 SamplingParams（采样参数）两个符号。
"""

from lazy.llm import LLM
from lazy.utils.sampler_params import SamplingParams

__all__ = ["LLM", "SamplingParams"]
