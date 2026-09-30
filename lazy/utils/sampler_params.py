"""采样参数。"""

from dataclasses import dataclass


@dataclass(slots=True)
class SamplingParams:
    """单次生成请求的采样参数。

    Attributes:
        temperature: 采样温度，0 表示纯 greedy。
        max_tokens: 最多生成多少个 token。
        ignore_eos: 是否忽略 eos，跑满 max_tokens 才停。
    """

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        """校验参数合法性。"""
        assert self.temperature >= 0, "temperature must be non-negative"
