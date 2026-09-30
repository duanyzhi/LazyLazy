"""序列对象：一条请求在引擎里的全部状态。"""

from copy import copy
from enum import auto
from enum import Enum
from itertools import count

from lazy.utils.sampler_params import SamplingParams


class SequenceStatus(Enum):
    """序列的生命周期状态。"""

    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    """一条生成请求。

    存着 prompt 和已生成的 token、当前状态，以及长度、缓存块这类记账
    信息。block table 目前还空着，留给后面的 paged attention 用。

    Attributes:
        BLOCK_SIZE: 单个 KV cache block 的 token 数。
    """

    BLOCK_SIZE = 256
    _counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams | None = None,
    ) -> None:
        """用 prompt token 和采样参数建一条序列。

        Args:
            token_ids: prompt 的 token id 列表。
            sampling_params: 采样参数，None 时用默认参数。
        """
        if sampling_params is None:
            sampling_params = SamplingParams()
        self.seq_id = next(Sequence._counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.num_scheduled_tokens = 0
        self.is_prefill = True
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        # hit_max / hit_eos，完成时由 scheduler 写入。
        self.finish_reason: str | None = None

    def __len__(self) -> int:
        return self.num_tokens

    def __getitem__(self, key) -> int:
        return self.token_ids[key]

    def append_token(self, token_id: int) -> None:
        """追加一个生成的 token，并同步长度和 last_token。"""
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
