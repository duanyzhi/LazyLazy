"""调度器：每步决定跑 prefill 还是 decode。"""

from collections import deque

import nvtx

from lazy.config import Config
from lazy.engine.sequence import Sequence
from lazy.engine.sequence import SequenceStatus
from lazy.utils.logger import init_logger

logger = init_logger(__name__)


class Scheduler:
    """请求队列加每步调度。

    维护 waiting / running / finished 三个队列：schedule 返回本步要跑的
    序列和阶段，postprocess 负责判停和收尾。
    """

    def __init__(self, config: Config, eos_token_id: int | None = None) -> None:
        """初始化三个队列。

        Args:
            config: 引擎配置。
            eos_token_id: eos 的 token id，命中就停。
        """
        self.eos_token_id = eos_token_id
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self.finished: list[Sequence] = []

    def is_finished(self) -> bool:
        """waiting 和 running 都空就算跑完了。"""
        return not self.waiting and not self.running

    def add(self, seq: Sequence) -> None:
        """把新序列放进 waiting 队列。"""
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        """挑出这一步要跑的序列。

        prefill 一次只取一条：多条打平成一个 batch 时 causal mask 会互相
        看见。waiting 空了才切到 decode。

        Returns:
            (本步的序列列表, 是否是 prefill 阶段)。
        """
        if self.waiting:
            with nvtx.annotate("prefill_scheduler", color="red"):
                seq = self.waiting.popleft()
                seq.status = SequenceStatus.RUNNING
                self.running.append(seq)
                return [seq], True

        with nvtx.annotate("decode_scheduler", color="red"):
            scheduled_seqs = list(self.running)
            for seq in scheduled_seqs:
                seq.is_prefill = False
            return scheduled_seqs, False

    def postprocess(
        self,
        seqs: list[Sequence],
        output_token_ids: list[int],
        is_prefill: bool,
    ) -> list[Sequence]:
        """回收本步输出 token、判停，返回跑完的序列。

        Args:
            seqs: 本步跑的序列。
            output_token_ids: 与 seqs 一一对应的新 token id。
            is_prefill: 本步是否是 prefill 阶段。

        Returns:
            本步跑完（hit_max 或 hit_eos）的序列列表。
        """
        finished = []
        for seq, token_id in zip(seqs, output_token_ids):
            seq.append_token(token_id)
            hit_max = seq.num_tokens - seq.num_prompt_tokens >= seq.max_tokens
            hit_eos = (not seq.ignore_eos and self.eos_token_id is not None and
                       token_id == self.eos_token_id)
            if hit_max or hit_eos:
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)
                self.finished.append(seq)
                seq.finish_reason = "hit_max" if hit_max else "hit_eos"
                finished.append(seq)
                logger.info(
                    "seq %d finished (%s), generated %d tokens",
                    seq.seq_id,
                    seq.finish_reason,
                    seq.num_tokens - seq.num_prompt_tokens,
                )
        return finished
